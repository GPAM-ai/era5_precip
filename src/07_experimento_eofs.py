#!/usr/bin/env python
"""
Etapa 07 - Experimento: quantos modos e quanta memoria?

MOTIVACAO (de 05):
  As features de grande escala respondem por 56% do ganho do modelo, e
  paramos em 8 componentes por grupo e defasagens de ate 3 meses. Duas
  perguntas em aberto:
    - mais componentes ajudam, ou os modos altos sao ruido?
    - o ENOS tem memoria de 6 a 12 meses; defasagens curtas desperdicam isso?

  E `a_cloud_cover` eh o melhor preditor local isolado (5.2% do ganho) mas
  hoje entra apenas como valor no ponto, sem decomposicao propria. Ganha
  um grupo de EOF so seu.

NOTA DE DESEMPENHO - por que X_fix eh carregado inteiro:
  A primeira versao deste script indexava o memmap diretamente:
  `X[idx, :18]` com 1.5 milhao de indices aleatorios sobre um arquivo de
  7.8 GB. Em Lustre isso eh o pior padrao de acesso possivel - cada
  acesso vira uma requisicao ao sistema de arquivos distribuido. Medido:
  30 minutos de relogio para 27 segundos de CPU, com o processo parado em
  espera de I/O.
  A correcao eh ler as 18 colunas fixas de uma vez (627 MB, sequencial,
  poucos segundos) e manter em RAM. Toda indexacao aleatoria passa a
  acontecer na memoria.

DESENHO DO EXPERIMENTO:
  As colunas fixas (estaticas, sazonais, anomalias locais - 18 no total)
  sao identicas nas tres configuracoes. So o bloco de PCs muda, o que
  isola exatamente a variavel de interesse.

DEFASAGENS LONGAS SO NOS PRIMEIROS MODOS:
  16 componentes x 6 grupos x 6 defasagens dariam ~780 colunas, inviavel.
  A restricao tambem eh fisicamente defensavel: PC1-PC3 de cada grupo
  carregam a configuracao de grande escala, que tem memoria de meses;
  PC14 eh ruido e sua defasagem de 12 meses nao significa nada.
    lags 1, 2, 3   -> todos os componentes
    lags 6, 9, 12  -> apenas os 3 primeiros de cada grupo + idx_pacifico
    tendencias     -> apenas os 3 primeiros

Uso:
    python src/07_experimento_eofs.py
"""

import json
import gc
import time

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from sklearn.decomposition import PCA

SAIDA = "dados_processados"

CONFIGS = (8, 12, 16)
LAGS_CURTOS = (1, 2, 3)
LAGS_LONGOS = (6, 9, 12)
N_MODOS_LONGOS = 3
MAX_LAG = max(LAGS_LONGOS)

CORTES = [2007, 2012, 2017]
N_AJUSTE = 1_500_000
SEMENTE = 42

PARAMS = dict(objective="l2", num_leaves=63, learning_rate=0.05,
              min_child_samples=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
              n_estimators=300, n_jobs=-1, verbose=-1, random_state=SEMENTE)

GRUPOS = {
    "slp": ["a_surface_pressure"],
    "z850": ["a_geopotential_850"],
    "circ": ["a_u_850", "a_v_850"],
    "umid": ["a_shum_850", "a_rel_hum_850"],
    "fluxo": ["a_qu", "a_qv", "a_conv_umid"],
    "nuv": ["a_cloud_cover"],
}


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def mem():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return int(l.split()[1]) / 1024
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

with open(f"{SAIDA}/features.json") as f:
    spec = json.load(f)
N_FIXAS = 6 + len(spec["grupos"]["locais"])

t0 = time.time()
Xmm = np.load(f"{SAIDA}/X_treino.npy", mmap_mode="r")
print(f"  X no disco: {Xmm.shape}")
print(f"  lendo as {N_FIXAS} colunas fixas para a RAM (leitura sequencial)...",
      flush=True)
X_fix = np.ascontiguousarray(Xmm[:, :N_FIXAS])
del Xmm
gc.collect()
print(f"  X_fix: {X_fix.shape}  {X_fix.nbytes/1e9:.2f} GB  "
      f"em {time.time()-t0:.1f}s  [{mem():.0f} MB]")

y = np.load(f"{SAIDA}/y_treino.npy")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
anos = meta["ano_feature"].values

tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_pt = tr_ds.sizes["lat"] * tr_ds.sizes["lon"]
n_blocos = len(y) // n_pt
tr_ds.close()

IDX_T0 = 3
blocos_descartar = MAX_LAG - IDX_T0
inicio = blocos_descartar * n_pt
tempo_do_bloco = np.arange(IDX_T0, IDX_T0 + n_blocos)

print(f"\n  blocos descartados para alinhar as defasagens longas: "
      f"{blocos_descartar} ({inicio:,} de {len(y):,} linhas)")
print("  (todas as configuracoes sao avaliadas sobre as MESMAS linhas)")


# ---------------------------------------------------------------------------
sep("2. CAMPOS PARA DECOMPOSICAO")

campos = xr.open_dataset(f"{SAIDA}/campos_eof.nc").load()
n_t = campos.sizes["time"]
peso = np.sqrt(np.cos(np.deg2rad(campos.lat))).astype("float32")
print(f"  {dict(campos.sizes)}, {len(campos.data_vars)} variaveis")
print(f"  grupos: {', '.join(GRUPOS)}  ('nuv' eh novo)")


def blocos_pca(n_pc):
    series = {}
    for nome, variaveis in GRUPOS.items():
        colunas = []
        for v in variaveis:
            A = (campos[v] * peso).values.reshape(n_t, -1).astype("float32")
            colunas.append((A - A.mean(0)) / (A.std(0) + 1e-9))
        A = np.nan_to_num(np.concatenate(colunas, axis=1))

        pca = PCA(n_components=n_pc, random_state=SEMENTE)
        Y = pca.fit_transform(A)
        Y = Y / (Y.std(0) + 1e-9)
        for i in range(n_pc):
            series[f"pc_{nome}_{i+1}"] = Y[:, i].astype("float32")
        del A, Y
        gc.collect()

    p = (campos["a_surface_pressure"]
         .sel(lat=slice(-20, 0), lon=slice(-90, -80))
         .mean(dim=("lat", "lon")).values)
    series["idx_pacifico"] = ((p - p.mean()) / p.std()).astype("float32")
    return series


def tabela_pcs(series):
    df = pd.DataFrame(series)
    base = list(df.columns)
    longos = [f"pc_{g}_{i+1}" for g in GRUPOS for i in range(N_MODOS_LONGOS)]
    longos.append("idx_pacifico")

    extras = {}
    for lag in LAGS_CURTOS:
        for c in base:
            extras[f"{c}_lag{lag}"] = df[c].shift(lag)
    for lag in LAGS_LONGOS:
        for c in longos:
            extras[f"{c}_lag{lag}"] = df[c].shift(lag)
    for c in longos:
        extras[f"{c}_tend"] = df[c] - df[c].shift(3)
        extras[f"{c}_tend12"] = df[c] - df[c].shift(12)

    return pd.concat([df, pd.DataFrame(extras)], axis=1).astype("float32")


# ---------------------------------------------------------------------------
sep("3. VALIDACAO POR CONFIGURACAO")

rng = np.random.default_rng(SEMENTE)
resultados = []

for n_pc in CONFIGS:
    t_cfg = time.time()
    series = blocos_pca(n_pc)
    tab = tabela_pcs(series)
    valores = tab.values[tempo_do_bloco]
    n_col_pc = valores.shape[1]

    assert not np.isnan(valores[blocos_descartar:]).any(), \
        "ha NaN nos PCs apos o descarte inicial"

    print(f"\n  --- {n_pc} componentes: {n_col_pc} colunas de PC, "
          f"{N_FIXAS + n_col_pc} no total", flush=True)

    linhas = []
    for corte in CORTES:
        t_fold = time.time()
        v_ini, v_fim = corte + 1, corte + 5

        m_fit = anos <= corte
        m_val = (anos >= v_ini) & (anos <= v_fim)
        m_fit[:inicio] = False
        m_val[:inicio] = False

        idx_fit = np.where(m_fit)[0]
        if len(idx_fit) > N_AJUSTE:
            idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
        idx_val = np.where(m_val)[0]

        # X_fix ja esta em RAM: indexacao aleatoria custa microssegundos
        Xf = np.hstack([X_fix[idx_fit], valores[idx_fit // n_pt]]).astype("float32")
        Xv = np.hstack([X_fix[idx_val], valores[idx_val // n_pt]]).astype("float32")

        mod = lgb.LGBMRegressor(**PARAMS)
        mod.fit(Xf, y[idx_fit])
        pred = mod.predict(Xv).astype("float32")

        yv = y[idx_val]
        r_clim = float(np.sqrt(np.mean(yv ** 2)))
        a = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
        r_mod = float(np.sqrt(np.mean((yv - a * pred) ** 2)))
        corr = float(np.corrcoef(pred, yv)[0, 1])

        linhas.append(dict(corte=corte, ganho=100 * (1 - r_mod / r_clim),
                           corr=corr, alfa=a))
        print(f"    fold ate {corte}: ganho {linhas[-1]['ganho']:+.3f}%  "
              f"r {corr:.3f}  alfa {a:.3f}  "
              f"({time.time()-t_fold:.0f}s)  [{mem():.0f} MB]", flush=True)

        del Xf, Xv, mod, pred
        gc.collect()

    d = pd.DataFrame(linhas)
    resultados.append(dict(n_pc=n_pc, n_colunas=N_FIXAS + n_col_pc,
                           ganho=d.ganho.mean(), desvio=d.ganho.std(),
                           corr=d["corr"].mean(), alfa=d.alfa.mean()))
    print(f"    media: ganho {d.ganho.mean():+.3f}%  r {d['corr'].mean():.3f}"
          f"  (configuracao em {time.time()-t_cfg:.0f}s)")

    del series, tab, valores
    gc.collect()


# ---------------------------------------------------------------------------
sep("4. COMPARACAO")

res = pd.DataFrame(resultados)
print(f"  {'n_pc':>6s} {'colunas':>9s} {'ganho':>9s} {'desvio':>8s} "
      f"{'r':>7s} {'alfa':>7s}")
for _, r in res.iterrows():
    print(f"  {int(r.n_pc):6d} {int(r.n_colunas):9d} {r.ganho:+8.3f}% "
          f"{r.desvio:8.3f} {r['corr']:7.3f} {r.alfa:7.3f}")

print("""
  REFERENCIA - configuracao em producao (8 componentes, lags 1-3, sem
  grupo de nuvens, 223 colunas): ganho +1.991%, r 0.197, medidos em 05
  com 5 folds. Aqui sao 3 folds e menos linhas de ajuste, entao os
  numeros absolutos nao sao comparaveis - vale a comparacao ENTRE as
  linhas acima.""")

melhor = res.loc[res.ganho.idxmax()]
delta = res.ganho.max() - res.ganho.min()
print(f"\n  >> melhor: {int(melhor.n_pc)} componentes "
      f"({int(melhor.n_colunas)} colunas), ganho {melhor.ganho:+.3f}%")
print(f"  amplitude entre configuracoes: {delta:.3f} pontos percentuais")

if delta < 0.15:
    print("""
  >> as tres praticamente empatam. Criterio definido ANTES de ver o
     resultado: escolher a MENOR. Menos colunas treina mais rapido,
     generaliza melhor e eh mais facil de documentar. Evidencia de que
     os modos altos sao ruido.""")
else:
    print(f"""
  >> ha diferenca real. Adotar {int(melhor.n_pc)} componentes e regerar
     03 e 04 com essa configuracao.""")

res.to_csv(f"{SAIDA}/experimento_eofs.csv", index=False)
print(f"\n  salvo: {SAIDA}/experimento_eofs.csv")