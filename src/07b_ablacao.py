#!/usr/bin/env python
"""
Etapa 07b - Ablacao: o que de fato ajuda?

O CONFUNDIMENTO DE 07:
  O experimento comparou 8, 12 e 16 componentes e concluiu corretamente
  que os modos altos sao ruido (r cai de 0.160 para 0.153). Mas ao
  comparar com a producao, tres coisas haviam mudado ao mesmo tempo:
    - features: +grupo de nuvens, +defasagens de 6/9/12 meses
    - linhas de ajuste: 2.5M -> 1.5M
    - arvores: 400 -> 300
  A configuracao nova ficou pior nos mesmos folds (+1.33% contra +1.96%),
  mas nao da para saber se a culpa eh da diluicao de features ou do
  orcamento de treino menor.

ESTE SCRIPT ISOLA CADA MUDANCA:
    A  base        8 componentes, lags 1-3, 5 grupos      (= producao)
    B  A + nuvens  acrescenta o grupo de EOF de cloud_cover
    C  A + memoria acrescenta lags 6, 9, 12 e tendencia de 12 meses
    D  B + C       as duas mudancas juntas                (= 07)

  Tudo com o MESMO orcamento de treino de 05 (2.5M linhas, 400 arvores)
  e os MESMOS folds, para que a unica diferenca seja o conjunto de
  features. As linhas avaliadas sao identicas nas quatro variantes.

Uso:
    python src/07b_ablacao.py
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

N_PC = 8                      # decidido em 07
LAGS_CURTOS = (1, 2, 3)
LAGS_LONGOS = (6, 9, 12)
N_MODOS_LONGOS = 3
MAX_LAG = 12                  # descarte fixo, igual para todas as variantes

CORTES = [2007, 2012, 2017]
N_AJUSTE = 2_500_000          # igual a 05
SEMENTE = 42

PARAMS = dict(objective="l2", num_leaves=63, learning_rate=0.05,
              min_child_samples=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
              n_estimators=400, n_jobs=-1, verbose=-1, random_state=SEMENTE)

GRUPOS_BASE = {
    "slp": ["a_surface_pressure"],
    "z850": ["a_geopotential_850"],
    "circ": ["a_u_850", "a_v_850"],
    "umid": ["a_shum_850", "a_rel_hum_850"],
    "fluxo": ["a_qu", "a_qv", "a_conv_umid"],
}
GRUPO_NUVENS = {"nuv": ["a_cloud_cover"]}

VARIANTES = {
    "A base":        dict(nuvens=False, memoria=False),
    "B +nuvens":     dict(nuvens=True,  memoria=False),
    "C +memoria":    dict(nuvens=False, memoria=True),
    "D +ambos":      dict(nuvens=True,  memoria=True),
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
X_fix = np.ascontiguousarray(Xmm[:, :N_FIXAS])     # leitura sequencial
del Xmm
gc.collect()
print(f"  X_fix: {X_fix.shape}  {X_fix.nbytes/1e9:.2f} GB  "
      f"em {time.time()-t0:.1f}s")

y = np.load(f"{SAIDA}/y_treino.npy")
anos = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")["ano_feature"].values

tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_pt = tr_ds.sizes["lat"] * tr_ds.sizes["lon"]
tr_ds.close()
n_blocos = len(y) // n_pt

IDX_T0 = 3
inicio = (MAX_LAG - IDX_T0) * n_pt
tempo_do_bloco = np.arange(IDX_T0, IDX_T0 + n_blocos)
print(f"  descarte inicial fixo: {inicio:,} linhas "
      "(identico nas quatro variantes)")

campos = xr.open_dataset(f"{SAIDA}/campos_eof.nc").load()
n_t = campos.sizes["time"]
peso = np.sqrt(np.cos(np.deg2rad(campos.lat))).astype("float32")


# ---------------------------------------------------------------------------
sep("2. MONTANDO OS BLOCOS DE FEATURES")


def series_de(grupos):
    out = {}
    for nome, variaveis in grupos.items():
        cols = []
        for v in variaveis:
            A = (campos[v] * peso).values.reshape(n_t, -1).astype("float32")
            cols.append((A - A.mean(0)) / (A.std(0) + 1e-9))
        A = np.nan_to_num(np.concatenate(cols, axis=1))
        pca = PCA(n_components=N_PC, random_state=SEMENTE)
        Y = pca.fit_transform(A)
        Y = Y / (Y.std(0) + 1e-9)
        for i in range(N_PC):
            out[f"pc_{nome}_{i+1}"] = Y[:, i].astype("float32")
        del A, Y
        gc.collect()
    return out


base = series_de(GRUPOS_BASE)
nuvens = series_de(GRUPO_NUVENS)

p = (campos["a_surface_pressure"].sel(lat=slice(-20, 0), lon=slice(-90, -80))
     .mean(dim=("lat", "lon")).values)
base["idx_pacifico"] = ((p - p.mean()) / p.std()).astype("float32")

print(f"  grupos base : {len(base)} series")
print(f"  grupo nuvens: {len(nuvens)} series")


def montar_tabela(usa_nuvens, usa_memoria):
    series = dict(base)
    grupos_ativos = list(GRUPOS_BASE)
    if usa_nuvens:
        series.update(nuvens)
        grupos_ativos.append("nuv")

    df = pd.DataFrame(series)
    cols = list(df.columns)
    extras = {}

    for lag in LAGS_CURTOS:
        for c in cols:
            extras[f"{c}_lag{lag}"] = df[c].shift(lag)

    if usa_memoria:
        longos = [f"pc_{g}_{i+1}" for g in grupos_ativos
                  for i in range(N_MODOS_LONGOS)]
        longos.append("idx_pacifico")
        for lag in LAGS_LONGOS:
            for c in longos:
                extras[f"{c}_lag{lag}"] = df[c].shift(lag)
        for c in longos:
            extras[f"{c}_tend12"] = df[c] - df[c].shift(12)

    for c in cols:
        extras[f"{c}_tend"] = df[c] - df[c].shift(3)

    return pd.concat([df, pd.DataFrame(extras)], axis=1).astype("float32")


# ---------------------------------------------------------------------------
sep("3. ABLACAO")

rng = np.random.default_rng(SEMENTE)
resultados = []

for rotulo, cfg in VARIANTES.items():
    tab = montar_tabela(cfg["nuvens"], cfg["memoria"])
    valores = tab.values[tempo_do_bloco]
    n_col = valores.shape[1]
    assert not np.isnan(valores[(MAX_LAG - IDX_T0):]).any()

    print(f"\n  --- {rotulo}: {n_col} colunas de PC, "
          f"{N_FIXAS + n_col} no total", flush=True)

    linhas = []
    for corte in CORTES:
        t_fold = time.time()
        m_fit = anos <= corte
        m_val = (anos >= corte + 1) & (anos <= corte + 5)
        m_fit[:inicio] = False
        m_val[:inicio] = False

        idx_fit = np.where(m_fit)[0]
        if len(idx_fit) > N_AJUSTE:
            idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
        idx_val = np.where(m_val)[0]

        Xf = np.hstack([X_fix[idx_fit], valores[idx_fit // n_pt]])
        Xv = np.hstack([X_fix[idx_val], valores[idx_val // n_pt]])

        mod = lgb.LGBMRegressor(**PARAMS)
        mod.fit(Xf.astype("float32"), y[idx_fit])
        pred = mod.predict(Xv.astype("float32")).astype("float32")

        yv = y[idx_val]
        r_clim = float(np.sqrt(np.mean(yv ** 2)))
        a = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
        r_mod = float(np.sqrt(np.mean((yv - a * pred) ** 2)))

        linhas.append(dict(corte=corte, ganho=100 * (1 - r_mod / r_clim),
                           corr=float(np.corrcoef(pred, yv)[0, 1]), alfa=a))
        print(f"    fold ate {corte}: ganho {linhas[-1]['ganho']:+.3f}%  "
              f"r {linhas[-1]['corr']:.3f}  alfa {a:.3f}  "
              f"({time.time()-t_fold:.0f}s)", flush=True)

        del Xf, Xv, mod, pred
        gc.collect()

    d = pd.DataFrame(linhas)
    resultados.append(dict(variante=rotulo, colunas=N_FIXAS + n_col,
                           ganho=d.ganho.mean(), desvio=d.ganho.std(),
                           corr=d["corr"].mean(), alfa=d.alfa.mean()))
    print(f"    media: ganho {d.ganho.mean():+.3f}%  r {d['corr'].mean():.3f}")

    del tab, valores
    gc.collect()


# ---------------------------------------------------------------------------
sep("4. LEITURA")

res = pd.DataFrame(resultados)
print(f"  {'variante':14s} {'colunas':>9s} {'ganho':>9s} {'desvio':>8s} "
      f"{'r':>7s} {'alfa':>7s}")
for _, r in res.iterrows():
    print(f"  {r.variante:14s} {int(r.colunas):9d} {r.ganho:+8.3f}% "
          f"{r.desvio:8.3f} {r['corr']:7.3f} {r.alfa:7.3f}")

g = dict(zip(res.variante, res.ganho))
print(f"""
  Efeitos isolados:
    grupo de nuvens : {g['B +nuvens'] - g['A base']:+.3f} pontos percentuais
    memoria longa   : {g['C +memoria'] - g['A base']:+.3f}
    as duas juntas  : {g['D +ambos'] - g['A base']:+.3f}
    soma dos efeitos: {(g['B +nuvens'] - g['A base']) + (g['C +memoria'] - g['A base']):+.3f}

  REFERENCIA: 05 mediu +1.96% nestes mesmos tres folds com a
  configuracao de producao. A variante A deve reproduzir isso de perto -
  se nao reproduzir, ha diferenca residual de desenho a investigar.""")

melhor = res.loc[res.ganho.idxmax()]
print(f"\n  >> melhor: {melhor.variante} ({int(melhor.colunas)} colunas), "
      f"ganho {melhor.ganho:+.3f}%")

if melhor.ganho - g["A base"] < 0.1:
    print("""
  >> nenhuma das adicoes paga o proprio custo. Manter a configuracao de
     producao e registrar a ablacao no README: testamos memoria longa e
     decomposicao de nuvens, e nenhuma ajudou.""")
else:
    print(f"""
  >> a adicao compensa. Regerar 03 e 04 com essa configuracao e refazer
     05 e 06.""")

res.to_csv(f"{SAIDA}/ablacao_features.csv", index=False)
print(f"\n  salvo: {SAIDA}/ablacao_features.csv")