#!/usr/bin/env python
"""
Etapa 06 - Modelo sobre a climatologia v3, com peso por recencia.

CONFIGURACAO FINAL, e de onde veio cada numero:

  climatologia v3 (01d)
    janela curta de 24 anos, mistura 0.50 com o periodo completo,
    tendencia local com encolhimento k=0.15. Busca conjunta validada em
    tres cortes temporais independentes, com plato de amplitude 0.0019
    em volta do otimo.

  peso amostral por recencia (08, parte 1)
    w = exp(-(2022 - ano) / 40). Melhorou em TODOS os cinco folds, com
    curva bem comportada: uniforme +1.881%, tau=60 +2.024%,
    tau=40 +2.063%, tau=25 +2.022%, tau=15 +1.924%.
    Com tau=40, 1940 pesa cerca de 12% do que 2022 pesa - reduz a
    influencia do periodo pre-1979 (reanalise retroativa, com pouca
    observacao assimilada sobre a America do Sul) sem descarta-lo.

  modelo unico, sem ensemble (08, parte 2)
    LightGBM, Ridge e MLP rasa foram comparados nos mesmos folds. O
    LightGBM domina nos cinco (r de 0.158 a 0.243, contra 0.093 a 0.168
    do Ridge e 0.043 a 0.168 da MLP). Os pesos otimos do ensemble ficam
    em 0.90-1.03 para o LightGBM e quase zero para os demais. Com os
    pesos validados cruzadamente entre folds, o ensemble marca +2.124%
    contra +2.134% do LightGBM sozinho: marginalmente PIOR. Nao adotado.

  8 componentes por grupo, lags 1-3, sem grupo de nuvens (07, 07b)
    12 e 16 componentes nao ajudam (r cai de 0.160 para 0.153). Ablacao
    controlada mostrou que decompor cloud_cover em EOF (-0.100 p.p.) e
    acrescentar defasagens de 6/9/12 meses (-0.045 p.p.) tambem nao
    ajudam. Mantida a configuracao mais simples.

SEM VAZAMENTO:
  A climatologia v3 eh reajustada dentro de cada fold da validacao,
  usando apenas anos anteriores ao corte. Sem isso, o baseline enxerga
  os anos que esta sendo avaliado a prever.

Uso:
    python src/06_submissao.py
"""

import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

DADOS = "dados_brutos"
SAIDA = "dados_processados"
SUBS = "saidas"

N_JANELA, W_MIX, K_TEND = 24, 0.50, 0.15      # 01d
# tau=40 melhorou todos os 5 folds do CV (08), mas o placar publico de 2023
# piorou com ele. RECENCIA=0 desliga, para testar a hipotese de que o CV
# nao representa o regime de 2023/24.
TAU = 40 if os.environ.get("RECENCIA", "1") == "1" else None
ANO_REF = 2022

CORTES = [1997, 2002, 2007, 2012, 2017]
N_AJUSTE = 2_500_000
SEMENTE = 42

PARAMS = dict(objective="l2",
              num_leaves=int(os.environ.get("NUM_LEAVES", 63)),
              learning_rate=float(os.environ.get("LR", 0.05)),
              min_child_samples=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
              n_estimators=int(os.environ.get("N_ARVORES", 400)),
              n_jobs=-1, verbose=-1, random_state=SEMENTE)
# Com o MME o sinal ficou forte (r 0.44) e o alfa passou de 1: o modelo
# subestima a amplitude. Capacidade passa a importar. Testar via
#   N_ARVORES=1000 NUM_LEAVES=127 FEATURES=mme2 python src/06_submissao.py


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


def rmse(a, b=None):
    return float(np.sqrt(np.mean((a if b is None else a - b) ** 2)))


def pesos(anos_vec):
    """Peso exponencial por recencia, normalizado para media 1.
    Devolve None (peso uniforme) quando RECENCIA=0."""
    if TAU is None:
        return None
    w = np.exp(-(ANO_REF - anos_vec) / TAU).astype("float64")
    return w / w.mean()


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

t0 = time.time()
# FEATURES escolhe o conjunto de matrizes:
#   (vazio)  -> X_treino.npy          original
#   sst      -> X_treino_sst.npy      + indices NOAA (09)
#   grade    -> X_treino_grade.npy    + indices + EOFs de TSM em grade (11)
# USAR_SST=1 eh mantido como sinonimo de FEATURES=sst
FEATURES = os.environ.get("FEATURES", "")
if not FEATURES and os.environ.get("USAR_SST", "0") == "1":
    FEATURES = "sst"
USAR_SST = FEATURES != ""
sufixo = f"_{FEATURES}" if FEATURES else ""
arq_feat = f"features{sufixo}.json"
print(f"  variante: {FEATURES or 'original'}")

# sem mmap_mode: leitura sequencial de 7.8 GB leva segundos, enquanto
# indexacao aleatoria sobre memmap em Lustre leva dezenas de minutos
X = np.load(f"{SAIDA}/X_treino{sufixo}.npy")
print(f"  X em RAM: {X.shape}  {X.nbytes/1e9:.2f} GB  "
      f"em {time.time()-t0:.0f}s  [RSS {mem():.0f} MB]")
assert mem() > 7000, "X nao foi carregado para a RAM"

Xte = np.load(f"{SAIDA}/X_teste{sufixo}.npy")
y_anom = np.load(f"{SAIDA}/y_treino.npy")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")

with open(f"{SAIDA}/{arq_feat}") as f:
    nomes = json.load(f)["nomes"]

anos = meta["ano_feature"].values
mes_alvo = meta["mes_alvo"].values
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)

print(f"  teste: {Xte.shape}")
if TAU is None:
    print("  peso por recencia: DESLIGADO (uniforme)")
else:
    print(f"  peso por recencia: tau={TAU}  "
          f"(1940 pesa {np.exp(-(ANO_REF-1940)/TAU):.3f} do peso de {ANO_REF})")


# ---------------------------------------------------------------------------
sep("2. RECONSTRUINDO O ALVO BRUTO")

tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
n_pt = n_lat * n_lon
stride = (len(tp.lat) - 1) // (n_lat - 1)
tr_ds.close()

clim_g = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")
clim_arr = (clim_g.isel(lat=slice(None, None, stride),
                        lon=slice(None, None, stride))
            .transpose("month", "lat", "lon").values.reshape(12, n_pt))

n_t = len(y_anom) // n_pt
ipt = np.tile(np.arange(n_pt, dtype="int32"), n_t)
y_bruto = y_anom + clim_arr[mes_alvo - 1, ipt]
tp_sub = tp.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))

print(f"  alvo bruto: media {y_bruto.mean():.4f} mm/dia "
      f"(tp global era 3.52)")


def v3_do_fold(corte):
    """Nivel, inclinacao e ano central da climatologia v3 ate o corte."""
    def media(ini, fim):
        return (tp_sub.sel(time=slice(f"{ini}-01-01", f"{fim}-12-31"))
                .groupby("time.month").mean("time")
                .transpose("month", "lat", "lon").values.reshape(12, n_pt))

    ini = max(1940, corte - N_JANELA + 1)
    nivel = (1 - W_MIX) * media(1940, corte) + W_MIX * media(ini, corte)

    sub = tp_sub.sel(time=slice(f"{ini}-01-01", f"{corte}-12-31"))
    incls = []
    for m in range(1, 13):
        sm = sub.sel(time=sub.time.dt.month == m)
        x = sm.time.dt.year.values.astype("float64")
        xc = x - x.mean()
        v = sm.values.reshape(len(x), -1)
        incls.append(np.tensordot(xc, v - v.mean(0), axes=(0, 0))
                     / (xc ** 2).sum())
    return nivel, np.stack(incls), float(x.mean())


def anomalia_v3(nivel, incl, centro):
    base = (nivel[mes_alvo - 1, ipt]
            + K_TEND * incl[mes_alvo - 1, ipt] * (ano_alvo - centro))
    return (y_bruto - base).astype("float32"), base.astype("float32")


# ---------------------------------------------------------------------------
sep("3. VALIDACAO WALK-FORWARD")

print("""  Cada fold reajusta a climatologia v3 com anos anteriores ao corte
  e aplica o peso por recencia no treino, para que o alfa medido
  corresponda ao modelo que sera de fato submetido.
""")

rng = np.random.default_rng(SEMENTE)
linhas = []

for corte in CORTES:
    t_fold = time.time()
    v_ini, v_fim = corte + 1, corte + 5

    # baseline de referencia: climatologia de media simples
    m_fit = anos <= corte
    soma = np.zeros((12, n_pt))
    cont = np.zeros((12, n_pt))
    np.add.at(soma, (mes_alvo[m_fit] - 1, ipt[m_fit]), y_bruto[m_fit])
    np.add.at(cont, (mes_alvo[m_fit] - 1, ipt[m_fit]), 1)
    y_simples = y_bruto - (soma / np.maximum(cont, 1))[mes_alvo - 1, ipt]

    nivel, incl, centro = v3_do_fold(corte)
    y_v3, _ = anomalia_v3(nivel, incl, centro)

    m_val = (anos >= v_ini) & (anos <= v_fim)
    idx_fit = np.where(m_fit)[0]
    if len(idx_fit) > N_AJUSTE:
        idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
    idx_val = np.where(m_val)[0]

    mod = lgb.LGBMRegressor(**PARAMS)
    mod.fit(X[idx_fit], y_v3[idx_fit], sample_weight=pesos(anos[idx_fit]))
    pred = mod.predict(X[idx_val]).astype("float32")

    yv = y_v3[idx_val]
    r_simples = rmse(y_simples[idx_val])
    r_v3 = rmse(yv)
    a_ot = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
    r_mod = rmse(yv, a_ot * pred)

    linhas.append(dict(corte=corte, aval=f"{v_ini}-{v_fim}",
                       clim_simples=r_simples, clim_v3=r_v3, modelo=r_mod,
                       alfa=a_ot,
                       ganho_clim=100 * (1 - r_v3 / r_simples),
                       ganho_mod=100 * (1 - r_mod / r_v3),
                       ganho_total=100 * (1 - r_mod / r_simples),
                       corr=float(np.corrcoef(pred, yv)[0, 1])))
    L = linhas[-1]
    print(f"  fold ate {corte} -> {v_ini}-{v_fim}  ({time.time()-t_fold:.0f}s)")
    print(f"    clim simples {r_simples:.4f} | v3 {r_v3:.4f} "
          f"({L['ganho_clim']:+.2f}%) | modelo {r_mod:.4f} "
          f"({L['ganho_mod']:+.2f}%) | total {L['ganho_total']:+.2f}% "
          f"| alfa {a_ot:.3f} | r {L['corr']:.3f}", flush=True)

    del mod, pred, y_v3, y_simples

res = pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
sep("4. RESUMO")

g_clim = res.ganho_clim.mean()
g_mod = res.ganho_mod.mean()
g_tot = res.ganho_total.mean()

print(f"  ganho da climatologia v3 : {g_clim:+.3f}%")
print(f"  ganho do modelo sobre v3 : {g_mod:+.3f}%")
print(f"  ganho total              : {g_tot:+.3f}%  "
      f"(soma simples: {g_clim + g_mod:+.3f}%)")
print(f"  alfa medio               : {res.alfa.mean():.3f}  "
      f"(min {res.alfa.min():.3f}, max {res.alfa.max():.3f})")
print(f"  correlacao media         : {res['corr'].mean():.3f}")

print(f"""
  REFERENCIA sem peso por recencia (execucao anterior):
    ganho do modelo +1.881%, total +2.829%, r 0.189
  Com tau={TAU} a expectativa eh de ganho proximo de +2.06% no modelo,
  conforme medido em 08.""")

if res.alfa.min() < 0.8:
    pior = res.loc[res.alfa.idxmin()]
    print(f"""
  ATENCAO: o fold ate {int(pior.corte)} tem alfa de {pior.alfa:.3f}, bem
  abaixo dos demais. Eh o mesmo fold historicamente mais dificil. Se 2024
  se parecer com ele, um alfa menor seria o correto - motivo para que a
  segunda submissao final use alfa reduzido.""")


# ---------------------------------------------------------------------------
sep("5. MODELO FINAL")

alfa = float(res.alfa.mean())
print(f"  alfa aplicado: {alfa:.3f}")

v3 = np.load(f"{SAIDA}/climatologia_v3.npz")
nivel_f = v3["nivel"].reshape(12, -1)
incl_f = v3["incl"].reshape(12, -1)
centro_f = float(v3["centro"])

nivel_s = v3["nivel"][:, ::stride, ::stride].reshape(12, n_pt)
incl_s = v3["incl"][:, ::stride, ::stride].reshape(12, n_pt)
y_final = (y_bruto - (nivel_s[mes_alvo - 1, ipt]
                      + K_TEND * incl_s[mes_alvo - 1, ipt]
                      * (ano_alvo - centro_f))).astype("float32")

idx = rng.choice(len(y_final), min(N_AJUSTE * 2, len(y_final)), replace=False)
t_fit = time.time()

# MEDIA DE SEMENTES: o bagging (fraction 0.7) introduz variancia de
# amostragem. Ajustar N modelos com sementes diferentes e tirar a media
# das previsoes reduz essa variancia sem mexer no vies. Nao precisa de
# CV para validar: reducao de variancia eh monotona. N_SEMENTES=1
# reproduz o comportamento anterior.
N_SEMENTES = int(os.environ.get("N_SEMENTES", 1))
modelos_finais = []
for s in range(N_SEMENTES):
    p = dict(PARAMS, random_state=SEMENTE + s)
    m = lgb.LGBMRegressor(**p)
    sub_idx = idx if N_SEMENTES == 1 else \
        np.random.default_rng(SEMENTE + s).choice(len(y_final), len(idx), replace=False)
    m.fit(X[sub_idx], y_final[sub_idx], sample_weight=pesos(anos[sub_idx]))
    modelos_finais.append(m)
    print(f"  semente {s+1}/{N_SEMENTES} ajustada ({time.time()-t_fit:.0f}s)", flush=True)
final = modelos_finais[0]
print(f"  {N_SEMENTES} modelo(s) em {len(idx):,} linhas cada")

imp = pd.DataFrame({"feature": nomes,
                    "gain": final.booster_.feature_importance("gain")})
imp["pct"] = 100 * imp.gain / imp.gain.sum()
grupos = {"estaticas": nomes[:4], "sazonais": nomes[4:6],
          "locais": [n for n in nomes if n.startswith("a_")],
          "grande_escala": [n for n in nomes
                            if n.startswith("pc_") or n.startswith("idx_")],
          "indices_noaa": [n for n in nomes if n.split("_")[0] in
                           ("nino12", "nino34", "nino4", "oni", "tna", "tsa",
                            "soi", "pdo", "dipolo", "nino")],
          "sst_grade": [n for n in nomes if n.startswith("sst_pc")],
          "nmme": [n for n in nomes if n.startswith("nmme_") or n.startswith("mme")
                   or n.split("_")[0] in ("cfsv2", "spear", "nasa", "ccsm4", "cansips",
                                          "seas5", "ukmo", "meteo", "dwd", "cmcc", "jma")],
          "niveis_sup": [n for n in nomes if n.startswith("pc_z200") or n.startswith("pc_u200")
                         or n.startswith("pc_v200") or n.startswith("pc_z500")],
          "mjo": [n for n in nomes if n.startswith("mjo_")]}
print(f"\n  {'grupo':18s} {'% do ganho':>12s}")
for g, cols in grupos.items():
    if cols:
        print(f"  {g:18s} {imp[imp.feature.isin(cols)].pct.sum():12.2f}")

if USAR_SST:
    print(f"\n  {'feature':22s} {'% do ganho':>12s}   (indices NOAA, top 8)")
    ind = imp[imp.feature.isin(grupos["indices_noaa"])].sort_values(
        "pct", ascending=False).head(8)
    for _, r in ind.iterrows():
        print(f"  {r.feature:22s} {r.pct:12.2f}")


# ---------------------------------------------------------------------------
sep("6. PREVISAO E SUBMISSAO")

anom_te = np.mean([m.predict(Xte) for m in modelos_finais], axis=0).astype("float32")
print(f"  anomalia prevista (media de {N_SEMENTES} semente(s)): "
      f"media {anom_te.mean():+.4f}, desvio {anom_te.std():.4f}")

sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
partes = sub["id"].str.split("_", expand=True)
ano_v = partes[0].astype(int).values
mes_v = partes[1].astype(int).values
lat_v = partes[2].astype(float).values
lon_v = partes[3].astype(float).values

ilat = np.rint((lat_v - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon_v - float(tp.lon[0])) / 0.25).astype(int)
assert np.allclose(tp.lat.values[ilat], lat_v), "indice de lat errado"
assert np.allclose(tp.lon.values[ilon], lon_v), "indice de lon errado"
ipt_te = ilat * len(tp.lon) + ilon

base_te = (nivel_f[mes_v - 1, ipt_te]
           + K_TEND * incl_f[mes_v - 1, ipt_te] * (ano_v - centro_f))
pred = base_te + alfa * anom_te

neg = int((pred < 0).sum())
if neg:
    print(f"  {neg:,} negativas ({100*neg/len(pred):.3f}%) truncadas em zero")
    pred = np.maximum(pred, 0.0)
assert np.isfinite(pred).all()
print(f"  previsao final: media {pred.mean():.4f}, max {pred.max():.2f}")

sub["tp_mm_day"] = pred
partes_tag = []
partes_tag.append(FEATURES if FEATURES else "nosst")
partes_tag.append("rec" if TAU is not None else "norec")
if PARAMS["n_estimators"] != 400 or PARAMS["num_leaves"] != 63:
    partes_tag.append(f"t{PARAMS['n_estimators']}l{PARAMS['num_leaves']}")
if N_SEMENTES > 1:
    partes_tag.append(f"s{N_SEMENTES}")
tag = "_".join(partes_tag)
caminho = f"{SUBS}/sub07_{tag}.csv"
sub.to_csv(caminho, index=False, float_format="%.4f")

# variante conservadora, para a segunda submissao final
alfa_cons = float(res.alfa.min())
sub_c = sub.copy()
sub_c["tp_mm_day"] = np.maximum(base_te + alfa_cons * anom_te, 0.0)
caminho_c = f"{SUBS}/sub07_{tag}_conservadora.csv"
sub_c.to_csv(caminho_c, index=False, float_format="%.4f")

with open(f"{SAIDA}/modelo_final_{tag}.pkl", "wb") as f:
    pickle.dump({"modelo": final, "modelos": modelos_finais, "n_sementes": N_SEMENTES,
                 "params": PARAMS, "alfa": alfa,
                 "alfa_conservador": alfa_cons, "tau": TAU,
                 "usar_sst": USAR_SST, "features": nomes,
                 "config_clim": dict(n=N_JANELA, w=W_MIX, k=K_TEND)}, f)
res.to_csv(f"{SAIDA}/cv_final_{tag}.csv", index=False)

esperado = 1.8395 * (1 - g_tot / 100)
print(f"""
  {caminho}          alfa {alfa:.3f}
  {caminho_c}  alfa {alfa_cons:.3f}  (conservadora)

  EXPECTATIVA: a climatologia inicial marcou 1.8395 no placar publico e
  a versao anterior deste modelo marcou 1.76781. Com ganho total de
  {g_tot:.2f}% no CV, espera-se algo em torno de {esperado:.4f}.

  As duas submissoes finais devem diferir no ALFA, nao na arquitetura:
  uma com {alfa:.3f} (media dos folds) e outra com {alfa_cons:.3f} (o fold
  mais dificil), cobrindo o cenario em que 2024 se pareca com ele.""")
