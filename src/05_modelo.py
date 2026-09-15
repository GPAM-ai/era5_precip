#!/usr/bin/env python
"""
Etapa 05 - Modelo, com validacao walk-forward sem vazamento.

O VAZAMENTO CORRIGIDO AQUI:
  Em 02 a climatologia foi ajustada em 1940-2022 e usada para calcular
  anom_alvo de TODOS os anos, inclusive os de validacao. Logo o "RMSE da
  climatologia" medido em 04b (1.7576 sobre 2013-2022) esta otimista: a
  climatologia tinha visto aqueles anos. No teste real ela nao ve 2023/24.

  Aqui cada fold recalcula a climatologia usando SO os anos anteriores ao
  fold. O alvo bruto eh reconstruido como:
      tp_alvo = anom_alvo + clim_global[mes_alvo]
  e re-anomalizado com a climatologia do fold.

DESENHO: WALK-FORWARD EXPANSIVO
  fold 1: ajusta ate 1997, avalia 1998-2002
  fold 2: ajusta ate 2002, avalia 2003-2007
  ...
  fold 5: ajusta ate 2017, avalia 2018-2022

  Sempre prevendo o futuro a partir do passado, que eh a estrutura do
  problema real. Leave-one-year-out embaralharia isso.

O QUE SE MEDE POR FOLD:
  RMSE da climatologia (baseline honesto), RMSE do modelo, alfa otimo,
  ganho percentual, vies. E ao final, intervalo de confianca por
  bootstrap sobre a diferenca - se cruzar zero, o ganho nao eh real.

Uso:
    python src/05_modelo.py [normal|detrend]
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

SAIDA = "dados_processados"
VARIANTE = sys.argv[1] if len(sys.argv) > 1 else "normal"

CORTES = [1997, 2002, 2007, 2012, 2017]
JANELA_AVAL = 5
N_AJUSTE = 2_500_000        # subamostra de linhas por fold
SEMENTE = 42

PARAMS = dict(objective="l2", num_leaves=63, learning_rate=0.05,
              min_child_samples=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
              n_estimators=400, n_jobs=-1, verbose=-1,
              random_state=SEMENTE)


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


# ---------------------------------------------------------------------------
sep(f"1. CARREGANDO (variante = {VARIANTE})")

arq_X = "X_treino_dt.npy" if VARIANTE == "detrend" else "X_treino.npy"
X = np.load(f"{SAIDA}/{arq_X}", mmap_mode="r")
y_anom = np.load(f"{SAIDA}/y_treino.npy")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")

with open(f"{SAIDA}/features.json") as f:
    nomes = json.load(f)["nomes"]

anos = meta["ano_feature"].values
mes_alvo = meta["mes_alvo"].values

print(f"  X: {X.shape}  ({arq_X})")
print(f"  anos: {anos.min()} a {anos.max()}")


# ---------------------------------------------------------------------------
sep("2. RECONSTRUINDO O ALVO BRUTO")

print("""  Para recalcular a climatologia por fold eh preciso do alvo em
  mm/dia, nao da anomalia. Reconstroi-se somando de volta a climatologia
  global que foi usada em 02.
""")

clim_g = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")
tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
n_pt = n_lat * n_lon

stride = (len(clim_g.lat) - 1) // (n_lat - 1)
clim_sub = clim_g.isel(lat=slice(None, None, stride),
                       lon=slice(None, None, stride))
assert clim_sub.sizes["lat"] == n_lat
clim_arr = clim_sub.transpose("month", "lat", "lon").values.reshape(12, n_pt)

# cada bloco de n_pt linhas corresponde a um passo de tempo
n_t = len(y_anom) // n_pt
assert n_t * n_pt == len(y_anom), "linhas nao sao multiplo do numero de pontos"

clim_por_linha = np.empty(len(y_anom), dtype="float32")
for j in range(n_t):
    a, b = j * n_pt, (j + 1) * n_pt
    clim_por_linha[a:b] = clim_arr[mes_alvo[a] - 1]

y_bruto = y_anom + clim_por_linha
print(f"  alvo bruto: media {y_bruto.mean():.4f} mm/dia, "
      f"faixa {y_bruto.min():.2f} a {y_bruto.max():.2f}")
print(f"  (media global de tp era 3.52 - deve bater)")

# indice do ponto de grade de cada linha
ipt = np.tile(np.arange(n_pt, dtype="int32"), n_t)


def clim_do_fold(corte):
    """Climatologia por (ponto, mes-alvo) usando so anos <= corte."""
    m = anos <= corte
    soma = np.zeros((12, n_pt), dtype="float64")
    cont = np.zeros((12, n_pt), dtype="int64")
    np.add.at(soma, (mes_alvo[m] - 1, ipt[m]), y_bruto[m])
    np.add.at(cont, (mes_alvo[m] - 1, ipt[m]), 1)
    return (soma / np.maximum(cont, 1)).astype("float32")


# ---------------------------------------------------------------------------
sep("3. VALIDACAO WALK-FORWARD")

rng = np.random.default_rng(SEMENTE)
linhas = []
residuos = {}

for corte in CORTES:
    v_ini, v_fim = corte + 1, corte + JANELA_AVAL
    m_fit = anos <= corte
    m_val = (anos >= v_ini) & (anos <= v_fim)
    if m_val.sum() == 0:
        continue

    # climatologia do fold, sem olhar os anos de avaliacao
    cf = clim_do_fold(corte)
    clim_linha = cf[mes_alvo - 1, ipt]
    alvo_fold = y_bruto - clim_linha           # anomalia honesta

    idx_fit = np.where(m_fit)[0]
    if len(idx_fit) > N_AJUSTE:
        idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
    idx_val = np.where(m_val)[0]

    mod = lgb.LGBMRegressor(**PARAMS)
    mod.fit(np.asarray(X[idx_fit]), alvo_fold[idx_fit])
    pred = mod.predict(np.asarray(X[idx_val])).astype("float32")

    yv = alvo_fold[idx_val]
    rmse_clim = float(np.sqrt(np.mean(yv ** 2)))
    rmse_mod = float(np.sqrt(np.mean((yv - pred) ** 2)))
    a_ot = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
    rmse_a = float(np.sqrt(np.mean((yv - a_ot * pred) ** 2)))
    corr = float(np.corrcoef(pred, yv)[0, 1])
    vies = float(np.mean(pred - yv))

    residuos[corte] = (yv, pred)
    linhas.append(dict(corte=corte, aval=f"{v_ini}-{v_fim}",
                       n_fit=len(idx_fit), n_val=len(idx_val),
                       rmse_clim=rmse_clim, rmse_mod=rmse_mod,
                       alfa=a_ot, rmse_alfa=rmse_a,
                       ganho=100 * (1 - rmse_a / rmse_clim),
                       corr=corr, vies=vies))

    print(f"  fold ate {corte} -> avalia {v_ini}-{v_fim}")
    print(f"    clim {rmse_clim:.4f} | modelo {rmse_mod:.4f} | "
          f"alfa {a_ot:.3f} -> {rmse_a:.4f} | ganho {linhas[-1]['ganho']:+.2f}% "
          f"| r {corr:.3f}", flush=True)

res = pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
sep("4. RESUMO")

print(f"  ganho medio      : {res.ganho.mean():+.3f}%  "
      f"(desvio {res.ganho.std():.3f})")
print(f"  alfa medio       : {res.alfa.mean():.3f}  "
      f"(min {res.alfa.min():.3f}, max {res.alfa.max():.3f})")
print(f"  correlacao media : {res['corr'].mean():.3f}")
print(f"  folds com ganho  : {int((res.ganho > 0).sum())} de {len(res)}")

print("""
  Um alfa estavel entre folds indica que o encolhimento eh propriedade
  do problema, nao ajuste a um periodo. Se variar muito, usar o valor
  mais conservador (menor) na submissao.""")


# ---------------------------------------------------------------------------
sep("5. BOOTSTRAP: O GANHO EH REAL?")

print("""  Reamostra os RESIDUOS POR MES, nao por linha: pontos de grade do
  mesmo mes sao fortemente correlacionados, e reamostrar linha a linha
  fingiria ter milhoes de amostras independentes quando ha dezenas.
""")

B = 1000
difs = []
for corte, (yv, pred) in residuos.items():
    n_meses = len(yv) // n_pt
    yv_m = yv.reshape(n_meses, n_pt)
    pr_m = pred.reshape(n_meses, n_pt)
    a = res.loc[res.corte == corte, "alfa"].iloc[0]

    for _ in range(B // len(residuos)):
        s = rng.integers(0, n_meses, n_meses)
        yy, pp = yv_m[s].ravel(), pr_m[s].ravel()
        r_c = np.sqrt(np.mean(yy ** 2))
        r_m = np.sqrt(np.mean((yy - a * pp) ** 2))
        difs.append(r_c - r_m)

difs = np.array(difs)
lo, hi = np.percentile(difs, [2.5, 97.5])
print(f"  diferenca media (clim - modelo): {difs.mean():+.5f} mm/dia")
print(f"  IC 95%: [{lo:+.5f}, {hi:+.5f}]")
if lo > 0:
    print("\n  >> o ganho eh estatisticamente significativo.")
else:
    print("\n  >> o IC cruza zero: o ganho NAO eh distinguivel de ruido.")
    print("     Nesse caso, preferir a climatologia pura, que eh mais")
    print("     simples e pontua melhor em reprodutibilidade.")


# ---------------------------------------------------------------------------
sep("6. MODELO FINAL E IMPORTANCIA DAS FEATURES")

print("  Ajuste em toda a serie, para uso na submissao.\n")

idx = rng.choice(len(y_anom), min(N_AJUSTE * 2, len(y_anom)), replace=False)
final = lgb.LGBMRegressor(**PARAMS)
final.fit(np.asarray(X[idx]), y_anom[idx])

imp = pd.DataFrame({"feature": nomes,
                    "ganho": final.booster_.feature_importance("gain")})
imp = imp.sort_values("ganho", ascending=False)
imp["pct"] = 100 * imp.ganho / imp.ganho.sum()

print(f"  {'feature':26s} {'% do ganho':>12s}")
for _, r in imp.head(15).iterrows():
    print(f"  {r.feature:26s} {r.pct:12.2f}")

grupos = {"estaticas": nomes[:4], "sazonais": nomes[4:6],
          "locais": [n for n in nomes if n.startswith("a_")],
          "grande_escala": [n for n in nomes
                            if n.startswith("pc_") or n.startswith("idx_")]}
print(f"\n  {'grupo':20s} {'% do ganho':>12s}")
for g, cols in grupos.items():
    pct = imp[imp.feature.isin(cols)].pct.sum()
    print(f"  {g:20s} {pct:12.2f}")

import pickle
with open(f"{SAIDA}/modelo_{VARIANTE}.pkl", "wb") as f:
    pickle.dump({"modelo": final, "params": PARAMS,
                 "alfa_cv": float(res.alfa.mean()),
                 "alfa_min": float(res.alfa.min()),
                 "ganho_cv": float(res.ganho.mean()),
                 "variante": VARIANTE}, f)
res.to_csv(f"{SAIDA}/cv_{VARIANTE}.csv", index=False)

print(f"""
  modelo_{VARIANTE}.pkl e cv_{VARIANTE}.csv salvos.

  Proximo passo: 06_submissao.py""")