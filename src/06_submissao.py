#!/usr/bin/env python
"""
Etapa 06 - Modelo sobre a climatologia v3, e submissao.

A PERGUNTA QUE ESTE SCRIPT RESPONDE:
  01d ganhou 0.55% melhorando a climatologia (janela de 24 anos, mistura
  0.5, tendencia com k=0.15). 05 ganhou 1.99% com o LightGBM sobre uma
  climatologia de media simples. Os ganhos SOMAM?

  Provavelmente nao somam inteiros. O modelo de 05 tinha clim_alvo, lat e
  lon entre as features e pode ter aprendido sozinho parte da deriva que a
  v3 ja captura explicitamente. Se for esse o caso, treinar sobre a v3
  deixa menos residuo para o modelo explicar.

  Aqui o alvo eh recalculado como anomalia em relacao a v3 e o modelo eh
  retreinado. A validacao refaz a v3 DENTRO de cada fold - janela,
  mistura e tendencia ajustadas so com anos anteriores - para que o
  baseline de comparacao seja honesto.

SOBRE O ENCOLHIMENTO:
  05 mediu alfa medio de 0.987, estavel entre folds (0.907 a 1.032). O
  LightGBM com lambda_l2=5 ja devolve anomalias calibradas; nao ha o que
  encolher. O alfa continua sendo estimado por fold e aplicado, mas a
  expectativa eh que fique perto de 1.

Uso:
    python src/06_submissao.py
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

DADOS = "/prj/cptec/alex.campos/satrain/previsao_precipitacao_america_sul/dados"
SAIDA = "dados_processados"
SUBS = "saidas"

N_JANELA, W_MIX, K_TEND = 24, 0.50, 0.15      # escolhidos em 01d
CORTES = [1997, 2002, 2007, 2012, 2017]
N_AJUSTE = 2_500_000
SEMENTE = 42

PARAMS = dict(objective="l2", num_leaves=63, learning_rate=0.05,
              min_child_samples=200, feature_fraction=0.7,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
              n_estimators=400, n_jobs=-1, verbose=-1, random_state=SEMENTE)


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

X = np.load(f"{SAIDA}/X_treino.npy", mmap_mode="r")
y_anom = np.load(f"{SAIDA}/y_treino.npy")
Xte = np.load(f"{SAIDA}/X_teste.npy", mmap_mode="r")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

with open(f"{SAIDA}/features.json") as f:
    nomes = json.load(f)["nomes"]

anos = meta["ano_feature"].values
mes_alvo = meta["mes_alvo"].values

tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
n_pt = n_lat * n_lon
stride = (len(tp.lat) - 1) // (n_lat - 1)

print(f"  treino: {X.shape}, grade {n_lat}x{n_lon} (stride {stride})")
print(f"  teste : {Xte.shape}")


# ---------------------------------------------------------------------------
sep("2. RECONSTRUINDO O ALVO BRUTO")

clim_g = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")
clim_sub = clim_g.isel(lat=slice(None, None, stride),
                       lon=slice(None, None, stride))
clim_arr = clim_sub.transpose("month", "lat", "lon").values.reshape(12, n_pt)

n_t = len(y_anom) // n_pt
ipt = np.tile(np.arange(n_pt, dtype="int32"), n_t)
y_bruto = y_anom + clim_arr[mes_alvo - 1, ipt]

# ano do ALVO (o mes-alvo pode cair no ano seguinte ao mes-feature)
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)

print(f"  alvo bruto: media {y_bruto.mean():.4f} mm/dia")
print(f"  anos-alvo: {ano_alvo.min()} a {ano_alvo.max()}")


# ---------------------------------------------------------------------------
sep("3. CLIMATOLOGIA v3 POR FOLD")

print(f"""  Configuracao de 01d: janela {N_JANELA} anos, mistura {W_MIX:.2f},
  tendencia k={K_TEND:.2f}. Dentro de cada fold, tudo eh reajustado com os
  anos anteriores ao corte - nada do periodo de avaliacao entra.
""")

tp_sub = tp.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))


def v3_do_fold(corte):
    """Devolve (nivel, inclinacao, centro) na grade subamostrada."""
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
        yv = sm.values.reshape(len(x), -1)
        incls.append(np.tensordot(xc, yv - yv.mean(0), axes=(0, 0))
                     / (xc ** 2).sum())
    return nivel, np.stack(incls), float(x.mean())


def anomalia_v3(nivel, incl, centro):
    base = nivel[mes_alvo - 1, ipt] + K_TEND * incl[mes_alvo - 1, ipt] * \
        (ano_alvo - centro)
    return (y_bruto - base).astype("float32"), base.astype("float32")


# ---------------------------------------------------------------------------
sep("4. VALIDACAO: OS GANHOS SOMAM?")

rng = np.random.default_rng(SEMENTE)
linhas = []

for corte in CORTES:
    v_ini, v_fim = corte + 1, corte + 5
    m_fit = anos <= corte
    m_val = (anos >= v_ini) & (anos <= v_fim)

    # baseline A: climatologia de media simples (a de 05)
    soma = np.zeros((12, n_pt)); cont = np.zeros((12, n_pt))
    np.add.at(soma, (mes_alvo[m_fit] - 1, ipt[m_fit]), y_bruto[m_fit])
    np.add.at(cont, (mes_alvo[m_fit] - 1, ipt[m_fit]), 1)
    base_simples = (soma / np.maximum(cont, 1))[mes_alvo - 1, ipt]
    y_simples = y_bruto - base_simples

    # baseline B: climatologia v3
    nivel, incl, centro = v3_do_fold(corte)
    y_v3, base_v3 = anomalia_v3(nivel, incl, centro)

    idx_fit = np.where(m_fit)[0]
    if len(idx_fit) > N_AJUSTE:
        idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
    idx_val = np.where(m_val)[0]
    Xf, Xv = np.asarray(X[idx_fit]), np.asarray(X[idx_val])

    r_simples = float(np.sqrt(np.mean(y_simples[idx_val] ** 2)))
    r_v3 = float(np.sqrt(np.mean(y_v3[idx_val] ** 2)))

    # modelo sobre a v3
    mod = lgb.LGBMRegressor(**PARAMS)
    mod.fit(Xf, y_v3[idx_fit])
    pred = mod.predict(Xv).astype("float32")
    yv = y_v3[idx_val]
    a_ot = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
    r_mod = float(np.sqrt(np.mean((yv - a_ot * pred) ** 2)))

    linhas.append(dict(corte=corte, aval=f"{v_ini}-{v_fim}",
                       clim_simples=r_simples, clim_v3=r_v3,
                       modelo_v3=r_mod, alfa=a_ot,
                       ganho_clim=100 * (1 - r_v3 / r_simples),
                       ganho_mod=100 * (1 - r_mod / r_v3),
                       ganho_total=100 * (1 - r_mod / r_simples),
                       corr=float(np.corrcoef(pred, yv)[0, 1])))

    L = linhas[-1]
    print(f"  fold ate {corte} -> {v_ini}-{v_fim}")
    print(f"    clim simples {r_simples:.4f} | clim v3 {r_v3:.4f} "
          f"({L['ganho_clim']:+.2f}%) | + modelo {r_mod:.4f} "
          f"({L['ganho_mod']:+.2f}%) | total {L['ganho_total']:+.2f}% "
          f"| alfa {a_ot:.3f}", flush=True)

res = pd.DataFrame(linhas)

sep("5. OS GANHOS SAO ADITIVOS?")

g_clim = res.ganho_clim.mean()
g_mod = res.ganho_mod.mean()
g_tot = res.ganho_total.mean()
esperado = g_clim + g_mod

print(f"  ganho so da climatologia v3 : {g_clim:+.3f}%")
print(f"  ganho do modelo sobre a v3  : {g_mod:+.3f}%")
print(f"  ganho total observado       : {g_tot:+.3f}%")
print(f"  soma simples seria          : {esperado:+.3f}%")
print(f"  alfa medio                  : {res.alfa.mean():.3f}")
print(f"  correlacao media            : {res['corr'].mean():.3f}")

print(f"\n  ganho do modelo em 05 (sobre media simples): +1.991%")
if g_mod < 1.7:
    print(f"""  >> o modelo perde forca sobre a v3 ({g_mod:.2f}% contra 1.99%).
     Parte do que ele explicava era a deriva que a v3 agora captura
     explicitamente. Os ganhos NAO somam inteiros - registrar no README.""")
else:
    print(f"""  >> o modelo mantem a forca sobre a v3 ({g_mod:.2f}%). Os ganhos
     sao largamente independentes: a v3 corrige o nivel, o modelo
     explica a variabilidade interanual.""")


# ---------------------------------------------------------------------------
sep("6. MODELO FINAL E PREVISAO")

alfa = float(res.alfa.mean())
print(f"  alfa aplicado: {alfa:.3f} (media dos folds)")

# v3 ajustada em toda a serie
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
final = lgb.LGBMRegressor(**PARAMS)
final.fit(np.asarray(X[idx]), y_final[idx])
print(f"  ajustado em {len(idx):,} linhas")

anom_te = final.predict(np.asarray(Xte)).astype("float32")
print(f"  anomalia prevista: media {anom_te.mean():+.4f}, "
      f"desvio {anom_te.std():.4f}")

sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
partes = sub["id"].str.split("_", expand=True)
ano_te_v = partes[0].astype(int).values
mes_te_v = partes[1].astype(int).values
lat_te = partes[2].astype(float).values
lon_te = partes[3].astype(float).values

ilat = np.rint((lat_te - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon_te - float(tp.lon[0])) / 0.25).astype(int)
assert np.allclose(tp.lat.values[ilat], lat_te)
ipt_te = ilat * len(tp.lon) + ilon

base_te = (nivel_f[mes_te_v - 1, ipt_te]
           + K_TEND * incl_f[mes_te_v - 1, ipt_te] * (ano_te_v - centro_f))
pred = base_te + alfa * anom_te

neg = int((pred < 0).sum())
if neg:
    print(f"  {neg:,} negativas ({100*neg/len(pred):.3f}%) truncadas em zero")
    pred = np.maximum(pred, 0.0)

print(f"  previsao final: media {pred.mean():.4f}, max {pred.max():.2f}")
assert np.isfinite(pred).all()

sub["tp_mm_day"] = pred
caminho = f"{SUBS}/sub04_modelo_v3.csv"
sub.to_csv(caminho, index=False, float_format="%.4f")

with open(f"{SAIDA}/modelo_final.pkl", "wb") as f:
    pickle.dump({"modelo": final, "params": PARAMS, "alfa": alfa,
                 "config_clim": dict(n=N_JANELA, w=W_MIX, k=K_TEND)}, f)
res.to_csv(f"{SAIDA}/cv_v3.csv", index=False)

esperado_placar = 1.8395 * (1 - g_tot / 100)
print(f"""
  salvo: {caminho}

  EXPECTATIVA: a climatologia original marcou 1.8395. Com ganho total de
  {g_tot:.2f}% no CV, espera-se algo em torno de {esperado_placar:.4f}.

  Comparacao com o placar de hoje: 1o lugar 1.746, 4o lugar 1.785.""")