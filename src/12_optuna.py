#!/usr/bin/env python
"""
Etapa 12 - Busca de hiperparametros (Optuna) para o LightGBM.

EXPECTATIVA CALIBRADA:
  Os parametros em uso (num_leaves=63, lr=0.05, 400 arvores, l2=5,
  min_child=200) sao padroes razoaveis que nunca foram ajustados. Uma
  busca bayesiana deve render 0.2 a 0.5% no CV. Eh pouco, mas roda
  sozinho em batch enquanto o esforco principal vai para informacao
  (TSM em grade), que eh onde os ganhos grandes vieram.

DESENHO:
  - objetivo = ganho medio do modelo sobre a climatologia v3, medido nos
    3 folds mais recentes (2007, 2012, 2017), que sao os mais parecidos
    com prever 2023/24. Os dois folds antigos ficam de fora para
    economizar tempo e porque pesam menos na decisao.
  - climatologia v3 recalculada por fold (sem vazamento), como em 06.
  - subamostra de 1.5M linhas por fold para o ajuste, com early stopping
    numa fatia de validacao interna, para que n_estimators seja
    escolhido pelos dados e nao fixado.
  - cada tentativa leva ~2-3 min; 40 tentativas ~ 2 horas.

CUIDADO:
  O CV ja divergiu do placar duas vezes. Um ganho de 0.3% no CV nao
  garante nada em 2023/24. O resultado eh candidato a teste no placar,
  nao decisao automatica.

Uso:
    FEATURES=grade python -u src/12_optuna.py [n_tentativas]
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    raise SystemExit("pip install optuna")

DADOS = "dados_brutos"
SAIDA = "dados_processados"

FEATURES = os.environ.get("FEATURES", "grade")
sufixo = f"_{FEATURES}" if FEATURES else ""
N_TENT = int(sys.argv[1]) if len(sys.argv) > 1 else 40

N_JANELA, W_MIX, K_TEND = 24, 0.50, 0.15
CORTES = [2007, 2012, 2017]
N_AJUSTE = 1_500_000
FRAC_VAL_INTERNA = 0.15
SEMENTE = 42


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def rmse(a, b=None):
    return float(np.sqrt(np.mean((a if b is None else a - b) ** 2)))


# ---------------------------------------------------------------------------
sep(f"1. CARREGANDO (features = {FEATURES or 'original'})")

X = np.load(f"{SAIDA}/X_treino{sufixo}.npy")
print(f"  X: {X.shape}  {X.nbytes/1e9:.2f} GB")
y_anom = np.load(f"{SAIDA}/y_treino.npy")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
anos = meta["ano_feature"].values
mes_alvo = meta["mes_alvo"].values
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)

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

# ERSST-style numpy puro para a climatologia v3 por fold (rapido)
tp_sub = tp.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
tp_np = tp_sub.transpose("time", "lat", "lon").values.reshape(len(tp_sub.time), -1)
anos_tp = tp_sub.time.dt.year.values
meses_tp = tp_sub.time.dt.month.values


def alvo_do_fold(corte):
    ini = max(1940, corte - N_JANELA + 1)

    def media(a, b):
        out = np.zeros((12, n_pt))
        m = (anos_tp >= a) & (anos_tp <= b)
        for k in range(12):
            sel = m & (meses_tp == k + 1)
            out[k] = tp_np[sel].mean(0)
        return out

    nivel = (1 - W_MIX) * media(1940, corte) + W_MIX * media(ini, corte)
    incl = np.zeros((12, n_pt))
    for k in range(12):
        sel = (anos_tp >= ini) & (anos_tp <= corte) & (meses_tp == k + 1)
        x = anos_tp[sel].astype("float64")
        xc = x - x.mean()
        v = tp_np[sel]
        incl[k] = (xc @ (v - v.mean(0))) / (xc ** 2).sum()
    centro = float(anos_tp[(anos_tp >= ini) & (anos_tp <= corte)].mean())
    base = nivel[mes_alvo - 1, ipt] + K_TEND * incl[mes_alvo - 1, ipt] * (ano_alvo - centro)
    return (y_bruto - base).astype("float32")


# pre-computa alvos e indices por fold, uma vez
rng = np.random.default_rng(SEMENTE)
folds = []
for corte in CORTES:
    yf = alvo_do_fold(corte)
    m_fit = anos <= corte
    m_val = (anos >= corte + 1) & (anos <= corte + 5)
    idx_fit = np.where(m_fit)[0]
    if len(idx_fit) > N_AJUSTE:
        idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
    # fatia interna para early stopping, separada por ANO para nao vazar
    anos_fit = np.unique(anos[idx_fit])
    anos_int = rng.choice(anos_fit, max(3, int(len(anos_fit) * FRAC_VAL_INTERNA)),
                          replace=False)
    m_int = np.isin(anos[idx_fit], anos_int)
    folds.append(dict(corte=corte, yf=yf,
                      idx_tr=idx_fit[~m_int], idx_int=idx_fit[m_int],
                      idx_val=np.where(m_val)[0]))
    print(f"  fold {corte}: treino {len(folds[-1]['idx_tr']):,}  "
          f"interno {len(folds[-1]['idx_int']):,}  aval {len(folds[-1]['idx_val']):,}")


# ---------------------------------------------------------------------------
sep(f"2. BUSCA ({N_TENT} tentativas)")

BASE = dict(objective="l2", n_jobs=-1, verbose=-1, random_state=SEMENTE,
            bagging_freq=1)

# referencia: os parametros em producao
REF = dict(num_leaves=63, learning_rate=0.05, min_child_samples=200,
           feature_fraction=0.7, bagging_fraction=0.7, lambda_l2=5.0,
           lambda_l1=1e-3, max_bin=255, n_estimators=400)
# lambda_l1=1e-3 em vez de 0: o espaco de busca eh logaritmico e nao
# aceita zero; na pratica eh a mesma coisa


def avaliar(params, n_max=2000):
    ganhos = []
    for f in folds:
        yf = f["yf"]
        mod = lgb.LGBMRegressor(**BASE, **params, n_estimators=n_max)
        mod.fit(X[f["idx_tr"]], yf[f["idx_tr"]],
                eval_set=[(X[f["idx_int"]], yf[f["idx_int"]])],
                callbacks=[lgb.early_stopping(50, verbose=False)])
        pred = mod.predict(X[f["idx_val"]]).astype("float32")
        yv = yf[f["idx_val"]]
        a = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
        ganhos.append(100 * (1 - rmse(yv, a * pred) / rmse(yv)))
        f["ultimo_n_iter"] = mod.best_iteration_
    return float(np.mean(ganhos)), ganhos


t0 = time.time()
ref_ganho, ref_por_fold = avaliar({k: v for k, v in REF.items()
                                   if k != "n_estimators"},
                                  n_max=REF["n_estimators"])
print(f"  referencia (producao): ganho {ref_ganho:+.3f}%  "
      f"folds {[f'{g:+.2f}' for g in ref_por_fold]}  ({time.time()-t0:.0f}s)\n")


def objetivo(trial):
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 50, 2000, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.3, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        lambda_l2=trial.suggest_float("lambda_l2", 0.1, 100, log=True),
        lambda_l1=trial.suggest_float("lambda_l1", 1e-3, 10, log=True),
        max_bin=trial.suggest_categorical("max_bin", [63, 127, 255]),
    )
    t = time.time()
    ganho, por_fold = avaliar(params)
    n_it = int(np.mean([f["ultimo_n_iter"] for f in folds]))
    trial.set_user_attr("n_iter", n_it)
    trial.set_user_attr("por_fold", por_fold)
    print(f"  #{trial.number:3d}  ganho {ganho:+.3f}%  "
          f"({'+' if ganho > ref_ganho else '-'}{abs(ganho-ref_ganho):.3f} vs ref)  "
          f"folhas {params['num_leaves']:3d}  lr {params['learning_rate']:.3f}  "
          f"iter {n_it:4d}  ({time.time()-t:.0f}s)", flush=True)
    return ganho


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=SEMENTE))
study.enqueue_trial({k: v for k, v in REF.items() if k != "n_estimators"})
study.optimize(objetivo, n_trials=N_TENT)


# ---------------------------------------------------------------------------
sep("3. RESULTADO")

melhor = study.best_trial
print(f"  referencia : {ref_ganho:+.3f}%")
print(f"  melhor     : {melhor.value:+.3f}%  (tentativa #{melhor.number})")
print(f"  diferenca  : {melhor.value - ref_ganho:+.3f} pontos percentuais\n")
print("  parametros:")
for k, v in melhor.params.items():
    print(f"    {k:20s} {v}")
print(f"    {'n_estimators':20s} {melhor.user_attrs['n_iter']}  (por early stopping)")
print(f"\n  por fold: {[f'{g:+.2f}' for g in melhor.user_attrs['por_fold']]}")
print(f"  referencia: {[f'{g:+.2f}' for g in ref_por_fold]}")

if melhor.value - ref_ganho < 0.15:
    print("""
  >> ganho abaixo de 0.15 p.p.: dentro do ruido entre folds. Manter os
     parametros de producao - sao mais simples de justificar e nao ha
     evidencia de que a mudanca transfira para 2023/24.""")
else:
    print("""
  >> ganho relevante no CV. Candidato a teste no placar: gerar submissao
     com estes parametros e comparar com a atual. Lembrar que o CV ja
     divergiu do placar duas vezes.""")

df = study.trials_dataframe()
df.to_csv(f"{SAIDA}/optuna_{FEATURES or 'orig'}.csv", index=False)
with open(f"{SAIDA}/optuna_melhor_{FEATURES or 'orig'}.json", "w") as f:
    json.dump({"params": melhor.params, "n_estimators": melhor.user_attrs["n_iter"],
               "ganho": melhor.value, "referencia": ref_ganho,
               "features": FEATURES}, f, indent=2)
print(f"\n  salvo: optuna_{FEATURES or 'orig'}.csv e optuna_melhor_*.json")