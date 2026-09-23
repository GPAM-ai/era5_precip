#!/usr/bin/env python
"""
Etapa 08 - Pesos amostrais e ensemble.

DUAS HIPOTESES, NA ORDEM:

  PARTE 1 - pesos amostrais favorecendo anos recentes.
    Em 07b os folds recentes foram consistentemente mais dificeis:
    +2.0% no fold ate 2007, +0.9% nos folds ate 2012 e 2017, e isso em
    TODAS as quatro variantes testadas. Somado ao deslocamento de
    distribuicao documentado em 04 (a_t2 desloca 0.8 K entre treino e
    teste), ha motivo para suspeitar que 1940-1970 ensina o modelo sobre
    um regime que nao existe mais.
    Testa-se peso exponencial: w = exp(-(2022 - ano) / tau). Com tau
    grande o peso eh quase uniforme; com tau pequeno so os anos recentes
    contam de fato.

  PARTE 2 - ensemble de modelos com vieses indutivos diferentes.
    LightGBM particiona em degraus; Ridge sobre os PCs captura a resposta
    linear ao ENOS; MLP rasa interpola suavemente, o que eh fisicamente
    plausivel num campo geofisico continuo. Tres modelos que erram de
    formas diferentes tendem a se compensar na media - historicamente o
    ganho mais confiavel que resta quando o ajuste de hiperparametro
    esgotou.

SEM VAZAMENTO:
  A climatologia v3 (janela 24, mistura 0.50, tendencia k=0.15) eh
  reajustada DENTRO de cada fold, com anos anteriores ao corte apenas.
  Esse cuidado faltou em 07b, que reusou o alvo anomalizado globalmente -
  por isso os numeros absolutos de la ficaram deprimidos (+1.31% contra
  +1.96% medidos em 05 nos mesmos folds).

PESOS DO ENSEMBLE, SEM OTIMISMO:
  Os pesos otimos de um fold sao ajustados no proprio conjunto de
  validacao daquele fold, o que eh otimista. O numero honesto reportado
  usa a media dos pesos dos OUTROS folds - validacao cruzada dos proprios
  pesos.

Uso:
    python src/08_ensemble.py [parte1|parte2|tudo]
"""

import gc
import json
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

DADOS = "dados_brutos"
SAIDA = "dados_processados"
PARTE = sys.argv[1] if len(sys.argv) > 1 else "tudo"

N_JANELA, W_MIX, K_TEND = 24, 0.50, 0.15
CORTES = [1997, 2002, 2007, 2012, 2017]
N_AJUSTE = 2_500_000
N_AJUSTE_MLP = 600_000
SEMENTE = 42

TAUS = [None, 60, 40, 25, 15]        # None = peso uniforme

PARAMS_LGB = dict(objective="l2", num_leaves=63, learning_rate=0.05,
                  min_child_samples=200, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
                  n_estimators=400, n_jobs=-1, verbose=-1,
                  random_state=SEMENTE)


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


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

t0 = time.time()
X = np.load(f"{SAIDA}/X_treino.npy")
print(f"  X em RAM: {X.shape}  {X.nbytes/1e9:.2f} GB  "
      f"em {time.time()-t0:.0f}s  [RSS {mem():.0f} MB]")
assert mem() > 7000, "X nao foi carregado para a RAM - ainda eh memmap"

y_anom = np.load(f"{SAIDA}/y_treino.npy")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
anos = meta["ano_feature"].values
mes_alvo = meta["mes_alvo"].values
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)

with open(f"{SAIDA}/features.json") as f:
    nomes = json.load(f)["nomes"]

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
print(f"  alvo bruto: media {y_bruto.mean():.4f} mm/dia  [{mem():.0f} MB]")


def alvo_do_fold(corte):
    """Anomalia em relacao a climatologia v3 ajustada so ate o corte."""
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
    incl = np.stack(incls)
    centro = float(x.mean())

    base = (nivel[mes_alvo - 1, ipt]
            + K_TEND * incl[mes_alvo - 1, ipt] * (ano_alvo - centro))
    return (y_bruto - base).astype("float32")


rng = np.random.default_rng(SEMENTE)


def indices(corte):
    m_fit = anos <= corte
    m_val = (anos >= corte + 1) & (anos <= corte + 5)
    idx_fit = np.where(m_fit)[0]
    if len(idx_fit) > N_AJUSTE:
        idx_fit = rng.choice(idx_fit, N_AJUSTE, replace=False)
    return idx_fit, np.where(m_val)[0]


# ---------------------------------------------------------------------------
if PARTE in ("parte1", "tudo"):
    sep("2. PESOS AMOSTRAIS POR RECENCIA")

    print("""  w = exp(-(2022 - ano) / tau)

  tau=60 -> 1940 pesa 0.25 do que 2022 pesa
  tau=25 -> 1940 pesa 0.04
  tau=15 -> 1940 pesa 0.004 (praticamente so os ultimos 40 anos contam)
""")
    print(f"  {'tau':>6s}" + "".join(f"{c:>10d}" for c in CORTES)
          + f"{'media':>10s}{'r':>8s}")

    linhas_peso = []
    for tau in TAUS:
        ganhos, corrs = [], []
        for corte in CORTES:
            yf = alvo_do_fold(corte)
            idx_fit, idx_val = indices(corte)

            if tau is None:
                w = None
            else:
                w = np.exp(-(2022 - anos[idx_fit]) / tau).astype("float64")
                w = w / w.mean()

            mod = lgb.LGBMRegressor(**PARAMS_LGB)
            mod.fit(X[idx_fit], yf[idx_fit], sample_weight=w)
            pred = mod.predict(X[idx_val]).astype("float32")

            yv = yf[idx_val]
            a = float(np.dot(pred, yv) / (np.dot(pred, pred) + 1e-12))
            ganhos.append(100 * (1 - rmse(yv, a * pred) / rmse(yv)))
            corrs.append(float(np.corrcoef(pred, yv)[0, 1]))

            del mod, pred, yf
            gc.collect()

        rot = "uniforme" if tau is None else str(tau)
        print(f"  {rot:>6s}" + "".join(f"{g:+9.3f}%" for g in ganhos)
              + f"{np.mean(ganhos):+9.3f}%{np.mean(corrs):8.3f}", flush=True)
        linhas_peso.append(dict(tau=(0 if tau is None else tau),
                                ganho=np.mean(ganhos), corr=np.mean(corrs)))

    rp = pd.DataFrame(linhas_peso)
    rp.to_csv(f"{SAIDA}/pesos_recencia.csv", index=False)

    melhor_tau = rp.loc[rp.ganho.idxmax(), "tau"]
    TAU_FINAL = None if melhor_tau == 0 else int(melhor_tau)
    ganho_unif = float(rp.loc[rp.tau == 0, "ganho"].iloc[0])
    delta = rp.ganho.max() - ganho_unif

    print(f"\n  >> melhor: tau = {'uniforme' if TAU_FINAL is None else TAU_FINAL}"
          f"  (ganho {rp.ganho.max():+.3f}%, "
          f"{delta:+.3f} sobre o uniforme)")
    if delta < 0.1:
        print("""  >> diferenca menor que o ruido entre folds. Manter peso
     uniforme: mais simples e sem hiperparametro extra para justificar.""")
        TAU_FINAL = None
else:
    TAU_FINAL = None


# ---------------------------------------------------------------------------
if PARTE in ("parte2", "tudo"):
    sep("3. ENSEMBLE DE TRES MODELOS")

    print(f"""  peso amostral: {'uniforme' if TAU_FINAL is None else f'tau={TAU_FINAL}'}

  LightGBM  particiona em degraus, captura nao-linearidade regional
  Ridge     resposta linear as componentes de grande escala
  MLP rasa  interpola suavemente, plausivel em campo geofisico continuo

  Os tres compartilham as mesmas features. O que muda eh o vies indutivo.
""")

    # a MLP usa um subconjunto: PCs base e lag1, sem lag2/lag3, para
    # treinar em tempo aceitavel sem perder o sinal principal
    cols_mlp = [i for i, n in enumerate(nomes)
                if not (n.endswith("_lag2") or n.endswith("_lag3"))]
    print(f"  colunas para a MLP: {len(cols_mlp)} de {len(nomes)}")

    resultados, preds_por_fold = [], {}

    for corte in CORTES:
        t_fold = time.time()
        yf = alvo_do_fold(corte)
        idx_fit, idx_val = indices(corte)
        yv = yf[idx_val]

        w = None if TAU_FINAL is None else \
            (lambda v: v / v.mean())(np.exp(-(2022 - anos[idx_fit]) / TAU_FINAL))

        # --- LightGBM
        m1 = lgb.LGBMRegressor(**PARAMS_LGB)
        m1.fit(X[idx_fit], yf[idx_fit], sample_weight=w)
        p1 = m1.predict(X[idx_val]).astype("float32")
        del m1

        # --- Ridge (precisa de padronizacao)
        sc = StandardScaler()
        Xf = sc.fit_transform(X[idx_fit])
        m2 = Ridge(alpha=100.0, random_state=SEMENTE)
        m2.fit(Xf, yf[idx_fit], sample_weight=w)
        p2 = m2.predict(sc.transform(X[idx_val])).astype("float32")
        del Xf, m2

        # --- MLP rasa
        sub = rng.choice(idx_fit, min(N_AJUSTE_MLP, len(idx_fit)),
                         replace=False)
        sc2 = StandardScaler()
        Xm = sc2.fit_transform(X[np.ix_(sub, cols_mlp)])
        m3 = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                          alpha=1e-3, learning_rate_init=1e-3,
                          max_iter=40, early_stopping=True,
                          n_iter_no_change=5, random_state=SEMENTE)
        m3.fit(Xm, yf[sub])
        p3 = m3.predict(sc2.transform(X[np.ix_(idx_val, cols_mlp)])
                        ).astype("float32")
        del Xm, m3, sc, sc2
        gc.collect()

        r_clim = rmse(yv)
        indiv = {}
        for nome, p in (("lgbm", p1), ("ridge", p2), ("mlp", p3)):
            a = float(np.dot(p, yv) / (np.dot(p, p) + 1e-12))
            indiv[nome] = (100 * (1 - rmse(yv, a * p) / r_clim),
                           float(np.corrcoef(p, yv)[0, 1]), a)

        # pesos otimos por minimos quadrados nao negativos (aproximacao
        # simples: resolve sem restricao e trunca negativos, renormalizando)
        P = np.vstack([p1, p2, p3]).T.astype("float64")
        coef = np.linalg.lstsq(P, yv.astype("float64"), rcond=None)[0]
        coef_pos = np.clip(coef, 0, None)
        p_ens = P @ coef_pos
        g_ens = 100 * (1 - rmse(yv, p_ens.astype("float32")) / r_clim)

        preds_por_fold[corte] = (P, yv, r_clim)
        resultados.append(dict(corte=corte, **{f"g_{k}": v[0]
                                               for k, v in indiv.items()},
                               **{f"r_{k}": v[1] for k, v in indiv.items()},
                               g_ens=g_ens,
                               w_lgbm=coef_pos[0], w_ridge=coef_pos[1],
                               w_mlp=coef_pos[2]))

        print(f"  fold ate {corte} ({time.time()-t_fold:.0f}s)")
        for k, v in indiv.items():
            print(f"    {k:6s} ganho {v[0]:+.3f}%  r {v[1]:.3f}")
        print(f"    ensemble ganho {g_ens:+.3f}%  "
              f"pesos [{coef_pos[0]:.2f}, {coef_pos[1]:.2f}, "
              f"{coef_pos[2]:.2f}]", flush=True)

        del p1, p2, p3, yf
        gc.collect()

    re_ = pd.DataFrame(resultados)

    sep("4. ENSEMBLE: O NUMERO HONESTO")

    print("""  Os pesos acima foram ajustados no proprio conjunto de
  validacao de cada fold - otimista. O numero honesto usa, em cada fold,
  a media dos pesos dos OUTROS folds.
""")

    honestos = []
    for corte in CORTES:
        outros = re_[re_.corte != corte]
        w = outros[["w_lgbm", "w_ridge", "w_mlp"]].mean().values
        P, yv, r_clim = preds_por_fold[corte]
        g = 100 * (1 - rmse(yv, (P @ w).astype("float32")) / r_clim)
        honestos.append(g)
        print(f"  fold ate {corte}: ganho {g:+.3f}%  "
              f"(pesos dos outros folds: {w.round(2)})")

    print(f"\n  {'modelo':12s} {'ganho medio':>13s}")
    for k in ("lgbm", "ridge", "mlp"):
        print(f"  {k:12s} {re_[f'g_{k}'].mean():+12.3f}%")
    print(f"  {'ensemble':12s} {re_.g_ens.mean():+12.3f}%  (otimista)")
    print(f"  {'ensemble':12s} {np.mean(honestos):+12.3f}%  (honesto)")

    melhor_ind = max(("lgbm", "ridge", "mlp"),
                     key=lambda k: re_[f"g_{k}"].mean())
    delta = np.mean(honestos) - re_[f"g_{melhor_ind}"].mean()
    print(f"\n  ganho do ensemble sobre o melhor individual "
          f"({melhor_ind}): {delta:+.3f} pontos percentuais")

    if delta > 0.1:
        print("""
  >> o ensemble compensa. Adotar na submissao, com os pesos medios de
     todos os folds.""")
        w_final = re_[["w_lgbm", "w_ridge", "w_mlp"]].mean()
        print(f"     pesos finais: lgbm {w_final.w_lgbm:.3f}, "
              f"ridge {w_final.w_ridge:.3f}, mlp {w_final.w_mlp:.3f}")
    else:
        print("""
  >> o ensemble nao supera o melhor individual de forma convincente.
     Manter o modelo unico e registrar o teste no README.""")

    re_.to_csv(f"{SAIDA}/ensemble.csv", index=False)
    print(f"\n  salvo: {SAIDA}/ensemble.csv")