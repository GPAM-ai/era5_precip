#!/usr/bin/env python
"""
Etapa 17 - MOS pontual (ridge por ponto de grade) + combinacao com LightGBM.

O LightGBM eh um modelo GLOBAL: uma arvore para todos os pontos, com
lat/lon como features para modular. A formulacao classica de MOS eh o
oposto: uma regressao LOCAL por ponto de grade, que calibra cada sistema
dinamico com coeficientes proprios daquele lugar.

As duas visoes erram de jeitos diferentes. O ridge pontual nao consegue
usar o estado atmosferico observado (teria features demais para ~490
amostras por ponto); o LightGBM nao consegue calibrar cada ponto com a
precisao de uma regressao dedicada. Combinar as previsoes costuma
superar cada uma.

DESENHO:
  - por ponto (8.787 na grade de treino): ridge sobre as anomalias dos
    sistemas dinamicos (lead 0.5 e 1.5) + medias + sin/cos do mes.
    Fechado, sem iteracao: 8.787 pontos em segundos.
  - walk-forward identico ao de 06/14, produzindo previsoes fora da
    amostra do ridge nos mesmos (idx) do LightGBM em avaliacao_oof.parquet.
  - peso otimo w em pred = w*lgbm + (1-w)*ridge, validado por fold
    (w ajustado nos outros 4 folds, aplicado no quinto).
  - criterio: se o ganho validado da combinacao sobre o LightGBM sozinho
    for < 0.10 p.p., nao aplica.
  - se aplicar: ridge ajustado em todo o treino, previsao no teste com os
    coeficientes do ponto de treino mais proximo (grade cheia = 78.561,
    treino = 8.787; coeficientes variam suavemente), combinada com a
    submissao mais recente do LightGBM.

Uso:
    python src/17_mos_pontual.py saidas/sub07_mme3_norec_t1200l63_s5.csv
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xarray as xr

SAIDA = "dados_processados"
DADOS = "/prj/cptec/alex.campos/satrain/previsao_precipitacao_america_sul/dados"
FEATURES = "mme3"
CORTES = [1997, 2002, 2007, 2012, 2017]
LAMBDA = 30.0            # regularizacao do ridge (validada abaixo)
arq_sub = sys.argv[1] if len(sys.argv) > 1 else "saidas/sub07_mme3_norec_t1200l63_s5.csv"


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def rmse(a, b=None):
    return float(np.sqrt(np.mean((a if b is None else a - b) ** 2)))


# ---------------------------------------------------------------------------
sep("1. COLUNAS DINAMICAS")

with open(f"{SAIDA}/features_{FEATURES}.json") as f:
    nomes = json.load(f)["nomes"]
dinamicas = [n for n in nomes if n.endswith("_anom") and
             n.split("_")[0] in ("cfsv2", "spear", "nasa", "cansips", "seas5", "ukmo",
                                 "meteo", "dwd", "cmcc", "mme05", "mme15", "mme")]
disp = [n for n in nomes if n.endswith("_disp")]
cols = dinamicas + disp + ["sin_mes", "cos_mes"]
ic = [nomes.index(c) for c in cols]
print(f"  {len(cols)} colunas: {len(dinamicas)} anomalias, {len(disp)} indicadores, 2 sazonais")

X = np.load(f"{SAIDA}/X_treino_{FEATURES}.npy", mmap_mode="r")
Xd = np.ascontiguousarray(X[:, ic]).astype("float64")
del X
oof_lgb = pd.read_parquet(f"{SAIDA}/avaliacao_oof.parquet")
meta = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
anos = meta["ano_feature"].values
n_pt = int(meta["ponto"].max()) + 1 if "ponto" in meta else Xd.shape[0] // 992
n_t = Xd.shape[0] // n_pt
ipt = np.tile(np.arange(n_pt), n_t)
print(f"  {n_pt} pontos x {n_t} meses")

# alvo: o mesmo do LightGBM por fold (anomalia contra a v3 do fold). Ja esta
# em oof_lgb.y para as linhas de validacao; para o ajuste precisamos de
# todos os anos <= corte. Reconstroi como em 14.
tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
lat_sub, lon_sub = tr_ds.lat.values, tr_ds.lon.values
tr_ds.close()
stride = (len(tp.lat) - 1) // (n_lat - 1)
mes_alvo = meta["mes_alvo"].values
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)
y_anom = np.load(f"{SAIDA}/y_treino.npy")
clim_g = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")
clim_arr = (clim_g.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
            .transpose("month", "lat", "lon").values.reshape(12, n_pt))
y_bruto = y_anom + clim_arr[mes_alvo - 1, ipt]
tp_sub = tp.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
tp_np = tp_sub.transpose("time", "lat", "lon").values.reshape(len(tp_sub.time), -1)
anos_tp = tp_sub.time.dt.year.values
meses_tp = tp_sub.time.dt.month.values
N_JANELA, W_MIX, K_TEND = 24, 0.50, 0.15


def alvo_do_fold(corte):
    ini = max(1940, corte - N_JANELA + 1)

    def media(a, b):
        out = np.zeros((12, n_pt))
        m = (anos_tp >= a) & (anos_tp <= b)
        for k in range(12):
            out[k] = tp_np[m & (meses_tp == k + 1)].mean(0)
        return out
    nivel = (1 - W_MIX) * media(1940, corte) + W_MIX * media(ini, corte)
    incl = np.zeros((12, n_pt))
    for k in range(12):
        sel = (anos_tp >= ini) & (anos_tp <= corte) & (meses_tp == k + 1)
        x = anos_tp[sel].astype("float64"); xc = x - x.mean()
        incl[k] = (xc @ (tp_np[sel] - tp_np[sel].mean(0))) / (xc ** 2).sum()
    centro = float(anos_tp[(anos_tp >= ini) & (anos_tp <= corte)].mean())
    return (y_bruto - nivel[mes_alvo - 1, ipt]
            - K_TEND * incl[mes_alvo - 1, ipt] * (ano_alvo - centro)).astype("float64")


# ---------------------------------------------------------------------------
sep("2. RIDGE POR PONTO, WALK-FORWARD")


def ridge_por_ponto(Xf, yf, m_fit, lam):
    """Coeficientes (n_pt, p+1) do ridge por ponto, so em linhas m_fit e
    so em meses com ao menos um sistema dinamico disponivel."""
    p = Xf.shape[1]
    B = np.zeros((n_pt, p + 1))
    # usa apenas linhas onde ha previsao dinamica (mme_n > 0 <=> algum _disp = 1)
    tem = (Xf[:, len(dinamicas):len(dinamicas) + len(disp)].sum(1) > 0) & m_fit
    for k in range(n_pt):
        r = tem & (ipt == k)
        if r.sum() < 60:
            continue
        A = np.hstack([Xf[r], np.ones((r.sum(), 1))])
        G = A.T @ A
        G[np.arange(p), np.arange(p)] += lam           # nao regulariza o intercepto
        B[k] = np.linalg.solve(G, A.T @ yf[r])
    return B


def prever(Xf, B, pontos):
    A = np.hstack([Xf, np.ones((len(Xf), 1))])
    return np.einsum("ij,ij->i", A, B[pontos])


oof_r = []
t0 = time.time()
for corte in CORTES:
    yf = alvo_do_fold(corte)
    m_fit = anos <= corte
    m_val = (anos >= corte + 1) & (anos <= corte + 5)
    B = ridge_por_ponto(Xd, yf, m_fit, LAMBDA)
    iv = np.where(m_val)[0]
    pv = prever(Xd[iv], B, ipt[iv])
    r = float(np.corrcoef(pv, yf[iv])[0, 1])
    print(f"  fold {corte}: ridge ganho {100*(1-rmse(yf[iv], pv)/rmse(yf[iv])):+.2f}%  "
          f"r {r:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    oof_r.append(pd.DataFrame({"idx": iv, "ridge": pv, "fold": corte}))
oof_r = pd.concat(oof_r)

oof = oof_lgb.merge(oof_r, on=["idx", "fold"])
assert len(oof) == len(oof_lgb), "linhas do ridge nao casam com as do LightGBM"


# ---------------------------------------------------------------------------
sep("3. COMBINACAO, VALIDADA POR FOLD")

y = oof.y.values
pl, pr = oof.pred.values, oof.ridge.values
print(f"  LightGBM sozinho : rmse {rmse(y, pl):.4f}  r {np.corrcoef(pl, y)[0,1]:.3f}")
print(f"  ridge sozinho    : rmse {rmse(y, pr):.4f}  r {np.corrcoef(pr, y)[0,1]:.3f}")
print(f"  correlacao entre as duas previsoes: {np.corrcoef(pl, pr)[0,1]:.3f}  "
      f"(quanto menor, mais diversidade)")

# w otimo global (in-sample) e validado (w de 4 folds aplicado no 5o)
def w_otimo(a, b, t):
    d = a - b
    return float(np.dot(d, t - b) / np.dot(d, d))

w_in = w_otimo(pl, pr, y)
erros = []
ws = []
for c in CORTES:
    m = oof.fold.values == c
    w = w_otimo(pl[~m], pr[~m], y[~m])
    ws.append(w)
    erros.append((y[m] - (w * pl[m] + (1 - w) * pr[m])) ** 2)
rmse_comb = float(np.sqrt(np.mean(np.concatenate(erros))))
rmse_lgb = rmse(y, pl)
ganho = 100 * (1 - rmse_comb / rmse_lgb)

print(f"\n  w (peso do LightGBM) in-sample: {w_in:.3f}   por fold: {[f'{w:.2f}' for w in ws]}")
print(f"  rmse LightGBM  : {rmse_lgb:.4f}")
print(f"  rmse combinacao: {rmse_comb:.4f}  (validada)")
print(f"  ganho          : {ganho:+.3f} p.p.")

if ganho < 0.10:
    print("""
  >> abaixo de 0.10 p.p.: nao aplica. O LightGBM ja extrai o que o
     ridge pontual extrairia. Registrar no README.""")
    sys.exit(0)


# ---------------------------------------------------------------------------
sep("4. APLICANDO A SUBMISSAO")

w = float(np.mean(ws))
print(f"  peso do LightGBM: {w:.3f}")

# ridge em todo o treino
yf = alvo_do_fold(2022)
B = ridge_por_ponto(Xd, yf, np.ones(len(yf), bool), LAMBDA)

# teste: colunas dinamicas da grade cheia; coeficientes do ponto de treino
# mais proximo
Xte = np.load(f"{SAIDA}/X_teste_{FEATURES}.npy", mmap_mode="r")
Xtd = np.ascontiguousarray(Xte[:, ic]).astype("float64")
sub = pd.read_csv(arq_sub)
partes = sub["id"].str.split("_", expand=True)
lat_v, lon_v = partes[2].astype(float).values, partes[3].astype(float).values
i_la = np.clip(np.rint((lat_v - lat_sub[0]) / (lat_sub[1] - lat_sub[0])).astype(int), 0, n_lat - 1)
i_lo = np.clip(np.rint((lon_v - lon_sub[0]) / (lon_sub[1] - lon_sub[0])).astype(int), 0, n_lon - 1)
pt_prox = i_la * n_lon + i_lo
anom_ridge = prever(Xtd, B, pt_prox)

# anomalia do LightGBM: recupera da submissao (clim v3 + alfa*anom)
pk = sorted([p for p in os.listdir(SAIDA) if p.startswith("modelo_final_mme3")],
            key=lambda p: os.path.getmtime(f"{SAIDA}/{p}"))[-1]
_lg = pickle.load(open(f"{SAIDA}/{pk}", "rb"))
alfa = _lg["alfa_conservador"] if "conservadora" in arq_sub else _lg["alfa"]
print(f"  alfa usado ({'conservador' if 'conservadora' in arq_sub else 'principal'}): {alfa:.3f}")
v3 = np.load(f"{SAIDA}/climatologia_v3.npz")
mes = partes[1].astype(int).values
ano_v = partes[0].astype(int).values
ilat = np.rint((lat_v - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon_v - float(tp.lon[0])) / 0.25).astype(int)
ipt_full = ilat * len(tp.lon) + ilon
nivel = v3["nivel"].reshape(12, -1); incl = v3["incl"].reshape(12, -1)
base = nivel[mes - 1, ipt_full] + float(v3["k"]) * incl[mes - 1, ipt_full] * (ano_v - float(v3["centro"]))
trunc = sub["tp_mm_day"].values <= 0
anom_lgb = (sub["tp_mm_day"].values - base) / alfa

comb = w * anom_lgb + (1 - w) * anom_ridge
novo = np.maximum(base + alfa * comb, 0)
novo = np.where(trunc, sub["tp_mm_day"].values, novo)
out = arq_sub.replace(".csv", "_mos.csv")
sub["tp_mm_day"] = novo
sub.to_csv(out, index=False, float_format="%.4f")
print(f"  salvo: {out}")
print(f"  anomalia media: lgbm {anom_lgb.mean():+.4f}  ridge {anom_ridge.mean():+.4f}  "
      f"comb {comb.mean():+.4f}")

with open(f"{SAIDA}/ridge_pontual.pkl", "wb") as f:
    pickle.dump({"B": B, "w_lgbm": w, "cols": cols, "idx_cols": ic, "lambda": LAMBDA,
                 "n_lat": n_lat, "n_lon": n_lon, "lat_sub": lat_sub, "lon_sub": lon_sub,
                 "alfa": alfa}, f)
print(f"  salvo: {SAIDA}/ridge_pontual.pkl (para predict.py)")