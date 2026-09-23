#!/usr/bin/env python
"""
predict.py - predicao rapida com os modelos salvos, sem retreinar.

Le X_teste_mme3.npy (de prepare_data.py), os modelos LightGBM salvos por
06_submissao.py (modelo_final_mme3.pkl, com as 5 sementes), o ridge
pontual salvo por 17_mos_pontual.py (ridge_pontual.pkl), e gera a
submissao final:

    previsao = clim_v3 + alfa * (w * anom_lgbm + (1-w) * anom_ridge)

Uso:
    python predict.py [saida.csv]

Tempo: ~3 min. Requer prepare_data.py concluido (para X_teste) e os dois
.pkl presentes (gerados por train.py, ou versionados no repositorio).
"""

import json
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr

cfg = json.load(open("settings.json"))
P = cfg["PROCESSED_DATA_DIR"]
DADOS = cfg["RAW_DATA_DIR"]
FEATURES = cfg["FEATURES"]
saida = sys.argv[1] if len(sys.argv) > 1 else f"{cfg['SUBMISSION_DIR']}/submissao_final.csv"

print("1. carregando modelos e teste")
lg = pickle.load(open(f"{P}/modelo_final_{FEATURES}.pkl", "rb"))
modelos = lg.get("modelos", [lg["modelo"]])
alfa = lg["alfa"]
rp = pickle.load(open(f"{P}/ridge_pontual.pkl", "rb"))
Xte = np.load(f"{P}/X_teste_{FEATURES}.npy")
print(f"   LightGBM: {len(modelos)} semente(s), alfa {alfa:.3f}")
print(f"   ridge pontual: {rp['B'].shape[0]} pontos, peso LightGBM {rp['w_lgbm']:.3f}")
print(f"   teste: {Xte.shape}")

print("2. anomalia do LightGBM")
anom_lgb = np.mean([m.predict(Xte) for m in modelos], axis=0)

print("3. anomalia do ridge pontual")
sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
partes = sub["id"].str.split("_", expand=True)
lat_v, lon_v = partes[2].astype(float).values, partes[3].astype(float).values
lat_sub, lon_sub, n_lat, n_lon = rp["lat_sub"], rp["lon_sub"], rp["n_lat"], rp["n_lon"]
i_la = np.clip(np.rint((lat_v - lat_sub[0]) / (lat_sub[1] - lat_sub[0])).astype(int), 0, n_lat - 1)
i_lo = np.clip(np.rint((lon_v - lon_sub[0]) / (lon_sub[1] - lon_sub[0])).astype(int), 0, n_lon - 1)
pt = i_la * n_lon + i_lo
Xd = Xte[:, rp["idx_cols"]].astype("float64")
A = np.hstack([Xd, np.ones((len(Xd), 1))])
anom_ridge = np.einsum("ij,ij->i", A, rp["B"][pt])

print("4. climatologia v3 e combinacao")
v3 = np.load(f"{P}/climatologia_v3.npz")
tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp
ano, mes = partes[0].astype(int).values, partes[1].astype(int).values
ilat = np.rint((lat_v - float(tp.lat[0])) / 0.25).astype(int)
ilon = np.rint((lon_v - float(tp.lon[0])) / 0.25).astype(int)
ipt = ilat * len(tp.lon) + ilon
nivel = v3["nivel"].reshape(12, -1); incl = v3["incl"].reshape(12, -1)
base = nivel[mes - 1, ipt] + float(v3["k"]) * incl[mes - 1, ipt] * (ano - float(v3["centro"]))

w = rp["w_lgbm"]
comb = w * anom_lgb + (1 - w) * anom_ridge
prev = np.maximum(base + alfa * comb, 0)

sub["tp_mm_day"] = prev
sub.to_csv(saida, index=False, float_format="%.4f")
print(f"\n   {saida}: {len(sub):,} linhas, media {prev.mean():.3f} mm/dia, "
      f"max {prev.max():.1f}, truncadas {(prev == 0).sum():,}")
print(f"   anomalia media: lgbm {anom_lgb.mean():+.4f}  ridge {anom_ridge.mean():+.4f}  "
      f"comb {comb.mean():+.4f}")
