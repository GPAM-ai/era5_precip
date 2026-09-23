#!/usr/bin/env python
"""
Etapa 15 - Niveis superiores do ERA5 (200/500 hPa) e MJO.

NIVEIS SUPERIORES:
  A competicao entregou so 850 hPa. A circulacao que governa a estacao
  chuvosa da America do Sul - Alta da Bolivia, cavado do Nordeste, jato
  subtropical - esta em 200 hPa. Mesmo tratamento dos campos de 850 em
  02/03: anomalia por mes calendario (mistura de janelas), agregacao a
  2 graus, PCA ajustada no treino, defasagens 1-3. Entram so via EOF
  (features globais), nao como anomalia local.

MJO:
  Indice RMM (Wheeler & Hendon 2004) do BoM, diario desde 1974. Media
  mensal de RMM1, RMM2 e amplitude, mais a fase dominante do mes. A
  fase da MJO modula a ZCAS com previsibilidade de 2-4 semanas.

ALINHAMENTO: mes-feature, como todas as covariaveis de reanalise.
  treino: time = mes-feature. teste: mes-feature = mes-alvo - 1.

Uso:
    python src/15_niveis_mjo.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

SAIDA = "dados_processados"
BRUTOS = f"{SAIDA}/indices_brutos"
BASE_FEATURES = "mme3"
N_PC = 8
LAGS = (1, 2, 3)
W_MIX, ANO_REC, COARSEN_DEG = 0.75, 1993, 2.0
SEMENTE = 42


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


pcs_ref = pd.read_parquet(f"{SAIDA}/pcs.parquet")
meses_feat = pcs_ref.index                          # 1940-01 a 2024-11
m_tr = np.asarray(meses_feat <= "2022-12-01")
series = {}

# ---------------------------------------------------------------------------
sep("1. ERA5 NIVEIS SUPERIORES")

arqs = sorted(glob.glob(f"{BRUTOS}/era5_niveis_*.nc"))
if not arqs:
    print("  sem arquivos era5_niveis_*.nc - pulando niveis superiores")
else:
    partes = []
    for a in arqs:
        ds = xr.open_dataset(a)
        ren = {}
        for x, y in (("latitude", "lat"), ("longitude", "lon"),
                     ("valid_time", "time"), ("pressure_level", "level")):
            if x in ds.dims or x in ds.coords:
                ren[x] = y
        partes.append(ds.rename(ren))
    ds = xr.concat(partes, dim="time").sortby("time").sortby("lat")
    ds = ds.sel(time=~ds.time.to_index().duplicated())
    ds = ds.assign_coords(time=pd.DatetimeIndex(ds.time.values).to_period("M").to_timestamp())
    print(f"  {dict(ds.sizes)}  {str(ds.time.values[0])[:7]} a {str(ds.time.values[-1])[:7]}")
    print(f"  variaveis: {list(ds.data_vars)}  niveis: {ds.level.values}")

    # nomes das variaveis no CDS: z, u, v (number e expver sao auxiliares)
    fisicas = [v for v in ds.data_vars if "level" in ds[v].dims]
    campos = {}
    for lvl in ds.level.values:
        for v in fisicas:
            campos[f"{v}{int(lvl)}"] = ds[v].sel(level=lvl)
    print(f"  campos: {list(campos)}")

    # agrega a 2 graus
    passo = float(abs(ds.lat.values[1] - ds.lat.values[0]))
    fator = max(1, int(round(COARSEN_DEG / passo)))
    print(f"  grade de {passo} grau; agregando por {fator}")

    def clim_mista(da):
        c = da.groupby("time.month").mean("time")
        r = da.sel(time=slice(f"{ANO_REC}-01-01", None)).groupby("time.month").mean("time")
        return (1 - W_MIX) * c + W_MIX * r

    todos = pd.DatetimeIndex(ds.time.values)
    assert meses_feat[-1] in todos, "ERA5 niveis nao cobre ate nov/2024"

    for nome, da in campos.items():
        da = da.coarsen(lat=fator, lon=fator, boundary="trim").mean() if fator > 1 else da
        cl = clim_mista(da.sel(time=da.time <= np.datetime64("2022-12-01")))
        an = (da.groupby("time.month") - cl).drop_vars("month", errors="ignore")
        an = an.sel(time=meses_feat)
        peso = np.sqrt(np.cos(np.deg2rad(an.lat))).astype("float32")
        A = (an * peso).values.reshape(len(meses_feat), -1)
        mu, sd = A[m_tr].mean(0), A[m_tr].std(0) + 1e-9
        A = np.nan_to_num((A - mu) / sd)
        pca = PCA(n_components=N_PC, random_state=SEMENTE)
        Y = pca.fit(A[m_tr]).transform(A)
        Y = Y / (Y[m_tr].std(0) + 1e-9)
        for i in range(N_PC):
            series[f"pc_{nome}_{i+1}"] = Y[:, i].astype("float32")
        ev = pca.explained_variance_ratio_
        print(f"  {nome:6s} var {100*ev[0]:5.1f}% {100*ev[1]:5.1f}% {100*ev[2]:5.1f}% "
              f"... acum {100*ev.sum():5.1f}%")

    # validacao: o PC1 de z200 deve responder ao ENOS (Alta da Bolivia
    # se desloca em El Nino). Compara com nino34.
    ind = pd.read_parquet(f"{SAIDA}/indices_noaa.parquet")
    nino = ind.loc[meses_feat, "nino34"].values
    print("\n  |r| com nino34 (deve haver ao menos um PC de z200 acima de 0.4):")
    for k in [k for k in series if k.startswith("pc_z200")][:4]:
        print(f"    {k:14s} {abs(np.corrcoef(series[k][m_tr], nino[m_tr])[0,1]):.3f}")


# ---------------------------------------------------------------------------
sep("2. MJO (RMM, BoM)")

arq = f"{BRUTOS}/rmm.txt"
if not os.path.exists(arq):
    print("  rmm.txt ausente - pulando MJO")
else:
    linhas = []
    with open(arq) as f:
        for l in f:
            t = l.split()
            if len(t) >= 7 and t[0].isdigit():
                try:
                    linhas.append([int(t[0]), int(t[1]), int(t[2]),
                                   float(t[3]), float(t[4]), int(t[5]), float(t[6])])
                except ValueError:
                    pass
    rmm = pd.DataFrame(linhas, columns=["ano", "mes", "dia", "rmm1", "rmm2", "fase", "amp"])
    rmm = rmm[(rmm.rmm1.abs() < 100) & (rmm.amp < 100)]      # remove missing 999
    rmm["data"] = pd.to_datetime(rmm[["ano", "mes", "dia"]].rename(
        columns={"ano": "year", "mes": "month", "dia": "day"}))
    rmm = rmm.set_index("data")
    print(f"  diario: {rmm.index[0]:%Y-%m-%d} a {rmm.index[-1]:%Y-%m-%d} ({len(rmm)} dias)")

    mensal = rmm.resample("MS").agg(rmm1=("rmm1", "mean"), rmm2=("rmm2", "mean"),
                                    amp=("amp", "mean"),
                                    fase=("fase", lambda s: s.mode().iloc[0] if len(s) else np.nan))
    mensal = mensal.reindex(meses_feat)
    # fase como sin/cos (ciclica, 8 fases)
    mensal["fase_sin"] = np.sin(2 * np.pi * mensal.fase / 8)
    mensal["fase_cos"] = np.cos(2 * np.pi * mensal.fase / 8)
    mensal = mensal.drop(columns="fase")
    nan_ini = int(mensal.isna().any(axis=1).sum())
    print(f"  mensal: {nan_ini} meses sem dado (antes de 1974) -> 0")
    assert not mensal.loc[meses_feat[-1]].isna().any(), "MJO nao cobre nov/2024"
    mensal = mensal.fillna(0.0)
    for c in mensal.columns:
        series[f"mjo_{c}"] = mensal[c].values.astype("float32")
    print(f"  colunas: {list(mensal.columns)}")

if not series:
    raise SystemExit("  nada carregado")


# ---------------------------------------------------------------------------
sep("3. DEFASAGENS E MATRIZES")

tab = pd.DataFrame(series, index=meses_feat)
base = list(tab.columns)
extras = {}
for lag in LAGS:
    for c in base:
        extras[f"{c}_lag{lag}"] = tab[c].shift(lag)
for c in base:
    extras[f"{c}_tend"] = tab[c] - tab[c].shift(3)
tab = pd.concat([tab, pd.DataFrame(extras, index=tab.index)], axis=1).fillna(0.0).astype("float32")
print(f"  {tab.shape[1]} colunas novas ({len(base)} base)")

with open(f"{SAIDA}/features_{BASE_FEATURES}.json") as f:
    nomes = json.load(f)["nomes"]
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

X = np.load(f"{SAIDA}/X_treino_{BASE_FEATURES}.npy")
n_pt = X.shape[0] // 992
idx_t = np.arange(3, 3 + 992)
bloco = tab.values[idx_t]
Xn = np.hstack([X, np.repeat(bloco, n_pt, axis=0)]).astype("float32")
del X
np.save(f"{SAIDA}/X_treino_mme4.npy", Xn)
print(f"  X_treino_mme4: {Xn.shape}  {Xn.nbytes/1e9:.2f} GB")
del Xn

Xte = np.load(f"{SAIDA}/X_teste_{BASE_FEATURES}.npy")
d_alvo = pd.to_datetime(meta_te["ano"].astype(str) + "-" + meta_te["mes_alvo"].astype(str).str.zfill(2) + "-01")
mf_te = d_alvo - pd.DateOffset(months=1)
Xten = np.hstack([Xte, tab.loc[mf_te].values]).astype("float32")
assert not np.isnan(Xten).any()
np.save(f"{SAIDA}/X_teste_mme4.npy", Xten)
print(f"  X_teste_mme4 : {Xten.shape}")

with open(f"{SAIDA}/features_mme4.json", "w") as f:
    json.dump({"nomes": nomes + list(tab.columns), "base": BASE_FEATURES,
               "novas": list(tab.columns),
               "citacao": "ERA5 (Hersbach et al. 2020) via CDS; RMM (Wheeler & Hendon 2004) via BoM"},
              f, indent=2)
print("\n  Proximo: FEATURES=mme4 RECENCIA=0 N_SEMENTES=5 python -u src/06_submissao.py")
