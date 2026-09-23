#!/usr/bin/env python
"""
Etapa 13b - Ensemble multimodelo (MME) do NMME.

POR QUE:
  O CFSv2 sozinho levou o modelo de 1.750 para 1.632 no placar publico.
  A razao de existir do NMME eh que a MEDIA de varios modelos dinamicos
  supera consistentemente o melhor modelo individual: os vieses sao
  parcialmente independentes e se cancelam na media, enquanto o sinal
  comum se reforca. Esta etapa acrescenta os outros modelos do consorcio
  com hindcast e operacional cobrindo 2023-24.

MODELOS E PERIODOS (IRI Data Library):
  NCEP-CFSv2         hindcast 1982-2010   operacional 2011-2025
  GFDL-SPEAR         hindcast 1991-2020   operacional 2020-2025
  NASA-GEOSS2S       hindcast 1981-2017   operacional 2017-2025
  COLA-RSMAS-CCSM4   hindcast 1982-2010   operacional 2011-2025 (se disponivel)

  Os hindcasts tem periodos diferentes. A anomalia de cada modelo eh
  calculada em relacao a climatologia do PROPRIO hindcast, o que remove
  o vies medio de cada um antes de combinar. Onde hindcast e operacional
  se sobrepoem, prevalece o hindcast (versao fixa do modelo).

FEATURES GERADAS (por ponto e mes-alvo):
  <modelo>_anom   anomalia de cada modelo (0 onde nao existe)
  <modelo>_disp   1 se aquele modelo tem previsao para o mes
  mme_anom        media das anomalias dos modelos disponiveis
  mme_n           quantos modelos entraram na media
  mme_spread      desvio entre os modelos (concordancia = confianca)
  nmme_prec       previsao bruta do CFSv2 (mantida de 13)

  O LightGBM recebe tanto os modelos individuais quanto a media: se um
  modelo for melhor numa regiao, ele aprende a pesar; se a media for
  melhor, tambem.

ALINHAMENTO: identico a 13 - inicio S = mes-alvo, lead 0.5.

Uso:
    python src/13b_mme.py
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr

DADOS = "dados_brutos"
SAIDA = "dados_processados"
BRUTOS = f"{SAIDA}/indices_brutos"
BASE_FEATURES = "sst"

MODELOS = {
    "cfsv2":  ("cfsv2_hind.nc", "cfsv2_fcst.nc"),
    "spear":  ("GFDL-SPEAR_hind.nc", "GFDL-SPEAR_fcst.nc"),
    "nasa":   ("NASA-GEOSS2S_hind.nc", "NASA-GEOSS2S_fcst.nc"),
    "ccsm4":  ("COLA-RSMAS-CCSM4_hind.nc", "COLA-RSMAS-CCSM4_fcst.nc"),
}


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def abrir_iri(caminho):
    ds = xr.open_dataset(caminho, decode_times=False)
    var = [v for v in ds.data_vars if v not in ("S", "L", "M", "X", "Y")][0]
    da = ds[var].squeeze(drop=True)
    meses = np.asarray(ds["S"].values).astype("int64")
    datas = pd.DatetimeIndex([pd.Timestamp("1960-01-01") + pd.DateOffset(months=int(m))
                              for m in meses])
    da = da.rename({"S": "time", "X": "lon", "Y": "lat"}).assign_coords(time=datas)
    lon = da.lon.values
    da = da.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon").sortby("lat")
    return da.where(da > -1e30).where(da < 1e30)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO OS MODELOS")

tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
lat_alvo, lon_alvo = tp.lat.values, tp.lon.values
fim_necessario = pd.Timestamp("2024-12-01")

anoms = {}
precs = {}
for nome, (h, f) in MODELOS.items():
    ah, af = f"{BRUTOS}/{h}", f"{BRUTOS}/{f}"
    if not (os.path.exists(ah) and os.path.exists(af)):
        print(f"  {nome:7s} ausente ({h} / {f}) - pulado")
        continue
    try:
        hind, fcst = abrir_iri(ah), abrir_iri(af)
    except Exception as e:
        print(f"  {nome:7s} erro ao abrir: {str(e)[:60]} - pulado")
        continue

    fcst = fcst.sel(time=fcst.time > hind.time.values[-1])
    serie = xr.concat([hind, fcst], dim="time").sortby("time")
    if pd.Timestamp(serie.time.values[-1]) < fim_necessario:
        print(f"  {nome:7s} termina em {str(serie.time.values[-1])[:7]}, "
              "antes de dez/2024 - pulado")
        continue

    clim = hind.groupby("time.month").mean("time")
    anom = (serie.groupby("time.month") - clim).drop_vars("month", errors="ignore")

    interp = lambda d: (d.interp(lat=lat_alvo, lon=lon_alvo, method="linear")
                        .interpolate_na("lon").interpolate_na("lat")
                        .bfill("lon").ffill("lon").bfill("lat").ffill("lat"))
    anoms[nome] = interp(anom).load()
    if nome == "cfsv2":
        precs[nome] = interp(serie).load()

    print(f"  {nome:7s} hind {str(hind.time.values[0])[:7]}-{str(hind.time.values[-1])[:7]}"
          f"  oper {str(fcst.time.values[0])[:7]}-{str(fcst.time.values[-1])[:7]}"
          f"  ({serie.sizes['time']} meses)  media {float(serie.mean()):.2f} mm/dia")

if not anoms:
    raise SystemExit("  nenhum modelo carregado")
print(f"\n  {len(anoms)} modelos: {', '.join(anoms)}")


# ---------------------------------------------------------------------------
sep("2. HABILIDADE INDIVIDUAL E DO MME")

clim_tp = tp.sel(time=slice("1982", "2010")).groupby("time.month").mean("time")
anom_tp = (tp.groupby("time.month") - clim_tp).drop_vars("month", errors="ignore")
obs_idx = pd.DatetimeIndex(anom_tp.time.values)


def r_com_obs(da):
    comum = pd.DatetimeIndex(da.time.values).intersection(obs_idx)
    comum = comum[comum >= "1982-01-01"]
    a = da.sel(time=comum).values.reshape(len(comum), -1)
    b = anom_tp.sel(time=comum).values.reshape(len(comum), -1)
    ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    return np.corrcoef(a[:, ok].ravel(), b[:, ok].ravel())[0, 1], comum


print(f"  {'modelo':8s} {'r global':>9s} {'periodo':>18s}")
for nome, da in anoms.items():
    r, comum = r_com_obs(da)
    print(f"  {nome:8s} {r:9.3f} {comum[0]:%Y-%m} a {comum[-1]:%Y-%m}")

# MME: media das anomalias dos modelos disponiveis em cada mes
todos_t = sorted(set().union(*[set(pd.DatetimeIndex(d.time.values)) for d in anoms.values()]))
todos_t = pd.DatetimeIndex(todos_t)
todos_t = todos_t[(todos_t >= "1982-01-01") & (todos_t <= "2024-12-01")]

pilha = np.full((len(anoms), len(todos_t), len(lat_alvo), len(lon_alvo)), np.nan, "float32")
for k, (nome, da) in enumerate(anoms.items()):
    idx = pd.DatetimeIndex(da.time.values)
    pos = idx.get_indexer(todos_t)
    ok = pos >= 0
    pilha[k, ok] = da.values[pos[ok]]

mme_n = np.isfinite(pilha[:, :, 0, 0]).sum(0)
with np.errstate(all="ignore"):
    mme_anom = np.nanmean(pilha, axis=0)
    mme_spread = np.nanstd(pilha, axis=0)
mme_anom = np.nan_to_num(mme_anom)
mme_spread = np.nan_to_num(mme_spread)

mme_da = xr.DataArray(mme_anom, dims=("time", "lat", "lon"),
                      coords={"time": todos_t, "lat": lat_alvo, "lon": lon_alvo})
r_mme, _ = r_com_obs(mme_da)
print(f"\n  {'MME':8s} {r_mme:9.3f}   (media dos disponiveis)")

print(f"\n  modelos disponiveis por periodo:")
for a in (1985, 1995, 2005, 2015, 2023, 2024):
    m = todos_t.year == a
    print(f"    {a}: {mme_n[m].mean():.1f} modelos em media")

print("""
  Se r(MME) > max r individual, a media esta cancelando vies e o
  LightGBM vai se beneficiar. Se r(MME) <= melhor individual, os
  modelos extras trazem mais ruido que sinal - ainda pode ajudar como
  feature individual, mas a expectativa cai.""")

# OOD do MME no teste
te = (todos_t >= "2023-01-01")
tr = ~te
p1, p99 = np.percentile(mme_anom[tr], [1, 99])
fora = ((mme_anom[te] < p1) | (mme_anom[te] > p99)).mean() * 100
print(f"\n  OOD do mme_anom no teste: {fora:.1f}% fora de [p1,p99]")


# ---------------------------------------------------------------------------
sep("3. MATRIZES")

with open(f"{SAIDA}/features_{BASE_FEATURES}.json") as f:
    nomes = json.load(f)["nomes"]
meta_tr = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
stride = (len(lat_alvo) - 1) // (n_lat - 1)
tr_ds.close()
n_pt = n_lat * n_lon
sl = slice(None, None, stride)

# colunas: por modelo (anom, disp), mme (anom, n, spread), cfsv2 prec bruto
novos = []
for nome in anoms:
    novos += [f"{nome}_anom", f"{nome}_disp"]
novos += ["mme_anom", "mme_n", "mme_spread", "nmme_prec"]
n_novos = len(novos)


def bloco_para(datas, sub):
    """Monta (n_datas, n_pontos, n_novos) para uma lista de meses-alvo.
    sub=True usa a grade subamostrada (treino); False, a cheia (teste)."""
    if sub:
        rec = lambda d: d.isel(lat=sl, lon=sl)
        npts = n_pt
    else:
        rec = lambda d: d
        npts = len(lat_alvo) * len(lon_alvo)
    out = np.zeros((len(datas), npts, n_novos), "float32")
    j = 0
    for nome, da in anoms.items():
        d = rec(da)
        idx = pd.DatetimeIndex(d.time.values)
        pos = idx.get_indexer(datas)
        ok = pos >= 0
        out[ok, :, j] = np.nan_to_num(d.values[pos[ok]].reshape(ok.sum(), -1))
        out[ok, :, j + 1] = 1.0
        j += 2
    pos = todos_t.get_indexer(datas)
    ok = pos >= 0
    ma = rec(mme_da).values
    ms = mme_spread[:, sl, sl] if sub else mme_spread
    out[ok, :, j] = ma[pos[ok]].reshape(ok.sum(), -1)
    out[ok, :, j + 1] = mme_n[pos[ok]][:, None]
    out[ok, :, j + 2] = ms[pos[ok]].reshape(ok.sum(), -1)
    j += 3
    if "cfsv2" in precs:
        d = rec(precs["cfsv2"])
        idx = pd.DatetimeIndex(d.time.values)
        pos = idx.get_indexer(datas)
        ok = pos >= 0
        out[ok, :, j] = np.nan_to_num(d.values[pos[ok]].reshape(ok.sum(), -1))
    return out


# treino
X = np.load(f"{SAIDA}/X_treino_{BASE_FEATURES}.npy")
n_t = len(X) // n_pt
anos = meta_tr["ano_feature"].values.reshape(n_t, n_pt)[:, 0]
mes_alvo = meta_tr["mes_alvo"].values.reshape(n_t, n_pt)[:, 0]
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)
datas_tr = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01") for a, m in zip(ano_alvo, mes_alvo)])
b_tr = bloco_para(datas_tr, sub=True)
Xm = np.hstack([X, b_tr.reshape(n_t * n_pt, n_novos)]).astype("float32")
del X, b_tr
np.save(f"{SAIDA}/X_treino_mme.npy", Xm)
print(f"  X_treino_mme: {Xm.shape}  {Xm.nbytes/1e9:.2f} GB")
del Xm

# teste
Xte = np.load(f"{SAIDA}/X_teste_{BASE_FEATURES}.npy")
d_te_all = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01")
                             for a, m in zip(meta_te["ano"], meta_te["mes_alvo"])])
d_te = pd.DatetimeIndex(sorted(set(d_te_all)))
b_te = bloco_para(d_te, sub=False)                    # (24, 78561, n_novos)
partes = pd.read_csv(f"{DADOS}/sample_submission.csv")["id"].str.split("_", expand=True)
ilat = np.rint((partes[2].astype(float).values - lat_alvo[0]) / 0.25).astype(int)
ilon = np.rint((partes[3].astype(float).values - lon_alvo[0]) / 0.25).astype(int)
ipt = ilat * len(lon_alvo) + ilon
it = d_te.get_indexer(d_te_all)
Xtem = np.hstack([Xte, b_te[it, ipt]]).astype("float32")
assert not np.isnan(Xtem).any()
np.save(f"{SAIDA}/X_teste_mme.npy", Xtem)
print(f"  X_teste_mme : {Xtem.shape}")

with open(f"{SAIDA}/features_mme.json", "w") as f:
    json.dump({"nomes": nomes + novos, "base": BASE_FEATURES,
               "modelos": list(anoms), "r_mme": float(r_mme),
               "citacao": "NMME Phase II; Kirtman et al. 2014, BAMS, "
                          "doi:10.1175/BAMS-D-12-00050.1; via IRI Data Library"},
              f, indent=2)

print(f"""
  {n_novos} colunas novas: {', '.join(novos)}
  Proximo: FEATURES=mme RECENCIA=0 python -u src/06_submissao.py""")