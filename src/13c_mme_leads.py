#!/usr/bin/env python
"""
Etapa 13c - MME com lag-ensemble (leads 0.5 e 1.5) e quarto modelo.

O QUE MUDA EM RELACAO A 13b:
  1. LAG-ENSEMBLE. Alem da previsao emitida no inicio do mes-alvo T
     (lead 0.5), entra a previsao emitida um mes antes, no inicio de M,
     para o mesmo T (lead 1.5). Sao condicoes iniciais diferentes, logo
     estimativas parcialmente independentes. Combinar leads eh pratica
     padrao do CPC e do IRI. A informacao continua dentro do corte:
     ambas usam apenas o que se sabia ate o fim de M.

  2. CanSIPS-IC3 como quarto modelo, se cobrir 2023-24.

ALINHAMENTO DO LEAD 1.5 - o ponto critico:
  A previsao com inicio S e lead L vale para o mes S + floor(L).
  Lead 0.5 com S=T vale para T. Lead 1.5 com S=T-1 tambem vale para T.
  As series de lead 1.5 sao reindexadas para o MES-ALVO (time = S + 1)
  antes de qualquer uso, para que todas as colunas falem do mesmo mes.
  A validacao contra a observacao confirma: se o realinhamento estiver
  errado por um mes, a correlacao cai para perto de zero.

FEATURES (por ponto e mes-alvo):
  <modelo>_L05_anom, <modelo>_L15_anom   anomalia de cada modelo e lead
  <modelo>_disp                          1 se o modelo tem lead 0.5
  mme05_anom, mme15_anom                 media por lead
  mme_anom                               media de tudo (modelos x leads)
  mme_n, mme_spread                      contagem e desvio entre membros
  mme_tend                               mme05 - mme15: a previsao mudou
                                         de um mes para o outro?
  nmme_prec                              CFSv2 lead 0.5 bruto

Uso:
    python src/13c_mme_leads.py
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

# (hindcast, forecast, lead)
FONTES = {
    "cfsv2":  {"L05": ("cfsv2_hind.nc", "cfsv2_fcst.nc"),
               "L15": ("cfsv2_hind_L15.nc", "cfsv2_fcst_L15.nc")},
    "spear":  {"L05": ("GFDL-SPEAR_hind.nc", "GFDL-SPEAR_fcst.nc"),
               "L15": ("GFDL-SPEAR_hind_L15.nc", "GFDL-SPEAR_fcst_L15.nc")},
    "nasa":   {"L05": ("NASA-GEOSS2S_hind.nc", "NASA-GEOSS2S_fcst.nc"),
               "L15": ("NASA-GEOSS2S_hind_L15.nc", "NASA-GEOSS2S_fcst_L15.nc")},
    "cansips": {"L05": ("CanSIPS-IC3_hind.nc", "CanSIPS-IC3_fcst.nc")},
}
DESLOC = {"L05": 0, "L15": 1}          # meses de S ate o alvo


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def abrir_iri(caminho, desloc):
    ds = xr.open_dataset(caminho, decode_times=False)
    var = [v for v in ds.data_vars if v not in ("S", "L", "M", "X", "Y")][0]
    da = ds[var].squeeze(drop=True)
    meses = np.asarray(ds["S"].values).astype("int64")
    # time = mes-ALVO = S + desloc
    datas = pd.DatetimeIndex([pd.Timestamp("1960-01-01") + pd.DateOffset(months=int(m) + desloc)
                              for m in meses])
    da = da.rename({"S": "time", "X": "lon", "Y": "lat"}).assign_coords(time=datas)
    lon = da.lon.values
    da = da.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon").sortby("lat")
    return da.where(da > -1e30).where(da < 1e30)


tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
lat_alvo, lon_alvo = tp.lat.values, tp.lon.values
FIM = pd.Timestamp("2024-12-01")


def interp(d):
    return (d.interp(lat=lat_alvo, lon=lon_alvo, method="linear")
            .interpolate_na("lon").interpolate_na("lat")
            .bfill("lon").ffill("lon").bfill("lat").ffill("lat"))


# ---------------------------------------------------------------------------
sep("1. CARREGANDO MODELOS E LEADS")

anoms = {}          # chave "cfsv2_L05" -> DataArray de anomalia (time = mes-alvo)
precs = {}
for modelo, leads in FONTES.items():
    for lead, (h, f) in leads.items():
        ah, af = f"{BRUTOS}/{h}", f"{BRUTOS}/{f}"
        chave = f"{modelo}_{lead}"
        if not (os.path.exists(ah) and os.path.exists(af)):
            print(f"  {chave:14s} ausente - pulado")
            continue
        try:
            hind = abrir_iri(ah, DESLOC[lead])
            fcst = abrir_iri(af, DESLOC[lead])
        except Exception as e:
            print(f"  {chave:14s} erro: {str(e)[:50]} - pulado")
            continue
        fcst = fcst.sel(time=fcst.time > hind.time.values[-1])
        serie = xr.concat([hind, fcst], dim="time").sortby("time")
        if pd.Timestamp(serie.time.values[-1]) < FIM:
            print(f"  {chave:14s} termina em {str(serie.time.values[-1])[:7]} - pulado")
            continue
        clim = hind.groupby("time.month").mean("time")
        anom = (serie.groupby("time.month") - clim).drop_vars("month", errors="ignore")
        anoms[chave] = interp(anom).load()
        if chave == "cfsv2_L05":
            precs["cfsv2"] = interp(serie).load()
        print(f"  {chave:14s} {str(serie.time.values[0])[:7]} a "
              f"{str(serie.time.values[-1])[:7]}  ({serie.sizes['time']} meses-alvo)")

if not anoms:
    raise SystemExit("  nada carregado")
print(f"\n  {len(anoms)} series: {', '.join(anoms)}")


# ---------------------------------------------------------------------------
sep("2. HABILIDADE POR MODELO E LEAD, E DOS MMEs")

clim_tp = tp.sel(time=slice("1982", "2010")).groupby("time.month").mean("time")
anom_tp = (tp.groupby("time.month") - clim_tp).drop_vars("month", errors="ignore")
obs_idx = pd.DatetimeIndex(anom_tp.time.values)


def r_obs(da):
    comum = pd.DatetimeIndex(da.time.values).intersection(obs_idx)
    comum = comum[comum >= "1982-01-01"]
    a = da.sel(time=comum).values.reshape(len(comum), -1)
    b = anom_tp.sel(time=comum).values.reshape(len(comum), -1)
    ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)
    return np.corrcoef(a[:, ok].ravel(), b[:, ok].ravel())[0, 1]


print(f"  {'serie':14s} {'r':>7s}")
for k, da in anoms.items():
    print(f"  {k:14s} {r_obs(da):7.3f}")

print("""
  Lead 1.5 deve ter r menor que lead 0.5 do mesmo modelo (previsao mais
  antiga = menos habilidade). Se lead 1.5 der r ~0, o realinhamento
  esta errado. Se der MAIOR que 0.5, tambem - investigar.""")

# grades temporais
todos_t = pd.DatetimeIndex(sorted(set().union(*[set(pd.DatetimeIndex(d.time.values))
                                                for d in anoms.values()])))
todos_t = todos_t[(todos_t >= "1982-01-01") & (todos_t <= FIM)]
nT, nY, nX = len(todos_t), len(lat_alvo), len(lon_alvo)


def empilhar(chaves):
    p = np.full((len(chaves), nT, nY, nX), np.nan, "float32")
    for k, ch in enumerate(chaves):
        idx = pd.DatetimeIndex(anoms[ch].time.values)
        pos = idx.get_indexer(todos_t)
        ok = pos >= 0
        p[k, ok] = anoms[ch].values[pos[ok]]
    return p


def media_std_n(p):
    with np.errstate(all="ignore"):
        m = np.nan_to_num(np.nanmean(p, 0))
        s = np.nan_to_num(np.nanstd(p, 0))
    n = np.isfinite(p[:, :, 0, 0]).sum(0)
    return m, s, n


k05 = [k for k in anoms if k.endswith("_L05")]
k15 = [k for k in anoms if k.endswith("_L15")]
mme05, _, n05 = media_std_n(empilhar(k05))
mme15, _, n15 = media_std_n(empilhar(k15)) if k15 else (np.zeros((nT, nY, nX), "float32"),
                                                        None, np.zeros(nT, int))
mme_all, mme_spread, mme_n = media_std_n(empilhar(list(anoms)))

def da_de(arr):
    return xr.DataArray(arr, dims=("time", "lat", "lon"),
                        coords={"time": todos_t, "lat": lat_alvo, "lon": lon_alvo})

print(f"\n  {'MME':14s} {'r':>7s} {'membros':>9s}")
print(f"  {'lead 0.5':14s} {r_obs(da_de(mme05)):7.3f} {len(k05):9d}")
if k15:
    print(f"  {'lead 1.5':14s} {r_obs(da_de(mme15)):7.3f} {len(k15):9d}")
print(f"  {'todos':14s} {r_obs(da_de(mme_all)):7.3f} {len(anoms):9d}")
print(f"\n  referencia 13b (3 modelos, lead 0.5): r = 0.431")

te = todos_t >= "2023-01-01"
p1, p99 = np.percentile(mme_all[~te], [1, 99])
print(f"  OOD mme_anom no teste: {((mme_all[te] < p1) | (mme_all[te] > p99)).mean()*100:.1f}%")
print(f"  membros no teste: {mme_n[te].mean():.1f} em media")


# ---------------------------------------------------------------------------
sep("3. MATRIZES")

with open(f"{SAIDA}/features_{BASE_FEATURES}.json") as f:
    nomes = json.load(f)["nomes"]
meta_tr = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")
tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
stride = (nY - 1) // (n_lat - 1)
tr_ds.close()
n_pt = n_lat * n_lon
sl = slice(None, None, stride)

novos = [f"{k}_anom" for k in anoms]
novos += [f"{m}_disp" for m in FONTES if f"{m}_L05" in anoms]
novos += ["mme05_anom", "mme15_anom", "mme_anom", "mme_n", "mme_spread",
          "mme_tend", "nmme_prec"]
n_novos = len(novos)


def bloco(datas, sub):
    rec = (lambda a: a[:, sl, sl]) if sub else (lambda a: a)
    npts = n_pt if sub else nY * nX
    out = np.zeros((len(datas), npts, n_novos), "float32")
    j = 0
    for k, da in anoms.items():
        idx = pd.DatetimeIndex(da.time.values)
        pos = idx.get_indexer(datas)
        ok = pos >= 0
        out[ok, :, j] = np.nan_to_num(rec(da.values)[pos[ok]].reshape(ok.sum(), -1))
        j += 1
    for m in FONTES:
        if f"{m}_L05" not in anoms:
            continue
        idx = pd.DatetimeIndex(anoms[f"{m}_L05"].time.values)
        out[idx.get_indexer(datas) >= 0, :, j] = 1.0
        j += 1
    pos = todos_t.get_indexer(datas)
    ok = pos >= 0
    for arr in (mme05, mme15, mme_all):
        out[ok, :, j] = rec(arr)[pos[ok]].reshape(ok.sum(), -1)
        j += 1
    out[ok, :, j] = mme_n[pos[ok]][:, None]; j += 1
    out[ok, :, j] = rec(mme_spread)[pos[ok]].reshape(ok.sum(), -1); j += 1
    out[ok, :, j] = rec(mme05 - mme15)[pos[ok]].reshape(ok.sum(), -1); j += 1
    if "cfsv2" in precs:
        d = precs["cfsv2"]
        idx = pd.DatetimeIndex(d.time.values)
        pos = idx.get_indexer(datas)
        ok = pos >= 0
        out[ok, :, j] = np.nan_to_num(rec(d.values)[pos[ok]].reshape(ok.sum(), -1))
    return out


X = np.load(f"{SAIDA}/X_treino_{BASE_FEATURES}.npy")
n_t = len(X) // n_pt
anos = meta_tr["ano_feature"].values.reshape(n_t, n_pt)[:, 0]
mes_alvo = meta_tr["mes_alvo"].values.reshape(n_t, n_pt)[:, 0]
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)
datas_tr = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01") for a, m in zip(ano_alvo, mes_alvo)])
Xm = np.hstack([X, bloco(datas_tr, True).reshape(n_t * n_pt, n_novos)]).astype("float32")
del X
np.save(f"{SAIDA}/X_treino_mme2.npy", Xm)
print(f"  X_treino_mme2: {Xm.shape}  {Xm.nbytes/1e9:.2f} GB")
del Xm

Xte = np.load(f"{SAIDA}/X_teste_{BASE_FEATURES}.npy")
d_all = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01")
                          for a, m in zip(meta_te["ano"], meta_te["mes_alvo"])])
d_te = pd.DatetimeIndex(sorted(set(d_all)))
b_te = bloco(d_te, False)
partes = pd.read_csv(f"{DADOS}/sample_submission.csv")["id"].str.split("_", expand=True)
ilat = np.rint((partes[2].astype(float).values - lat_alvo[0]) / 0.25).astype(int)
ilon = np.rint((partes[3].astype(float).values - lon_alvo[0]) / 0.25).astype(int)
ipt = ilat * nX + ilon
Xtem = np.hstack([Xte, b_te[d_te.get_indexer(d_all), ipt]]).astype("float32")
assert not np.isnan(Xtem).any()
np.save(f"{SAIDA}/X_teste_mme2.npy", Xtem)
print(f"  X_teste_mme2 : {Xtem.shape}")

with open(f"{SAIDA}/features_mme2.json", "w") as f:
    json.dump({"nomes": nomes + novos, "base": BASE_FEATURES,
               "series": list(anoms), "r_mme_todos": float(r_obs(da_de(mme_all))),
               "citacao": "NMME Phase II; Kirtman et al. 2014; via IRI Data Library"},
              f, indent=2)
print(f"\n  {n_novos} colunas: {', '.join(novos)}")
print("  Proximo: FEATURES=mme2 RECENCIA=0 python -u src/06_submissao.py")