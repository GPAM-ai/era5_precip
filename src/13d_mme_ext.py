#!/usr/bin/env python
"""
Etapa 13d - MME estendido: NMME (3 modelos x 2 leads) + CanSIPS + SEAS5.

ACRESCENTA A 13c:
  1. CanSIPS costurado. O IC3 termina em jun/2024 e o IC4 comeca em
     jul/2024. Sao versoes do mesmo sistema canadense. Cada trecho eh
     anomalizado contra o proprio hindcast; se o hindcast do IC4 nao
     estiver disponivel, usa-se o do IC3 com aviso (vies pequeno,
     afeta so jul-dez/2024).

  2. ECMWF SEAS5 via Copernicus. O melhor sistema sazonal em operacao,
     ausente do NMME. Dois sistemas costurados da mesma forma: s5
     (2017 a out/2022) e s51 (nov/2022 em diante), cada um com o proprio
     hindcast 1981-2016. O loader tolera as duas convencoes de nome que
     o CDS usa (antiga: time/number; nova: forecast_reference_time/
     forecastMonth).

  Tudo que nao existir eh pulado com aviso. O script roda com o que ha.

Uso:
    python src/13d_mme_ext.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import xarray as xr

DADOS = "dados_brutos"
SAIDA = "dados_processados"
BRUTOS = f"{SAIDA}/indices_brutos"
BASE_FEATURES = "sst"
FIM = pd.Timestamp("2024-12-01")


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp.load()
lat_alvo, lon_alvo = tp.lat.values, tp.lon.values


def interp(d):
    return (d.interp(lat=lat_alvo, lon=lon_alvo, method="linear")
            .interpolate_na("lon").interpolate_na("lat")
            .bfill("lon").ffill("lon").bfill("lat").ffill("lat"))


def anomalia_vs(hind, serie):
    clim = hind.groupby("time.month").mean("time")
    return (serie.groupby("time.month") - clim).drop_vars("month", errors="ignore")


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def abrir_iri(caminho, desloc):
    ds = xr.open_dataset(caminho, decode_times=False)
    var = [v for v in ds.data_vars if v not in ("S", "L", "M", "X", "Y")][0]
    da = ds[var].squeeze(drop=True)
    meses = np.asarray(ds["S"].values).astype("int64")
    datas = pd.DatetimeIndex([pd.Timestamp("1960-01-01") + pd.DateOffset(months=int(m) + desloc)
                              for m in meses])
    da = da.rename({"S": "time", "X": "lon", "Y": "lat"}).assign_coords(time=datas)
    lon = da.lon.values
    da = da.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon").sortby("lat")
    return da.where(da > -1e30).where(da < 1e30)


def abrir_cds(caminho, lead_mes):
    """SEAS5 do CDS. Devolve DataArray (time = MES-ALVO, lat, lon) em mm/dia
    para o leadtime_month pedido (1 = mes de emissao, 2 = seguinte)."""
    ds = xr.open_dataset(caminho)
    # nomes variam entre versoes do CDS
    ren = {}
    for a, b in (("latitude", "lat"), ("longitude", "lon"),
                 ("forecast_reference_time", "time"), ("indexing_time", "time")):
        if a in ds.dims or a in ds.coords:
            ren[a] = b
    ds = ds.rename(ren)
    var = [v for v in ds.data_vars if "tp" in v or "precip" in v or "prate" in v]
    if not var:
        var = [v for v in ds.data_vars]
    da = ds[var[0]]

    # dimensao de lead
    for cand in ("forecastMonth", "leadtime_month", "forecast_month", "step"):
        if cand in da.dims:
            vals = np.asarray(da[cand].values)
            # forecastMonth eh 1,2,...; step pode vir em timedelta
            if np.issubdtype(vals.dtype, np.timedelta64):
                ordem = np.argsort(vals)
                da = da.isel({cand: ordem[lead_mes - 1]})
            else:
                da = da.sel({cand: lead_mes})
            break
    else:
        raise ValueError(f"dimensao de lead nao encontrada em {list(da.dims)}")

    # hindcasts vem com os membros individuais (dim 'number'); operacionais
    # ja vem como ensemble_mean. Media sobre os membros quando existir.
    if "number" in da.dims:
        da = da.mean("number")
    da = da.squeeze(drop=True)
    if "time" not in da.dims:
        raise ValueError(f"dimensao de tempo nao encontrada: {list(da.dims)}")

    # time = inicio; mes-alvo = inicio + (lead_mes - 1)
    t = pd.DatetimeIndex(da.time.values).to_period("M").to_timestamp()
    t = t + pd.DateOffset(months=lead_mes - 1)
    da = da.assign_coords(time=t).sortby("time")
    lon = da.lon.values
    da = da.assign_coords(lon=np.where(lon > 180, lon - 360, lon)).sortby("lon").sortby("lat")

    # unidade: CDS entrega taxa em m/s (tprate) ou acumulado em m.
    # Detecta pela magnitude: chuva media ~3.5 mm/dia = 4e-8 m/s = 0.1 m/mes
    med = float(np.nanmean(da.values))
    if med < 1e-5:                       # m/s
        da = da * 86400 * 1000
        unid = "m/s -> mm/dia"
    elif med < 1:                         # m acumulado no mes
        dias = da.time.dt.days_in_month
        da = da * 1000 / dias
        unid = "m/mes -> mm/dia"
    else:
        unid = "ja em mm/dia?"
    print(f"      unidade: {unid}, media {float(np.nanmean(da.values)):.2f} mm/dia")
    return da


# ---------------------------------------------------------------------------
sep("1. NMME (3 modelos x 2 leads)")

NMME = {
    "cfsv2": {"L05": ("cfsv2_hind.nc", "cfsv2_fcst.nc"),
              "L15": ("cfsv2_hind_L15.nc", "cfsv2_fcst_L15.nc")},
    "spear": {"L05": ("GFDL-SPEAR_hind.nc", "GFDL-SPEAR_fcst.nc"),
              "L15": ("GFDL-SPEAR_hind_L15.nc", "GFDL-SPEAR_fcst_L15.nc")},
    "nasa":  {"L05": ("NASA-GEOSS2S_hind.nc", "NASA-GEOSS2S_fcst.nc"),
              "L15": ("NASA-GEOSS2S_hind_L15.nc", "NASA-GEOSS2S_fcst_L15.nc")},
}
DESLOC = {"L05": 0, "L15": 1}
anoms, precs = {}, {}

for modelo, leads in NMME.items():
    for lead, (h, f) in leads.items():
        ah, af = f"{BRUTOS}/{h}", f"{BRUTOS}/{f}"
        chave = f"{modelo}_{lead}"
        if not (os.path.exists(ah) and os.path.exists(af)):
            print(f"  {chave:14s} ausente"); continue
        hind, fcst = abrir_iri(ah, DESLOC[lead]), abrir_iri(af, DESLOC[lead])
        fcst = fcst.sel(time=fcst.time > hind.time.values[-1])
        serie = xr.concat([hind, fcst], dim="time").sortby("time")
        if pd.Timestamp(serie.time.values[-1]) < FIM:
            print(f"  {chave:14s} termina em {str(serie.time.values[-1])[:7]}"); continue
        anoms[chave] = interp(anomalia_vs(hind, serie)).load()
        if chave == "cfsv2_L05":
            precs["cfsv2"] = interp(serie).load()
        print(f"  {chave:14s} ok  {str(serie.time.values[0])[:7]} a {str(serie.time.values[-1])[:7]}")


# ---------------------------------------------------------------------------
sep("2. CanSIPS costurado (IC3 ate jun/2024, IC4 de jul/2024)")

ic3h, ic3f = f"{BRUTOS}/CanSIPS-IC3_hind.nc", f"{BRUTOS}/CanSIPS-IC3_fcst.nc"
ic4h, ic4f = f"{BRUTOS}/CanSIPS-IC4_hind.nc", f"{BRUTOS}/CanSIPS-IC4_fcst.nc"

if os.path.exists(ic3h) and os.path.exists(ic3f) and os.path.exists(ic4f):
    h3, f3 = abrir_iri(ic3h, 0), abrir_iri(ic3f, 0)
    f4 = abrir_iri(ic4f, 0)
    f3 = f3.sel(time=f3.time > h3.time.values[-1])
    s3 = xr.concat([h3, f3], dim="time").sortby("time")
    a3 = anomalia_vs(h3, s3)

    ic4h_blocos = sorted(glob.glob(f"{BRUTOS}/CanSIPS-IC4_hind*.nc"))
    if ic4h_blocos and sum(os.path.getsize(a) for a in ic4h_blocos) > 3_000_000:
        h4 = xr.concat([abrir_iri(a, 0) for a in ic4h_blocos], dim="time").sortby("time")
        h4 = h4.sel(time=~h4.time.to_index().duplicated())
        a4 = anomalia_vs(h4, f4)
        print(f"  IC4 anomalizado contra o proprio hindcast "
              f"({str(h4.time.values[0])[:7]} a {str(h4.time.values[-1])[:7]}, "
              f"{len(ic4h_blocos)} blocos)")
    else:
        a4 = anomalia_vs(h3, f4)
        print("  !! hindcast do IC4 ausente: IC4 anomalizado contra o hindcast do IC3")
        print("     (vies pequeno, afeta so jul-dez/2024; baixar IC4_hind quando possivel)")

    # costura: IC3 ate onde vai, IC4 depois
    a4 = a4.sel(time=a4.time > a3.time.values[-1])
    if pd.Timestamp(a4.time.values[-1]) >= FIM:
        serie = xr.concat([a3, a4], dim="time").sortby("time")
        anoms["cansips_L05"] = interp(serie).load()
        print(f"  cansips_L05    ok  {str(serie.time.values[0])[:7]} a "
              f"{str(serie.time.values[-1])[:7]}  (IC3 ate {str(a3.time.values[-1])[:7]})")
    else:
        print(f"  IC4 termina em {str(a4.time.values[-1])[:7]} - cansips pulado")
else:
    print("  arquivos do CanSIPS incompletos - pulado")


# ---------------------------------------------------------------------------
def abrir_cds_multi(padrao, lead_mes):
    """Abre todos os arquivos que casam com o padrao (blocos de anos) e
    concatena no tempo. Devolve None se nao houver nenhum."""
    arqs = sorted(glob.glob(padrao))
    if not arqs:
        return None
    partes = []
    for a in arqs:
        try:
            partes.append(abrir_cds(a, lead_mes))
        except Exception as e:
            print(f"      {os.path.basename(a)}: {str(e)[:60]}")
    if not partes:
        return None
    da = xr.concat(partes, dim="time").sortby("time")
    return da.sel(time=~da.time.to_index().duplicated())


sep("3. SEAS5 (ECMWF via Copernicus)")

for lead_mes, tag in ((1, "L05"), (2, "L15")):
    print(f"  seas5_{tag}:")
    try:
        h51 = abrir_cds_multi(f"{BRUTOS}/seas5_hind_s51*.nc", lead_mes)
        h5 = abrir_cds_multi(f"{BRUTOS}/seas5_hind_s5_*.nc", lead_mes)
        if h5 is None:
            h5 = abrir_cds_multi(f"{BRUTOS}/seas5_hind_s5.nc", lead_mes)
        # globs explicitos: "s5*" pegaria tambem os arquivos "s51"
        f5 = abrir_cds_multi(f"{BRUTOS}/seas5_fcst_s5_*.nc", lead_mes)
        if f5 is None:
            f5 = abrir_cds_multi(f"{BRUTOS}/seas5_fcst_s5.nc", lead_mes)
        f51 = abrir_cds_multi(f"{BRUTOS}/seas5_fcst_s51*.nc", lead_mes)

        hind = h51 if h51 is not None else h5
        if hind is None:
            print("      nenhum hindcast disponivel - pulado"); continue

        # ALINHAMENTO DE GRADE: o s51 operacional vem com coordenadas que
        # nao batem exatamente com as do hindcast (float/offset), e a
        # subtracao da climatologia devolvia NaN em todos os 26 meses de
        # nov/2022 a dez/2024. Reamostra os operacionais para a grade do
        # hindcast antes de anomalizar.
        def alinhar(d):
            if d is None:
                return None
            mesmo = (d.sizes["lat"] == hind.sizes["lat"] and d.sizes["lon"] == hind.sizes["lon"]
                     and np.allclose(d.lat.values, hind.lat.values, atol=1e-3)
                     and np.allclose(d.lon.values, hind.lon.values, atol=1e-3))
            if mesmo:
                return d.assign_coords(lat=hind.lat.values, lon=hind.lon.values)
            print(f"      grade diferente do hindcast ({d.sizes['lat']}x{d.sizes['lon']} "
                  f"vs {hind.sizes['lat']}x{hind.sizes['lon']}); interpolando")
            return d.interp(lat=hind.lat.values, lon=hind.lon.values, method="linear")

        f5, f51 = alinhar(f5), alinhar(f51)
        if h5 is not None and h5 is not hind:
            h5 = alinhar(h5)
        n_h = hind.sizes["time"]
        print(f"      hindcast: {str(hind.time.values[0])[:7]} a "
              f"{str(hind.time.values[-1])[:7]} ({n_h} meses, "
              f"{'s51' if h51 is not None else 's5'})")
        if n_h < 300:
            print(f"      !! hindcast incompleto ({n_h} de ~432 meses); anomalia sera ruidosa")

        partes = [anomalia_vs(hind, hind)]
        if f5 is not None:
            partes.append(anomalia_vs(h5 if h5 is not None else hind, f5))
            print(f"      oper s5 : {str(f5.time.values[0])[:7]} a {str(f5.time.values[-1])[:7]}")
        if f51 is not None:
            f51 = f51.sel(time=f51.time >= np.datetime64("2022-11-01"))
            partes.append(anomalia_vs(hind, f51))
            print(f"      oper s51: {str(f51.time.values[0])[:7]} a {str(f51.time.values[-1])[:7]}")

        serie = xr.concat(partes, dim="time").sortby("time")
        serie = serie.sel(time=~serie.time.to_index().duplicated())
        if pd.Timestamp(serie.time.values[-1]) < FIM:
            print(f"      termina em {str(serie.time.values[-1])[:7]} - pulado"); continue
        # confere que 2023-24 nao ficou vazio
        te_chk = serie.sel(time=slice("2023-01-01", "2024-12-01"))
        frac_nan = float(te_chk.isnull().mean())
        if frac_nan > 0.5:
            print(f"      !! {100*frac_nan:.0f}% de NaN em 2023-24 - ainda desalinhado, pulado")
            continue
        anoms[f"seas5_{tag}"] = interp(serie).load()
        print(f"      ok  {str(serie.time.values[0])[:7]} a {str(serie.time.values[-1])[:7]}")
    except Exception as e:
        print(f"      falhou: {str(e)[:120]}")

if not anoms:
    raise SystemExit("  nada carregado")


# ---------------------------------------------------------------------------
sep("3b. DEMAIS SISTEMAS DO C3S (UKMO, Meteo-France, DWD, CMCC, JMA)")

print("""  Mesmo tratamento do SEAS5: hindcast com membros (media no loader),
  operacionais com ensemble_mean, grade dos operacionais alinhada a do
  hindcast, anomalia contra o proprio hindcast. Versoes diferentes do
  mesmo centro sao costuradas; a anomalia usa o hindcast da versao
  escolhida para 2023-24. Arquivos: c3s_<centro>_hind_*.nc / _fcst_*.nc
""")

CENTROS_C3S = ["ukmo", "meteo_france", "dwd", "cmcc", "jma"]

for centro in CENTROS_C3S:
    hind_arqs = sorted(glob.glob(f"{BRUTOS}/c3s_{centro}_hind_*.nc"))
    fcst_arqs = sorted(glob.glob(f"{BRUTOS}/c3s_{centro}_fcst_*.nc"))
    if not hind_arqs or not fcst_arqs:
        print(f"  {centro:13s} sem arquivos - pulado"); continue

    for lead_mes, tag in ((1, "L05"), (2, "L15")):
        chave = f"{centro}_{tag}"
        try:
            hind = abrir_cds_multi(f"{BRUTOS}/c3s_{centro}_hind_*.nc", lead_mes)
            fcst = abrir_cds_multi(f"{BRUTOS}/c3s_{centro}_fcst_*.nc", lead_mes)
            if hind is None or fcst is None:
                print(f"  {chave:13s} falha ao abrir - pulado"); continue

            # alinha a grade do operacional a do hindcast
            mesmo = (fcst.sizes["lat"] == hind.sizes["lat"] and fcst.sizes["lon"] == hind.sizes["lon"]
                     and np.allclose(fcst.lat.values, hind.lat.values, atol=1e-3)
                     and np.allclose(fcst.lon.values, hind.lon.values, atol=1e-3))
            if mesmo:
                fcst = fcst.assign_coords(lat=hind.lat.values, lon=hind.lon.values)
            else:
                print(f"  {chave:13s} grade {fcst.sizes['lat']}x{fcst.sizes['lon']} "
                      f"vs {hind.sizes['lat']}x{hind.sizes['lon']}; interpolando")
                fcst = fcst.interp(lat=hind.lat.values, lon=hind.lon.values, method="linear")

            fcst = fcst.sel(time=fcst.time > hind.time.values[-1])
            serie = xr.concat([anomalia_vs(hind, hind), anomalia_vs(hind, fcst)],
                              dim="time").sortby("time")
            serie = serie.sel(time=~serie.time.to_index().duplicated())
            if pd.Timestamp(serie.time.values[-1]) < FIM:
                print(f"  {chave:13s} termina em {str(serie.time.values[-1])[:7]} - pulado"); continue
            te_chk = serie.sel(time=slice("2023-01-01", "2024-12-01"))
            if float(te_chk.isnull().mean()) > 0.5:
                print(f"  {chave:13s} NaN em 2023-24 - pulado"); continue
            anoms[chave] = interp(serie).load()
            n_oper = int((pd.DatetimeIndex(serie.time.values) > hind.time.values[-1]).sum())
            print(f"  {chave:13s} ok  hind {str(hind.time.values[0])[:7]}-"
                  f"{str(hind.time.values[-1])[:7]}  + {n_oper} meses operacionais")
        except Exception as e:
            print(f"  {chave:13s} falhou: {str(e)[:100]}")

print(f"\n  {len(anoms)} series: {', '.join(anoms)}")


# ---------------------------------------------------------------------------
sep("4. HABILIDADE")

clim_tp = tp.sel(time=slice("1982", "2010")).groupby("time.month").mean("time")
anom_tp = (tp.groupby("time.month") - clim_tp).drop_vars("month", errors="ignore")
obs_idx = pd.DatetimeIndex(anom_tp.time.values)


def r_obs(da):
    comum = pd.DatetimeIndex(da.time.values).intersection(obs_idx)
    comum = comum[comum >= "1982-01-01"]
    a = da.sel(time=comum).values.reshape(len(comum), -1)
    b = anom_tp.sel(time=comum).values.reshape(len(comum), -1)
    ok = np.isfinite(a) & np.isfinite(b)          # elemento a elemento
    if ok.sum() < 1000:
        return float("nan")
    return np.corrcoef(a[ok], b[ok])[0, 1]


print("  NaN por serie (passos de tempo com QUALQUER NaN / total):")
for k, da in anoms.items():
    v = da.values.reshape(da.sizes["time"], -1)
    ruins = np.isnan(v).any(1)
    t = pd.DatetimeIndex(da.time.values)
    ex = [f"{d:%Y-%m}" for d in t[ruins][:6]]
    frac = np.isnan(v).mean() * 100
    print(f"    {k:14s} {int(ruins.sum()):4d}/{len(t)}  ({frac:.1f}% das celulas)  "
          f"{'ex: ' + ', '.join(ex) if ex else ''}")
print()


print(f"  {'serie':14s} {'r':>7s}")
for k, da in anoms.items():
    print(f"  {k:14s} {r_obs(da):7.3f}")

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
        m, s = np.nan_to_num(np.nanmean(p, 0)), np.nan_to_num(np.nanstd(p, 0))
    return m, s, np.isfinite(p[:, :, 0, 0]).sum(0)


def da_de(arr):
    return xr.DataArray(arr, dims=("time", "lat", "lon"),
                        coords={"time": todos_t, "lat": lat_alvo, "lon": lon_alvo})


k05 = [k for k in anoms if k.endswith("_L05")]
k15 = [k for k in anoms if k.endswith("_L15")]
mme05, _, _ = media_std_n(empilhar(k05))
mme15 = media_std_n(empilhar(k15))[0] if k15 else np.zeros((nT, nY, nX), "float32")
mme_all, mme_spread, mme_n = media_std_n(empilhar(list(anoms)))

print(f"\n  MME lead 0.5 ({len(k05)} membros): r = {r_obs(da_de(mme05)):.3f}")
print(f"  referencia 13b (3 NMME lead 0.5):  r = 0.431")
te = todos_t >= "2023-01-01"
print(f"  membros lead 0.5 no teste: "
      f"{np.isfinite(empilhar(k05)[:, te, 0, 0]).sum(0).mean():.1f} em media")
p1, p99 = np.percentile(mme05[~te], [1, 99])
print(f"  OOD mme05 no teste: {((mme05[te] < p1) | (mme05[te] > p99)).mean()*100:.1f}%")


# ---------------------------------------------------------------------------
sep("5. MATRIZES")

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

modelos_l05 = sorted({k.rsplit("_", 1)[0] for k in k05})
novos = [f"{k}_anom" for k in anoms] + [f"{m}_disp" for m in modelos_l05]
novos += ["mme05_anom", "mme15_anom", "mme_anom", "mme_n", "mme_spread", "mme_tend", "nmme_prec"]
n_novos = len(novos)


def bloco(datas, sub):
    rec = (lambda a: a[:, sl, sl]) if sub else (lambda a: a)
    npts = n_pt if sub else nY * nX
    out = np.zeros((len(datas), npts, n_novos), "float32")
    j = 0
    for k, da in anoms.items():
        pos = pd.DatetimeIndex(da.time.values).get_indexer(datas); ok = pos >= 0
        out[ok, :, j] = np.nan_to_num(rec(da.values)[pos[ok]].reshape(ok.sum(), -1)); j += 1
    for m in modelos_l05:
        pos = pd.DatetimeIndex(anoms[f"{m}_L05"].time.values).get_indexer(datas)
        out[pos >= 0, :, j] = 1.0; j += 1
    pos = todos_t.get_indexer(datas); ok = pos >= 0
    for arr in (mme05, mme15, mme_all):
        out[ok, :, j] = rec(arr)[pos[ok]].reshape(ok.sum(), -1); j += 1
    out[ok, :, j] = mme_n[pos[ok]][:, None]; j += 1
    out[ok, :, j] = rec(mme_spread)[pos[ok]].reshape(ok.sum(), -1); j += 1
    out[ok, :, j] = rec(mme05 - mme15)[pos[ok]].reshape(ok.sum(), -1); j += 1
    if "cfsv2" in precs:
        d = precs["cfsv2"]
        pos = pd.DatetimeIndex(d.time.values).get_indexer(datas); ok = pos >= 0
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
np.save(f"{SAIDA}/X_treino_mme3.npy", Xm)
print(f"  X_treino_mme3: {Xm.shape}  {Xm.nbytes/1e9:.2f} GB")
del Xm

Xte = np.load(f"{SAIDA}/X_teste_{BASE_FEATURES}.npy")
d_all = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01")
                          for a, m in zip(meta_te["ano"], meta_te["mes_alvo"])])
d_te = pd.DatetimeIndex(sorted(set(d_all)))
b_te = bloco(d_te, False)
partes = pd.read_csv(f"{DADOS}/sample_submission.csv")["id"].str.split("_", expand=True)
ilat = np.rint((partes[2].astype(float).values - lat_alvo[0]) / 0.25).astype(int)
ilon = np.rint((partes[3].astype(float).values - lon_alvo[0]) / 0.25).astype(int)
Xtem = np.hstack([Xte, b_te[d_te.get_indexer(d_all), ilat * nX + ilon]]).astype("float32")
assert not np.isnan(Xtem).any()
np.save(f"{SAIDA}/X_teste_mme3.npy", Xtem)
print(f"  X_teste_mme3 : {Xtem.shape}")

with open(f"{SAIDA}/features_mme3.json", "w") as f:
    json.dump({"nomes": nomes + novos, "base": BASE_FEATURES, "series": list(anoms),
               "r_mme05": float(r_obs(da_de(mme05))),
               "citacao": "NMME (Kirtman et al. 2014); CanSIPS (ECCC); "
                          "SEAS5 (Johnson et al. 2019, doi:10.5194/gmd-12-1087-2019) "
                          "via Copernicus CDS"}, f, indent=2)
print(f"\n  {n_novos} colunas. Proximo: FEATURES=mme3 RECENCIA=0 N_SEMENTES=5 "
      "python -u src/06_submissao.py")
