#!/usr/bin/env python
"""
Etapa 13 - Previsao dinamica (NMME / CFSv2) como preditor.

O QUE EH:
  O NMME publica previsoes mensais de precipitacao de modelos acoplados
  oceano-atmosfera. Usar a saida do modelo dinamico como ENTRADA de um
  modelo estatistico eh MOS (Model Output Statistics), pratica padrao em
  centros operacionais - o CPTEC inclusive. O ML aprende a corrigir o vies
  sistematico do modelo dinamico e a combina-lo com o resto da informacao.

POR QUE AGORA:
  Todo ganho real do projeto veio de informacao nova (TSM), nunca de
  modelagem. A distancia para o podio (0.033) eh maior do que a soma de
  todos os ajustes incrementais restantes. A unica fonte com esse tamanho
  eh um modelo que ve oceano e atmosfera acoplados.

REGULAMENTO E ESPIRITO:
  Secao 2.6 permite dados externos publicos, gratuitos e acessiveis a
  todos - o NMME eh. O espirito ("ML a partir de reanalise") eh
  discutivel, e por isso esta etapa eh isolada: o ganho atribuivel ao
  NMME eh medido separadamente e reportado com transparencia. A banca
  decide; o relatorio nao esconde.

ALINHAMENTO TEMPORAL (o ponto critico):
  Nosso alvo eh o mes T = M+1; as features sao de M. Usa-se a previsao
  com inicio S = T e lead 0.5, emitida no dia 1 de T com condicao inicial
  do fim de M. Nenhuma informacao posterior ao fim de M entra.

MODELO: NCEP-CFSv2
  Hindcast 1982-2010 (24 membros) + previsao operacional 2011-presente.
  Mesma versao do modelo nos dois periodos, o que evita salto de vies.
  Treino 1940-1981 fica sem NMME: coluna preenchida com 0 e indicador
  `nmme_disp` = 0, para o LightGBM separar os regimes.

FEATURES GERADAS (por ponto e mes-alvo):
  nmme_prec      previsao bruta, mm/dia, interpolada para 0.25 graus
  nmme_anom      previsao menos a climatologia do PROPRIO CFSv2 (por
                 ponto e mes, do hindcast) - remove o vies medio
  nmme_disp      1 se ha previsao para aquele mes, 0 caso contrario

FONTE (citar):
  Kirtman et al. (2014), BAMS, doi:10.1175/BAMS-D-12-00050.1.
  Dados via IRI Data Library, http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/

Uso:
    python src/13_nmme.py
"""

import json
import os

import numpy as np
import pandas as pd
import xarray as xr

DADOS = "dados_brutos"
SAIDA = "dados_processados"
BRUTOS = f"{SAIDA}/indices_brutos"
BASE_FEATURES = "sst"           # empilha sobre a versao com indices robustos


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


def abrir_iri(caminho):
    """Abre netcdf do IRI. S vem em 'months since 1960-01-01', que o xarray
    nao decodifica; converte a mao."""
    ds = xr.open_dataset(caminho, decode_times=False)
    var = [v for v in ds.data_vars if v not in ("S", "L", "M", "X", "Y")][0]
    da = ds[var]
    # remove dimensoes de tamanho 1 (L, e M se ja foi mediado)
    da = da.squeeze(drop=True)
    meses = np.asarray(ds["S"].values).astype("int64")
    tempo = pd.to_datetime("1960-01-01") + pd.to_timedelta(0, "D")
    datas = pd.DatetimeIndex([pd.Timestamp("1960-01-01") + pd.DateOffset(months=int(m))
                              for m in meses])
    da = da.rename({"S": "time", "X": "lon", "Y": "lat"}).assign_coords(time=datas)
    # longitude 0-360 -> -180..180
    lon = da.lon.values
    lon = np.where(lon > 180, lon - 360, lon)
    da = da.assign_coords(lon=lon).sortby("lon").sortby("lat")
    da = da.where(da > -1e30)
    return da


# ---------------------------------------------------------------------------
sep("1. CARREGANDO NMME (CFSv2)")

arq_h = f"{BRUTOS}/cfsv2_hind.nc"
arq_f = f"{BRUTOS}/cfsv2_fcst.nc"
for a in (arq_h, arq_f):
    if not os.path.exists(a):
        raise SystemExit(f"""
  {a} nao encontrado. Baixar no Mac (o cluster bloqueia o IRI):
    B="http://iridl.ldeo.columbia.edu/SOURCES/.Models/.NMME/.NCEP-CFSv2"
    OPS="L/0.5/VALUE/X/270/335/RANGEEDGES/Y/-60/15/RANGEEDGES/[M]average/data.nc"
    curl -g -fo cfsv2_hind.nc "$B/.HINDCAST/.MONTHLY/.prec/$OPS"
    curl -g -fo cfsv2_fcst.nc "$B/.FORECAST/.MONTHLY/.prec/$OPS"
    scp cfsv2_*.nc sd:~/era5_precip/dados_processados/indices_brutos/""")

hind = abrir_iri(arq_h)
fcst = abrir_iri(arq_f)
print(f"  hindcast : {dict(hind.sizes)}  {str(hind.time.values[0])[:7]} a "
      f"{str(hind.time.values[-1])[:7]}")
print(f"  forecast : {dict(fcst.sizes)}  {str(fcst.time.values[0])[:7]} a "
      f"{str(fcst.time.values[-1])[:7]}")
print(f"  lat {float(hind.lat.min())} a {float(hind.lat.max())}, "
      f"lon {float(hind.lon.min())} a {float(hind.lon.max())}")
print(f"  unidade: mm/dia (mesma do alvo). Media hindcast: "
      f"{float(hind.mean()):.3f}, forecast: {float(fcst.mean()):.3f}")

# concatena, sem sobreposicao (o forecast comeca onde o hindcast termina)
fcst = fcst.sel(time=fcst.time > hind.time.values[-1])
nmme = xr.concat([hind, fcst], dim="time").sortby("time")
print(f"  serie unificada: {str(nmme.time.values[0])[:7]} a "
      f"{str(nmme.time.values[-1])[:7]} ({nmme.sizes['time']} meses)")

fim_necessario = pd.Timestamp("2024-12-01")
if pd.Timestamp(nmme.time.values[-1]) < fim_necessario:
    raise SystemExit(f"  !! NMME termina antes de dez/2024. Nao cobre o teste.")

# meses faltando no meio da serie?
esperado = pd.date_range(nmme.time.values[0], nmme.time.values[-1], freq="MS")
faltando = esperado.difference(pd.DatetimeIndex(nmme.time.values))
if len(faltando):
    print(f"  !! {len(faltando)} meses faltando: {[f'{d:%Y-%m}' for d in faltando[:10]]}")
    print("     serao preenchidos com a climatologia do CFSv2 (anomalia zero)")


# ---------------------------------------------------------------------------
sep("2. CLIMATOLOGIA DO CFSv2 E ANOMALIA")

print("""  O modelo dinamico tem vies sistematico proprio (chove demais ou de
  menos em certas regioes). A anomalia em relacao a climatologia do
  PROPRIO modelo remove o vies medio; o LightGBM aprende o restante.
  Climatologia do hindcast (1982-2010), por ponto e mes calendario.
""")

clim_nmme = hind.groupby("time.month").mean("time")
anom_nmme = nmme.groupby("time.month") - clim_nmme
anom_nmme = anom_nmme.drop_vars("month", errors="ignore")
print(f"  anomalia: media {float(anom_nmme.mean()):+.4f}, "
      f"desvio {float(anom_nmme.std()):.4f} mm/dia")


# ---------------------------------------------------------------------------
sep("3. INTERPOLACAO PARA A GRADE DE 0.25 GRAUS")

tp = xr.open_dataset(f"{DADOS}/treino_tp.nc").tp
lat_alvo, lon_alvo = tp.lat.values, tp.lon.values

nmme_i = nmme.interp(lat=lat_alvo, lon=lon_alvo, method="linear")
anom_i = anom_nmme.interp(lat=lat_alvo, lon=lon_alvo, method="linear")
# bordas fora do dominio do NMME ficam NaN; preenche com vizinho mais proximo
nmme_i = nmme_i.interpolate_na("lon").interpolate_na("lat").bfill("lon").ffill("lon").bfill("lat").ffill("lat")
anom_i = anom_i.interpolate_na("lon").interpolate_na("lat").bfill("lon").ffill("lon").bfill("lat").ffill("lat")
print(f"  interpolado: {dict(nmme_i.sizes)}")
print(f"  NaN restantes: {int(nmme_i.isnull().sum())} (esperado 0)")


# ---------------------------------------------------------------------------
sep("4. VALIDACAO: O CFSv2 TEM HABILIDADE AQUI?")

print("""  Correlacao entre a anomalia prevista pelo CFSv2 e a anomalia
  observada, no periodo em que ambas existem. Se o modelo dinamico tem
  habilidade em escala mensal sobre a America do Sul, deve dar r > 0.2.
  Se der perto de zero, nao ha o que aproveitar.
""")

tp_l = tp.load()
clim_tp = tp_l.sel(time=slice("1982", "2010")).groupby("time.month").mean("time")
anom_tp = (tp_l.groupby("time.month") - clim_tp).drop_vars("month", errors="ignore")

comum = pd.DatetimeIndex(anom_i.time.values).intersection(
    pd.DatetimeIndex(anom_tp.time.values))
a = anom_i.sel(time=comum).values.reshape(len(comum), -1)
b = anom_tp.sel(time=comum).values.reshape(len(comum), -1)
ok = np.isfinite(a).all(0) & np.isfinite(b).all(0)

r_global = np.corrcoef(a[:, ok].ravel(), b[:, ok].ravel())[0, 1]
print(f"  periodo comum: {comum[0]:%Y-%m} a {comum[-1]:%Y-%m} ({len(comum)} meses)")
print(f"  r global (todos os pontos e meses): {r_global:.3f}")

# por regiao
def r_reg(la, lo):
    m = (anom_i.lat >= la[0]) & (anom_i.lat <= la[1])
    n = (anom_i.lon >= lo[0]) & (anom_i.lon <= lo[1])
    x = anom_i.sel(time=comum).where(m & n).mean(dim=("lat", "lon")).values
    y = anom_tp.sel(time=comum).where(m & n).mean(dim=("lat", "lon")).values
    return np.corrcoef(x, y)[0, 1]

for nome, la, lo in (("Amazonia norte", (-5, 2), (-70, -55)),
                     ("Nordeste      ", (-12, -4), (-45, -36)),
                     ("Sul do Brasil ", (-32, -25), (-57, -50)),
                     ("Peru/Equador  ", (-10, 2), (-82, -75))):
    print(f"    {nome}  r = {r_reg(la, lo):.3f}")

print(f"""
  Referencia: nosso LightGBM tem r ~0.20 na anomalia. Se o CFSv2 sozinho
  estiver nessa faixa, os dois combinados devem superar cada um.""")

# em 2023-24
te = pd.date_range("2023-01-01", "2024-12-01", freq="MS")
ai = anom_i.sel(time=te)
print(f"\n  anomalia media prevista pelo CFSv2 no teste: "
      f"2023 {float(ai.sel(time=slice('2023','2023')).mean()):+.3f}  "
      f"2024 {float(ai.sel(time=slice('2024','2024')).mean()):+.3f} mm/dia")


# ---------------------------------------------------------------------------
sep("5. OOD: A PREVISAO DO TESTE ESTA NA FAIXA DO TREINO?")

tr_m = pd.DatetimeIndex(anom_i.time.values) <= "2022-12-01"
p1, p99 = np.nanpercentile(anom_i.values[tr_m], [1, 99])
te_v = anom_i.sel(time=te).values
fora = ((te_v < p1) | (te_v > p99)).mean() * 100
print(f"  nmme_anom: {fora:.1f}% do teste fora de [p1, p99] do treino")
print(f"  (estavel ~2%; TNA/TSA davam 58-75%)")
if fora > 10:
    print("  !! deslocamento alto - o modelo dinamico preve algo que nunca previu")


# ---------------------------------------------------------------------------
sep("6. MATRIZES")

with open(f"{SAIDA}/features_{BASE_FEATURES}.json") as f:
    spec = json.load(f)
nomes = spec["nomes"]

meta_tr = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

tr_ds = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc")
n_lat, n_lon = tr_ds.sizes["lat"], tr_ds.sizes["lon"]
stride = (len(lat_alvo) - 1) // (n_lat - 1)
tr_ds.close()
n_pt = n_lat * n_lon

# --- treino: mes-alvo = mes-feature + 1
X = np.load(f"{SAIDA}/X_treino_{BASE_FEATURES}.npy")
n_t = len(X) // n_pt
anos = meta_tr["ano_feature"].values.reshape(n_t, n_pt)[:, 0]
mes_alvo = meta_tr["mes_alvo"].values.reshape(n_t, n_pt)[:, 0]
ano_alvo = np.where(mes_alvo == 1, anos + 1, anos)
datas_alvo = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01")
                               for a, m in zip(ano_alvo, mes_alvo)])

sub_prec = nmme_i.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
sub_anom = anom_i.isel(lat=slice(None, None, stride), lon=slice(None, None, stride))
idx_nmme = pd.DatetimeIndex(sub_prec.time.values)

bloco = np.zeros((n_t, n_pt, 3), dtype="float32")
disp = datas_alvo.isin(idx_nmme)
pos = idx_nmme.get_indexer(datas_alvo[disp])
bloco[disp, :, 0] = sub_prec.values[pos].reshape(len(pos), -1)
bloco[disp, :, 1] = sub_anom.values[pos].reshape(len(pos), -1)
bloco[disp, :, 2] = 1.0
bloco = np.nan_to_num(bloco)
print(f"  treino: {int(disp.sum())} de {n_t} meses-alvo com NMME "
      f"({datas_alvo[disp][0]:%Y-%m} em diante)")

Xn = np.hstack([X, bloco.reshape(n_t * n_pt, 3)]).astype("float32")
del X
np.save(f"{SAIDA}/X_treino_nmme.npy", Xn)
print(f"  X_treino_nmme: {Xn.shape}  {Xn.nbytes/1e9:.2f} GB")
del Xn

# --- teste: grade cheia
Xte = np.load(f"{SAIDA}/X_teste_{BASE_FEATURES}.npy")
d_te = pd.DatetimeIndex([pd.Timestamp(f"{a}-{m:02d}-01")
                         for a, m in zip(meta_te["ano"], meta_te["mes_alvo"])])
partes = pd.read_csv(f"{DADOS}/sample_submission.csv")["id"].str.split("_", expand=True)
lat_v = partes[2].astype(float).values
lon_v = partes[3].astype(float).values
ilat = np.rint((lat_v - lat_alvo[0]) / 0.25).astype(int)
ilon = np.rint((lon_v - lon_alvo[0]) / 0.25).astype(int)
ipt = ilat * len(lon_alvo) + ilon
it = pd.DatetimeIndex(nmme_i.time.values).get_indexer(d_te)
assert (it >= 0).all(), "meses do teste ausentes no NMME"

prec_full = nmme_i.values.reshape(nmme_i.sizes["time"], -1)
anom_full = anom_i.values.reshape(anom_i.sizes["time"], -1)
bloco_te = np.stack([prec_full[it, ipt], anom_full[it, ipt],
                     np.ones(len(it))], axis=1).astype("float32")
bloco_te = np.nan_to_num(bloco_te)
Xten = np.hstack([Xte, bloco_te]).astype("float32")
assert not np.isnan(Xten).any()
np.save(f"{SAIDA}/X_teste_nmme.npy", Xten)
print(f"  X_teste_nmme : {Xten.shape}")

novos = ["nmme_prec", "nmme_anom", "nmme_disp"]
with open(f"{SAIDA}/features_nmme.json", "w") as f:
    json.dump({"nomes": nomes + novos, "base": BASE_FEATURES,
               "nmme": dict(modelo="NCEP-CFSv2", lead=0.5,
                            hindcast="1982-2010", forecast="2011-2024",
                            r_global_anomalia=float(r_global)),
               "citacao": "Kirtman et al. 2014, BAMS, "
                          "doi:10.1175/BAMS-D-12-00050.1; via IRI Data Library"},
              f, indent=2)

print(f"""
  Proximo: FEATURES=nmme RECENCIA=0 python -u src/06_submissao.py

  ATENCAO na leitura do CV: os folds ate 1997 e 2002 avaliam periodos
  onde o NMME existe (1982+), mas o treino deles inclui 1940-1981 sem
  NMME. O fold ate 2017 eh o mais representativo do teste.""")