#!/usr/bin/env python
"""
Etapa 11 - TSM em grade (ERSST v5): padroes espaciais do oceano.

POR QUE:
  Os 8 indices da NOAA deram o maior salto do projeto (1.768 -> 1.750).
  Sao 8 numeros por mes resumindo o oceano inteiro. O campo completo de
  TSM, decomposto em EOFs como fizemos com a pressao em 03, entrega o
  padrao espacial: a forma do El Nino (leste vs central), o dipolo do
  Atlantico em toda a bacia, o Indico. Eh a mesma alavanca, ampliada.

A LICAO DE OOD APLICADA (diagnostico em 10):
  TNA e TSA absolutos estavam fora da distribuicao em 2023/24 - o oceano
  inteiro bateu recorde de temperatura. TSM em grade bruta repetiria o
  problema em cada ponto. Dois cuidados, ambos obrigatorios:

  1. ANOMALIA MOVEL. Para cada ponto e mes calendario, a climatologia eh
     a media dos 30 anos ANTERIORES, nao uma base fixa. Um recorde
     absoluto vira "desvio em relacao ao esperado para a epoca", que o
     modelo ja viu. ERSST comeca em 1854, entao ha historico suficiente
     para 1940 em diante sem perder nenhum ano.

  2. REMOCAO DA MEDIA GLOBAL. A cada mes, subtrai-se a media (ponderada
     por area) da anomalia. Isso elimina o modo de aquecimento uniforme,
     que em TSM global eh sempre o EOF 1 e nao carrega padrao. Sobram
     os modos de forma: ENOS, dipolos, PDO.

  A validacao ao final mede quantos PCs do teste caem fora do intervalo
  de treino. Se a fracao for a de uma distribuicao estavel (~2%), o
  desenho funcionou.

FONTE (citar):
  NOAA ERSST v5, Huang et al. (2017), doi:10.7289/V5T72FNM.
  Distribuido pelo NOAA/PSL, https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
  Licenca: sem restricoes de acesso ou uso.

Uso:
    python src/11_sst_grade.py [n_pc]
"""

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

SAIDA = "dados_processados"
ARQ = f"{SAIDA}/indices_brutos/sst.mnmean.nc"
N_PC = int(sys.argv[1]) if len(sys.argv) > 1 else 12
JANELA_CLIM = 30
LAT_MIN, LAT_MAX = -40, 30          # faixa tropical e subtropical, global
LAGS = (1, 2, 3)
SEMENTE = 42


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO ERSST v5")

if not os.path.exists(ARQ):
    raise SystemExit(f"""
  {ARQ} nao encontrado. O cluster bloqueia a NOAA; baixar no Mac:
    curl -fO https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc
    scp sst.mnmean.nc sd:~/era5_precip/dados_processados/indices_brutos/""")

ds = xr.open_dataset(ARQ)
sst = ds["sst"]
print(f"  dims: {dict(sst.sizes)}")
print(f"  periodo: {str(sst.time.values[0])[:7]} a {str(sst.time.values[-1])[:7]}")
print(f"  lat: {float(sst.lat.min())} a {float(sst.lat.max())}  "
      f"lon: {float(sst.lon.min())} a {float(sst.lon.max())}")

# recorte espacial; ERSST usa lat decrescente em alguns arquivos
if float(sst.lat[0]) > float(sst.lat[-1]):
    sst = sst.sel(lat=slice(LAT_MAX, LAT_MIN))
else:
    sst = sst.sel(lat=slice(LAT_MIN, LAT_MAX))
sst = sst.load()

# oceano vs terra: terra eh NaN (ou valor de missing)
if "missing_value" in sst.attrs:
    sst = sst.where(sst != sst.attrs["missing_value"])
sst = sst.where(sst > -5)            # descarta sentinelas como -9.96e36
oceano = ~sst.isel(time=0).isnull()
print(f"  recorte {LAT_MIN} a {LAT_MAX}: {dict(sst.sizes)}, "
      f"{int(oceano.sum())} celulas oceanicas de {oceano.size}")


# ---------------------------------------------------------------------------
sep(f"2. ANOMALIA MOVEL ({JANELA_CLIM} anos anteriores)")

print("""  Para o mes t, climatologia = media dos mesmos meses calendario nos
  30 anos anteriores a t. Assim 2023 eh comparado a 1993-2022, e 1940 a
  1910-1939. A tendencia de longo prazo fica embutida na referencia em
  vez de aparecer na anomalia.
""")

# precisa de dados desde 1940 - 30 = 1910
ini_uso = pd.Timestamp("1940-01-01")
sst_uso = sst.sel(time=slice(f"{ini_uso.year - JANELA_CLIM}-01-01", None))
anos = sst_uso.time.dt.year.values
meses = sst_uso.time.dt.month.values
valores = sst_uso.values                                  # (t, lat, lon)

anom = np.full_like(valores, np.nan, dtype="float32")
alvo_idx = np.where(anos >= ini_uso.year)[0]

for i in alvo_idx:
    m = (meses == meses[i]) & (anos >= anos[i] - JANELA_CLIM) & (anos < anos[i])
    anom[i] = valores[i] - np.nanmean(valores[m], axis=0)

anom = anom[alvo_idx]
tempo = pd.to_datetime(sst_uso.time.values[alvo_idx])
lat = sst_uso.lat.values
lon = sst_uso.lon.values
print(f"  anomalias: {anom.shape}, de {tempo[0]:%Y-%m} a {tempo[-1]:%Y-%m}")


# ---------------------------------------------------------------------------
sep("3. REMOCAO DA MEDIA GLOBAL PONDERADA")

peso_lat = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
peso_lat = np.where(np.isnan(anom[0]), 0, peso_lat)
media_glob = np.nansum(anom * peso_lat, axis=(1, 2)) / peso_lat.sum()
anom_c = anom - media_glob[:, None, None]

print(f"  media global da anomalia antes: {np.nanmean(media_glob):+.4f} K")
print(f"  em 2023-24: {media_glob[tempo >= '2023-01-01'].mean():+.4f} K")
print("  (positivo = oceano mais quente que os 30 anos anteriores;")
print("   esse aquecimento residual eh o que removemos)")

# padroniza por celula (desvio do treino) e achata para (t, celulas)
mask_treino = tempo <= "2022-12-01"
achat = anom_c.reshape(len(tempo), -1)
oce = ~np.isnan(achat[0])
achat = achat[:, oce]
sd_cel = np.nanstd(achat[mask_treino], axis=0) + 1e-6
achat = np.nan_to_num(achat / sd_cel)
w = np.sqrt(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, len(lon)))
w = w.ravel()[oce]
achat = (achat * w).astype("float32")
print(f"  matriz para PCA: {achat.shape}  (meses x celulas oceanicas)")


# ---------------------------------------------------------------------------
sep(f"4. EOFs ({N_PC} componentes, ajuste so no treino)")

# a serie de meses-feature vai ate nov/2024; ERSST cobre isso
pcs_ref = pd.read_parquet(f"{SAIDA}/pcs.parquet")
meses_feat = pcs_ref.index
assert meses_feat[-1] <= tempo[-1], "ERSST nao cobre o ultimo mes-feature"

pos = pd.Index(tempo).get_indexer(meses_feat)
assert (pos >= 0).all(), "meses-feature ausentes no ERSST"
A = achat[pos]                                   # (1019, celulas)
m_tr = np.asarray(meses_feat <= "2022-12-01")

pca = PCA(n_components=N_PC, random_state=SEMENTE)
Y_tr = pca.fit_transform(A[m_tr])
Y = pca.transform(A)
esc = Y_tr.std(0) + 1e-9
Y = (Y / esc).astype("float32")

ev = pca.explained_variance_ratio_
print("  variancia explicada:", " ".join(f"{100*v:.1f}%" for v in ev[:6]),
      f"... acumulada {100*ev.sum():.1f}%")


# ---------------------------------------------------------------------------
sep("5. VALIDACAO")

ind = pd.read_parquet(f"{SAIDA}/indices_noaa.parquet")
comum = meses_feat.intersection(ind.index)
nino = ind.loc[comum, "nino34"].values
dip = ind.loc[comum, "dipolo_atl"].values if "dipolo_atl" in ind else None
Yc = Y[meses_feat.get_indexer(comum)]

print("  correlacao dos PCs com indices conhecidos (o modo de ENOS deve")
print("  aparecer nos primeiros; sinal eh arbitrario):\n")
print(f"  {'PC':>5s} {'var%':>6s} {'|r| nino34':>11s} {'|r| dipolo':>11s}")
for k in range(min(N_PC, 8)):
    r_n = abs(np.corrcoef(Yc[:, k], nino)[0, 1])
    r_d = abs(np.corrcoef(Yc[:, k], dip)[0, 1]) if dip is not None else np.nan
    marca = "  <-- ENOS" if r_n > 0.7 else ("  <-- Atlantico" if r_d > 0.5 else "")
    print(f"  {k+1:5d} {100*ev[k]:6.1f} {r_n:11.3f} {r_d:11.3f}{marca}")

print("\n  DESLOCAMENTO treino x teste (a razao de tudo isso):")
te = ~m_tr
p1, p99 = np.percentile(Y[m_tr], [1, 99], axis=0)
fora = ((Y[te] < p1) | (Y[te] > p99)).mean(axis=0) * 100
print(f"  {'PC':>5s} {'fora p1-p99 no teste':>22s}")
for k in range(min(N_PC, 8)):
    marca = "  <-- OOD" if fora[k] > 15 else ""
    print(f"  {k+1:5d} {fora[k]:21.1f}%{marca}")
print(f"\n  media: {fora.mean():.1f}%  (distribuicao estavel: ~2%;")
print("   TNA/TSA absolutos davam 58-75%)")

if fora.mean() > 10:
    print("""
  !! deslocamento ainda alto. Verificar se a media global foi removida e
     se a janela movel esta funcionando. NAO usar antes de resolver.""")


# ---------------------------------------------------------------------------
sep("6. TABELA E MATRIZES")

cols = [f"sst_pc{k+1}" for k in range(N_PC)]
tab = pd.DataFrame(Y, index=meses_feat, columns=cols)
extras = {}
for lag in LAGS:
    for c in cols:
        extras[f"{c}_lag{lag}"] = tab[c].shift(lag)
for c in cols:
    extras[f"{c}_tend"] = tab[c] - tab[c].shift(3)
tab = pd.concat([tab, pd.DataFrame(extras, index=tab.index)], axis=1)
tab = tab.fillna(0.0).astype("float32")
tab.index.name = "mes_feature"
tab.to_parquet(f"{SAIDA}/sst_grade_pcs.parquet")
print(f"  {tab.shape[1]} colunas de TSM em grade")

# empilha sobre a versao com indices robustos (X_*_sst)
with open(f"{SAIDA}/features_sst.json") as f:
    spec = json.load(f)
nomes = spec["nomes"]

X = np.load(f"{SAIDA}/X_treino_sst.npy")
n_pt = X.shape[0] // 992
idx_t = np.arange(3, 3 + 992)
bloco = tab.values[idx_t]
Xg = np.hstack([X, np.repeat(bloco, n_pt, axis=0)]).astype("float32")
del X
np.save(f"{SAIDA}/X_treino_grade.npy", Xg)
print(f"  X_treino_grade: {Xg.shape}  {Xg.nbytes/1e9:.2f} GB")
del Xg

Xte = np.load(f"{SAIDA}/X_teste_sst.npy")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")
data_alvo = pd.to_datetime(meta_te["ano"].astype(str) + "-"
                           + meta_te["mes_alvo"].astype(str).str.zfill(2) + "-01")
mes_feat_te = data_alvo - pd.DateOffset(months=1)
Xteg = np.hstack([Xte, tab.loc[mes_feat_te].values]).astype("float32")
assert not np.isnan(Xteg).any()
np.save(f"{SAIDA}/X_teste_grade.npy", Xteg)
print(f"  X_teste_grade : {Xteg.shape}")

with open(f"{SAIDA}/features_grade.json", "w") as f:
    json.dump({"nomes": nomes + list(tab.columns),
               "n_sst_grade": int(tab.shape[1]),
               "config": dict(n_pc=N_PC, janela_clim=JANELA_CLIM,
                              lat=[LAT_MIN, LAT_MAX], lags=LAGS),
               "citacao": "NOAA ERSST v5, Huang et al. 2017, "
                          "doi:10.7289/V5T72FNM, via NOAA/PSL"}, f, indent=2)

with open(f"{SAIDA}/sst_grade_pca.pkl", "wb") as f:
    pickle.dump({"pca": pca, "escala": esc, "sd_cel": sd_cel, "oceano": oce,
                 "lat": lat, "lon": lon, "peso": w}, f)

print(f"""
  Proximo: FEATURES=grade python -u src/06_submissao.py
  (o 06 precisa da variavel FEATURES apontando para o sufixo)""")