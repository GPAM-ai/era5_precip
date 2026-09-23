#!/usr/bin/env python
"""
Etapa 04 - Tabela de features.

ESPECIFICACAO EXPLICITA DO VETOR DE ATRIBUTOS
=============================================
Cada linha eh um par (mes-alvo, ponto de grade). O alvo eh a ANOMALIA de
precipitacao daquele ponto naquele mes. As colunas sao, em ordem:

  [A] ESTATICAS DO PONTO (4)
      lat, lon          coordenadas geograficas
      clim_alvo         climatologia do ponto no mes-alvo (mm/dia)
      clim_anual        climatologia anual media do ponto (mm/dia)
      Justificativa: informam ao modelo o regime local. Sem elas, o modelo
      nao distingue uma anomalia de +2 mm/dia no Atacama (enorme) de +2 no
      centro da Amazonia (modesta).

  [B] SAZONAIS (2)
      sin_mes, cos_mes  seno e cosseno de 2*pi*mes_alvo/12
      Justificativa: codificacao ciclica, para que dezembro e janeiro
      fiquem proximos. One-hot de 12 niveis quebraria essa continuidade.

  [C] ANOMALIAS LOCAIS (12)
      a_t2, a_cloud_cover, a_shum_850, a_surface_pressure, a_u_850,
      a_v_850, a_temperature_850, a_rel_hum_850, a_geopotential_850,
      a_qu, a_qv, a_conv_umid
      Todas no MES-FEATURE (o mes anterior ao alvo), no proprio ponto.

  [D] MODOS DE GRANDE ESCALA (41 base + defasagens)
      pc_<grupo>_<n>, idx_pacifico, e suas versoes _lag1.._lag3 e _tend
      Valores GLOBAIS: identicos para todos os pontos de um mesmo mes.
      Justificativa: a habilidade preditiva em escala mensal vem da
      configuracao de grande escala, nao do estado do pixel.

O QUE DELIBERADAMENTE NAO ENTRA
  - precipitacao do mes anterior (tp): existe no treino mas NAO no teste,
    onde so ha tp_ultima_obs, constante em dez/2022. Treinar com ela
    produziria dependencia de uma feature ausente na inferencia.
  - valores brutos das covariaveis: so anomalias, para que a topografia
    e o ciclo sazonal nao dominem os splits.

Uso:
    python src/04_features.py [completo|reduzido]
      completo (padrao): todas as defasagens dos PCs
      reduzido         : so lag1 e tendencia, para treino mais rapido
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

SAIDA = "dados_processados"
MODO = sys.argv[1] if len(sys.argv) > 1 else "completo"

DADOS = "dados_brutos"

LOCAIS = ["a_t2", "a_cloud_cover", "a_shum_850", "a_surface_pressure",
          "a_u_850", "a_v_850", "a_temperature_850", "a_rel_hum_850",
          "a_geopotential_850", "a_qu", "a_qv", "a_conv_umid"]


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


def mem():
    try:
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith("VmRSS:"):
                    return int(l.split()[1]) / 1024
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
sep(f"1. CARREGANDO (modo = {MODO})")

tr = xr.open_dataset(f"{SAIDA}/anomalias_treino.nc").load()
te = xr.open_dataset(f"{SAIDA}/anomalias_teste.nc").load()
pcs = pd.read_parquet(f"{SAIDA}/pcs.parquet")
clim = xr.open_dataarray(f"{SAIDA}/climatologia_tp.nc")

print(f"  treino: {dict(tr.sizes)}")
print(f"  teste : {dict(te.sizes)}")
print(f"  pcs   : {pcs.shape}")

# seleciona as colunas de PC conforme o modo
if MODO == "reduzido":
    cols_pc = [c for c in pcs.columns
               if not (c.endswith("_lag2") or c.endswith("_lag3"))]
else:
    cols_pc = list(pcs.columns)
print(f"  PCs usados: {len(cols_pc)} de {pcs.shape[1]}")


# ---------------------------------------------------------------------------
sep("2. SELECAO DE LINHAS VALIDAS (treino)")

# uma linha de treino eh valida se:
#   - o alvo nao eh NaN (o ultimo mes nao tem alvo)
#   - os PCs daquele mes-feature nao tem NaN (os 3 primeiros meses tem)
mes_feat_tr = pd.to_datetime(tr.time.values)
pc_tr = pcs.reindex(mes_feat_tr)

alvo_ok = ~tr["anom_alvo"].isnull().all(dim=("lat", "lon")).values
pc_ok = ~pc_tr[cols_pc].isnull().any(axis=1).values
validos = alvo_ok & pc_ok

print(f"  passos com alvo      : {int(alvo_ok.sum())} de {len(alvo_ok)}")
print(f"  passos com PC completo: {int(pc_ok.sum())}")
print(f"  passos utilizaveis    : {int(validos.sum())}")
print(f"  descartados: {int((~validos).sum())} "
      f"(3 do inicio da serie + 1 do fim sem alvo)")

idx_t = np.where(validos)[0]
n_t = len(idx_t)
n_lat, n_lon = tr.sizes["lat"], tr.sizes["lon"]
n_pt = n_lat * n_lon
n_linhas = n_t * n_pt

print(f"\n  linhas de treino: {n_t} meses x {n_pt} pontos = {n_linhas:,}")


# ---------------------------------------------------------------------------
sep("3. MONTANDO A MATRIZ DE TREINO")

nomes = ["lat", "lon", "clim_alvo", "clim_anual", "sin_mes", "cos_mes"]
nomes += LOCAIS
nomes += cols_pc
n_col = len(nomes)

print(f"  colunas: {n_col}")
print(f"    [A] estaticas          4")
print(f"    [B] sazonais           2")
print(f"    [C] anomalias locais  {len(LOCAIS)}")
print(f"    [D] grande escala     {len(cols_pc)}")
print(f"\n  memoria da matriz: {n_linhas * n_col * 4 / 1e9:.2f} GB")

X = np.empty((n_linhas, n_col), dtype="float32")
y = np.empty(n_linhas, dtype="float32")

lat_g, lon_g = np.meshgrid(tr.lat.values, tr.lon.values, indexing="ij")
lat_f = lat_g.ravel().astype("float32")
lon_f = lon_g.ravel().astype("float32")

# climatologia no ponto, por mes (subamostrada para bater com o treino)
stride = int(round(float(clim.lat[1] - clim.lat[0]) and
                   (len(clim.lat) - 1) / (n_lat - 1)))
clim_sub = clim.isel(lat=slice(None, None, stride),
                     lon=slice(None, None, stride))
assert clim_sub.sizes["lat"] == n_lat, "stride da climatologia nao bate"
clim_arr = clim_sub.transpose("month", "lat", "lon").values.reshape(12, -1)
clim_anual = clim_arr.mean(0).astype("float32")

# mes-ALVO de cada passo de treino
mes_alvo_tr = ((mes_feat_tr.month.values % 12) + 1)

pc_vals = pc_tr[cols_pc].values.astype("float32")
locais_vals = {v: tr[v].values for v in LOCAIS}
alvo_vals = tr["anom_alvo"].values

for j, t in enumerate(idx_t):
    a, b = j * n_pt, (j + 1) * n_pt
    m = mes_alvo_tr[t]

    X[a:b, 0] = lat_f
    X[a:b, 1] = lon_f
    X[a:b, 2] = clim_arr[m - 1]
    X[a:b, 3] = clim_anual
    X[a:b, 4] = np.sin(2 * np.pi * m / 12)
    X[a:b, 5] = np.cos(2 * np.pi * m / 12)

    for k, v in enumerate(LOCAIS):
        X[a:b, 6 + k] = locais_vals[v][t].ravel()

    X[a:b, 6 + len(LOCAIS):] = pc_vals[t]
    y[a:b] = alvo_vals[t].ravel()

    if j % 200 == 0:
        print(f"    {j:4d}/{n_t}  [{mem():.0f} MB]", flush=True)

print(f"\n  X: {X.shape}  y: {y.shape}")
print(f"  NaN em X: {int(np.isnan(X).sum())}   NaN em y: {int(np.isnan(y).sum())}")
assert not np.isnan(X).any(), "ha NaN nas features"
assert not np.isnan(y).any(), "ha NaN no alvo"


# ---------------------------------------------------------------------------
sep("4. METADADOS DO TREINO (para o CV por ano)")

anos = np.repeat(mes_feat_tr.year.values[idx_t], n_pt).astype("int16")
meses_m = np.repeat(mes_alvo_tr[idx_t], n_pt).astype("int8")

meta_tr = pd.DataFrame({"ano_feature": anos, "mes_alvo": meses_m})
print(f"  anos: {meta_tr.ano_feature.min()} a {meta_tr.ano_feature.max()}")
print(f"  linhas por ano (media): {len(meta_tr) / meta_tr.ano_feature.nunique():,.0f}")


# ---------------------------------------------------------------------------
sep("5. MONTANDO A MATRIZ DE TESTE (grade cheia)")

sub = pd.read_csv(f"{DADOS}/sample_submission.csv")
partes = sub["id"].str.split("_", expand=True)
ano_te = partes[0].astype(int).values
mes_te = partes[1].astype(int).values          # mes-ALVO
lat_te = partes[2].astype(float).values
lon_te = partes[3].astype(float).values

lat0, lon0, passo = float(clim.lat[0]), float(clim.lon[0]), 0.25
ilat = np.rint((lat_te - lat0) / passo).astype(int)
ilon = np.rint((lon_te - lon0) / passo).astype(int)
assert np.allclose(clim.lat.values[ilat], lat_te), "indice de lat errado"
assert np.allclose(clim.lon.values[ilon], lon_te), "indice de lon errado"
print("  indices de grade conferem com o sample_submission")

n_te_linhas = len(sub)
n_pt_te = te.sizes["lat"] * te.sizes["lon"]

# posicao de cada linha dentro do passo de tempo do arquivo de teste
data_te = pd.to_datetime([f"{a}-{m:02d}-01" for a, m in zip(ano_te, mes_te)])
tempos_te = pd.to_datetime(te.time.values)
mapa_t = {t: i for i, t in enumerate(tempos_te)}
it = np.array([mapa_t[d] for d in data_te], dtype="int32")
ipt = ilat * te.sizes["lon"] + ilon

print(f"  linhas de teste: {n_te_linhas:,}")
print(f"  memoria: {n_te_linhas * n_col * 4 / 1e9:.2f} GB")

Xte = np.empty((n_te_linhas, n_col), dtype="float32")

clim_cheia = clim.transpose("month", "lat", "lon").values.reshape(12, -1)
Xte[:, 0] = lat_te
Xte[:, 1] = lon_te
Xte[:, 2] = clim_cheia[mes_te - 1, ipt]
Xte[:, 3] = clim_cheia.mean(0)[ipt]
Xte[:, 4] = np.sin(2 * np.pi * mes_te / 12)
Xte[:, 5] = np.cos(2 * np.pi * mes_te / 12)

for k, v in enumerate(LOCAIS):
    campo = te[v].values.reshape(te.sizes["time"], -1)
    Xte[:, 6 + k] = campo[it, ipt]

# PCs: indexados pelo MES-FEATURE, que eh o mes anterior ao alvo
mes_feat_te = data_te - pd.DateOffset(months=1)
pc_te = pcs.reindex(mes_feat_te)[cols_pc].values.astype("float32")
Xte[:, 6 + len(LOCAIS):] = pc_te

print(f"\n  Xte: {Xte.shape}")
print(f"  NaN em Xte: {int(np.isnan(Xte).sum())}")
assert not np.isnan(Xte).any(), "ha NaN nas features de teste"

meta_te = pd.DataFrame({
    "id": sub["id"].values,
    "ano": ano_te,
    "mes_alvo": mes_te,
    "clim": Xte[:, 2],
})


# ---------------------------------------------------------------------------
sep("6. CONFERENCIA CRUZADA TREINO x TESTE")

print("  As distribuicoes devem ser parecidas. Divergencia grande indica")
print("  deslocamento de dominio - o modelo veria no teste algo que nao")
print("  viu no treino.\n")

print(f"  {'coluna':24s} {'treino (m/dp)':>22s} {'teste (m/dp)':>22s}")
for k in [0, 1, 2, 6, 8, 11, 17]:
    if k < n_col:
        a, b = X[:, k], Xte[:, k]
        print(f"  {nomes[k]:24s} {a.mean():10.3f}/{a.std():<10.3f} "
              f"{b.mean():10.3f}/{b.std():<10.3f}")

# checagem especifica dos PCs, onde o deslocamento foi notado em 03
i_slp2 = nomes.index("pc_slp_2") if "pc_slp_2" in nomes else None
if i_slp2:
    a, b = X[:, i_slp2], Xte[:, i_slp2]
    print(f"\n  pc_slp_2  treino {a.mean():+.3f}  teste {b.mean():+.3f}")
    if abs(b.mean() - a.mean()) > 0.5:
        print("  >> deslocamento relevante. O CV deve pesar anos recentes,")
        print("     e o encolhimento (alfa) deve ser conservador.")


# ---------------------------------------------------------------------------
sep("7. GRAVANDO")

np.save(f"{SAIDA}/X_treino.npy", X)
np.save(f"{SAIDA}/y_treino.npy", y)
np.save(f"{SAIDA}/X_teste.npy", Xte)
meta_tr.to_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te.to_parquet(f"{SAIDA}/meta_teste.parquet")

with open(f"{SAIDA}/features.json", "w") as f:
    json.dump({
        "nomes": nomes,
        "modo": MODO,
        "grupos": {
            "estaticas": nomes[:4],
            "sazonais": nomes[4:6],
            "locais": nomes[6:6 + len(LOCAIS)],
            "grande_escala": nomes[6 + len(LOCAIS):],
        },
        "excluidas_deliberadamente": {
            "tp": "presente no treino, ausente no teste (so tp_ultima_obs, "
                  "constante em dez/2022)",
            "valores_brutos": "usadas apenas anomalias",
        },
        "n_linhas_treino": int(n_linhas),
        "n_linhas_teste": int(n_te_linhas),
    }, f, indent=2, ensure_ascii=False)

for arq in ("X_treino.npy", "y_treino.npy", "X_teste.npy"):
    print(f"  {arq:20s} {os.path.getsize(f'{SAIDA}/{arq}')/1e9:6.2f} GB")
print(f"  features.json        {n_col} colunas documentadas")

print("""
  Proximo passo: 05_modelo.py""")