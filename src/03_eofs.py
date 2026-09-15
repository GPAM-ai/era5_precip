#!/usr/bin/env python
"""
Etapa 03 - EOFs: modos de grande escala, para treino e teste.

POR QUE:
  A habilidade preditiva em escala mensal nao esta no ponto de grade - esta
  na configuracao de grande escala (ENOS, dipolo do Atlantico, posicao da
  ZCAS). A PCA comprime 1.184 celulas em ~8 numeros por mes que carregam o
  grosso desse sinal. Sao features GLOBAIS: um valor por mes, o mesmo para
  todos os pontos de grade, complementando as anomalias locais.

A SERIE EH CONTINUA:
  O ultimo passo do treino (features de dez/2022) e o primeiro do teste
  (rotulo 2023_01, features de dez/2022) sao o MESMO mes de features -
  verificado em 00b. Entao treino e teste se encaixam numa unica serie
  temporal de meses-feature, de jan/1940 a nov/2024, com um passo de
  sobreposicao. Isso resolve as defasagens do inicio de 2023 sem gambiarra:
  o lag de jan/2023 puxa dez/2022, que existe.

SEM VAZAMENTO:
  A PCA e as estatisticas de normalizacao sao ajustadas SO no treino
  (1940-2022) e depois aplicadas ao teste com transform. Nenhum valor de
  2023/2024 influencia os eixos.

Uso:
    python src/03_eofs.py [n_componentes]
"""

import os
import pickle
import sys

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

SAIDA = "dados_processados"
N_PC = int(sys.argv[1]) if len(sys.argv) > 1 else 8
LAGS = (1, 2, 3)
COARSEN = 8

GRUPOS = {
    "slp": ["a_surface_pressure"],
    "z850": ["a_geopotential_850"],
    "circ": ["a_u_850", "a_v_850"],
    "umid": ["a_shum_850", "a_rel_hum_850"],
    "fluxo": ["a_qu", "a_qv", "a_conv_umid"],
}

EL_NINO = [1958, 1966, 1973, 1983, 1987, 1992, 1998, 2003, 2010, 2016]
LA_NINA = [1950, 1956, 1974, 1976, 1989, 1999, 2000, 2008, 2011, 2021]


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO CAMPOS")

tr = xr.open_dataset(f"{SAIDA}/campos_eof.nc").load()
print(f"  treino (2 graus): {dict(tr.sizes)}")

# o teste esta em resolucao cheia; agrega com a mesma regra
te_cheio = xr.open_dataset(f"{SAIDA}/anomalias_teste.nc").load()
te = te_cheio.coarsen(lat=COARSEN, lon=COARSEN, boundary="trim").mean()
print(f"  teste  (2 graus): {dict(te.sizes)}")

assert tr.sizes["lat"] == te.sizes["lat"], "grades agregadas incompativeis"
assert tr.sizes["lon"] == te.sizes["lon"], "grades agregadas incompativeis"

n_tr, n_te = tr.sizes["time"], te.sizes["time"]
peso = np.sqrt(np.cos(np.deg2rad(tr.lat))).astype("float32")
print(f"\n  peso de area: {float(peso.min()):.3f} a {float(peso.max()):.3f}")


# ---------------------------------------------------------------------------
sep("2. MONTANDO A SERIE CONTINUA DE MESES-FEATURE")

# treino: a coordenada time JA eh o mes das features
meses_feat_tr = pd.to_datetime(tr.time.values)

# teste: a coordenada eh o mes-ALVO; o mes-feature eh o anterior
meses_feat_te = pd.to_datetime(te.time.values) - pd.DateOffset(months=1)

print(f"  treino: {meses_feat_tr[0]:%Y-%m} a {meses_feat_tr[-1]:%Y-%m} "
      f"({n_tr} passos)")
print(f"  teste : {meses_feat_te[0]:%Y-%m} a {meses_feat_te[-1]:%Y-%m} "
      f"({n_te} passos)")

sobrepostos = meses_feat_te.isin(meses_feat_tr)
print(f"\n  passos do teste que ja existem no treino: {int(sobrepostos.sum())}")
print(f"  (esperado 1: dez/2022 aparece nos dois)")

assert int(sobrepostos.sum()) == 1, "sobreposicao inesperada entre treino e teste"


# ---------------------------------------------------------------------------
sep(f"3. DECOMPOSICAO ({N_PC} componentes por grupo)")

print("  PCA ajustada SO no treino; o teste recebe transform.\n")

series = {}
modelos = {}

for nome, variaveis in GRUPOS.items():
    blocos_tr, blocos_te, normas = [], [], []

    for v in variaveis:
        X_tr = (tr[v] * peso).values.reshape(n_tr, -1).astype("float32")
        X_te = (te[v] * peso).values.reshape(n_te, -1).astype("float32")

        mu = X_tr.mean(0)
        sd = X_tr.std(0) + 1e-9
        normas.append((mu, sd))

        blocos_tr.append((X_tr - mu) / sd)
        blocos_te.append((X_te - mu) / sd)      # estatisticas do TREINO

    A_tr = np.nan_to_num(np.concatenate(blocos_tr, axis=1))
    A_te = np.nan_to_num(np.concatenate(blocos_te, axis=1))

    pca = PCA(n_components=N_PC, random_state=42)
    Y_tr = pca.fit_transform(A_tr)
    Y_te = pca.transform(A_te)                  # mesmos eixos

    escala = Y_tr.std(0) + 1e-9                 # escala do TREINO
    Y_tr, Y_te = Y_tr / escala, Y_te / escala

    for i in range(N_PC):
        series[f"pc_{nome}_{i+1}"] = (Y_tr[:, i].astype("float32"),
                                      Y_te[:, i].astype("float32"))

    modelos[nome] = {"pca": pca, "normas": normas, "escala": escala,
                     "variaveis": variaveis}

    ev = pca.explained_variance_ratio_
    print(f"  {nome:7s} ({A_tr.shape[1]:5d} colunas)  "
          f"var: {100*ev[0]:5.1f}% {100*ev[1]:5.1f}% {100*ev[2]:5.1f}% "
          f"... acum {100*ev.sum():5.1f}%")

# indice de pressao do Pacifico: alternativa interpretavel aos EOFs
reg = dict(lat=slice(-20, 0), lon=slice(-90, -80))
p_tr = tr["a_surface_pressure"].sel(**reg).mean(dim=("lat", "lon")).values
p_te = te["a_surface_pressure"].sel(**reg).mean(dim=("lat", "lon")).values
mu_p, sd_p = p_tr.mean(), p_tr.std()
series["idx_pacifico"] = (((p_tr - mu_p) / sd_p).astype("float32"),
                          ((p_te - mu_p) / sd_p).astype("float32"))
modelos["idx_pacifico"] = {"mu": float(mu_p), "sd": float(sd_p), "regiao": reg}

print(f"\n  total de series: {len(series)}")


# ---------------------------------------------------------------------------
sep("4. VALIDACAO: OS MODOS RESPONDEM AO ENOS?")

print("""  Sem TSM, o ENOS so pode entrar via pressao e geopotencial. Se um
  modo for de fato ENOS, sua media em DJF de El Nino deve diferir da
  media em DJF de La Nina. Acima de ~0.8 desvios indica separacao forte.
""")

anos = meses_feat_tr.year.values
meses = meses_feat_tr.month.values
ano_evento = np.where(meses >= 11, anos + 1, anos)
djf = np.isin(meses, [11, 12, 1, 2])

m_nino = djf & np.isin(ano_evento, EL_NINO)
m_nina = djf & np.isin(ano_evento, LA_NINA)
print(f"  meses DJF: {int(m_nino.sum())} em El Nino, "
      f"{int(m_nina.sum())} em La Nina\n")

ranking = sorted(
    ((abs(s[0][m_nino].mean() - s[0][m_nina].mean()),
      s[0][m_nino].mean() - s[0][m_nina].mean(), k)
     for k, s in series.items()),
    reverse=True)

print(f"  {'modo':22s} {'separacao':>12s}")
for mag, d, k in ranking[:8]:
    marca = "  <-- forte" if mag > 0.8 else ("  <-- moderada" if mag > 0.4 else "")
    print(f"  {k:22s} {d:+12.3f}{marca}")

print(f"""
  O indice do Pacifico, construido a mao, serve de contraprova: se a PCA
  achou o ENOS, os dois devem concordar em sinal e magnitude.""")


# ---------------------------------------------------------------------------
sep("5. ONDE 2023 E 2024 CAEM NESSE ESPACO")

print("""  2023 foi El Nino forte; 2024 comecou com ele enfraquecendo e virou
  para La Nina. Se os PCs do teste capturam isso, o modo de ENOS deve
  mudar de sinal ao longo dos 24 meses. Eh a checagem de que o transform
  no teste esta correto.
""")

k_enos = ranking[0][2]
serie_te = series[k_enos][1]
print(f"  modo {k_enos} nos meses-feature do teste:")
for i in range(0, n_te, 3):
    print(f"    {meses_feat_te[i]:%Y-%m}: {serie_te[i]:+.2f}")

sinal_ini = np.mean(serie_te[:6])
sinal_fim = np.mean(serie_te[-6:])
print(f"\n  media dos 6 primeiros meses: {sinal_ini:+.2f}")
print(f"  media dos 6 ultimos meses  : {sinal_fim:+.2f}")
if np.sign(sinal_ini) != np.sign(sinal_fim):
    print("  >> houve inversao de sinal, coerente com a transicao ENOS.")
else:
    print("  >> sem inversao clara. Nao invalida nada, mas vale notar.")


# ---------------------------------------------------------------------------
sep("6. TABELA UNIFICADA COM DEFASAGENS")

# empilha treino e teste por mes-feature, descartando a sobreposicao
idx = meses_feat_tr.append(meses_feat_te[~sobrepostos])
dados = {k: np.concatenate([v[0], v[1][~sobrepostos]])
         for k, v in series.items()}

tabela = pd.DataFrame(dados, index=idx).sort_index()
base = list(tabela.columns)

# monta todas as colunas de uma vez, evitando fragmentacao do DataFrame
extras = {}
for lag in LAGS:
    for col in base:
        extras[f"{col}_lag{lag}"] = tabela[col].shift(lag)
for col in base:
    extras[f"{col}_tend"] = tabela[col] - tabela[col].shift(3)

tabela = pd.concat([tabela, pd.DataFrame(extras, index=tabela.index)], axis=1)
tabela.index.name = "mes_feature"

print(f"  serie unificada: {tabela.shape[0]} meses, {tabela.shape[1]} colunas")
print(f"  de {tabela.index[0]:%Y-%m} a {tabela.index[-1]:%Y-%m}")
print(f"  linhas com NaN (inicio da serie): "
      f"{int(tabela.isnull().any(axis=1).sum())} (esperado {max(LAGS)})")

# conferencia: as defasagens do inicio do teste puxam valores reais?
jan23 = pd.Timestamp("2023-01-01")
if jan23 in tabela.index:
    linha = tabela.loc[jan23]
    faltando = int(linha.isnull().sum())
    print(f"\n  jan/2023 (features): {faltando} NaN de {len(linha)} colunas")
    print("  (esperado 0 - as defasagens puxam dez/2022 e antes, do treino)")


# ---------------------------------------------------------------------------
sep("7. GRAVANDO")

tabela.to_parquet(f"{SAIDA}/pcs.parquet")
print(f"  pcs.parquet  {tabela.shape}")

with open(f"{SAIDA}/pca_modelos.pkl", "wb") as f:
    pickle.dump({"modelos": modelos, "grupos": GRUPOS, "n_pc": N_PC,
                 "lags": LAGS, "coarsen": COARSEN}, f)
print(f"  pca_modelos.pkl  ({len(modelos)} objetos ajustados)")

print("""
  NOTA sobre vazamento, para o README:
  A PCA e as normalizacoes usam apenas 1940-2022. No CV por ano, a rigor
  a PCA deveria ser reajustada dentro de cada fold. Como ela eh nao
  supervisionada (nao olha o alvo) e usa 83 anos, o efeito eh
  desprezivel - mas a escolha fica registrada.

  Proximo passo: 04_features.py""")