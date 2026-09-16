#!/usr/bin/env python
"""
Etapa 10 - Diagnostico: o teste esta fora da distribuicao de treino?

O SINTOMA:
  Tres submissoes, CV subindo, placar publico descendo:
    sub04  CV +1.88%  placar 1.76781
    sub05  CV +2.07%  placar 1.77689   (+ peso por recencia)
    sub06  CV +2.35%  placar 1.79552   (+ indices de TSM)
  Divergencia sistematica, nao ruido.

A HIPOTESE:
  2023 e 2024 sao anos de temperatura recorde - Atlantico Norte tropical
  acima de qualquer valor do historico. TNA eh a feature mais importante
  do modelo com TSM. Arvores nao extrapolam: para um valor nunca visto,
  devolvem o comportamento do bin mais alto conhecido, que nao
  necessariamente descreve 2023. Peso por recencia agrava, porque
  concentra o treino nos anos quentes e deixa o modelo mais dependente
  de temperatura exatamente onde ela sai da escala.

O QUE SE MEDE:
  Para cada feature, a fracao de linhas do teste que caem FORA do
  intervalo [p1, p99] do treino. Features com fracao alta sao as que
  forcam extrapolacao. Se forem as mesmas que dominam a importancia,
  a hipotese se confirma.

Uso:
    python src/10_diagnostico_ood.py
"""

import json
import pickle

import numpy as np
import pandas as pd

SAIDA = "dados_processados"


def sep(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70, flush=True)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

with open(f"{SAIDA}/features_sst.json") as f:
    nomes = json.load(f)["nomes"]

X = np.load(f"{SAIDA}/X_treino_sst.npy", mmap_mode="r")
Xte = np.load(f"{SAIDA}/X_teste_sst.npy")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

# subamostra do treino com passo fixo: leitura em blocos, nao aleatoria
passo = 37
Xs = np.asarray(X[::passo])
print(f"  treino (1/{passo}): {Xs.shape}")
print(f"  teste: {Xte.shape}")

try:
    with open(f"{SAIDA}/modelo_final_sst.pkl", "rb") as f:
        m = pickle.load(f)["modelo"]
    imp = pd.Series(m.booster_.feature_importance("gain"), index=nomes)
    imp = 100 * imp / imp.sum()
except Exception:
    imp = pd.Series(np.nan, index=nomes)


# ---------------------------------------------------------------------------
sep("2. FRACAO DO TESTE FORA DE [p1, p99] DO TREINO")

p1 = np.percentile(Xs, 1, axis=0)
p99 = np.percentile(Xs, 99, axis=0)
p0 = Xs.min(axis=0)
p100 = Xs.max(axis=0)
sd = Xs.std(axis=0) + 1e-9

fora_p = ((Xte < p1) | (Xte > p99)).mean(axis=0) * 100
fora_abs = ((Xte < p0) | (Xte > p100)).mean(axis=0) * 100
excesso = np.maximum(Xte.max(axis=0) - p100, p0 - Xte.min(axis=0)) / sd

df = pd.DataFrame({
    "feature": nomes,
    "fora_p1p99_%": fora_p,
    "fora_minmax_%": fora_abs,
    "excesso_sd": excesso,
    "importancia_%": imp.values,
}).sort_values("fora_p1p99_%", ascending=False)

print("  Em distribuicao estavel, ~2% das linhas caem fora de [p1, p99].")
print("  Muito acima disso = deslocamento. 'excesso' = quantos desvios o")
print("  teste ultrapassa o EXTREMO do treino (0 = dentro do visto).\n")
print(f"  {'feature':24s} {'fora p1-p99':>12s} {'fora min-max':>13s} "
      f"{'excesso(sd)':>12s} {'import.':>9s}")
for _, r in df.head(25).iterrows():
    marca = "  <--" if r["fora_minmax_%"] > 5 else ""
    print(f"  {r.feature:24s} {r['fora_p1p99_%']:11.1f}% "
          f"{r['fora_minmax_%']:12.1f}% {r.excesso_sd:12.2f} "
          f"{r['importancia_%']:8.2f}{marca}")


# ---------------------------------------------------------------------------
sep("3. AS FEATURES IMPORTANTES ESTAO DESLOCADAS?")

top = df.sort_values("importancia_%", ascending=False).head(15)
print(f"  {'top 15 por importancia':26s} {'import.':>8s} {'fora p1-p99':>12s} "
      f"{'excesso(sd)':>12s}")
for _, r in top.iterrows():
    marca = "  <-- PROBLEMA" if r["fora_p1p99_%"] > 15 else ""
    print(f"  {r.feature:26s} {r['importancia_%']:7.2f}% "
          f"{r['fora_p1p99_%']:11.1f}% {r.excesso_sd:12.2f}{marca}")

peso_desloc = df[df["fora_p1p99_%"] > 15]["importancia_%"].sum()
print(f"\n  importancia concentrada em features com >15% fora: {peso_desloc:.1f}%")


# ---------------------------------------------------------------------------
sep("4. 2023 vs 2024")

print("  Se 2024 estiver tao deslocado quanto 2023, o que aconteceu no")
print("  placar publico se repete no privado.\n")

chave = ["tna", "tsa", "dipolo_atl", "nino34", "nino12", "a_t2",
         "a_temperature_850", "a_geopotential_850", "pc_slp_2"]
print(f"  {'feature':20s} {'2023 fora':>10s} {'2024 fora':>10s} "
      f"{'media 2023':>11s} {'media 2024':>11s} {'treino p99':>11s}")
for nome in chave:
    if nome not in nomes:
        continue
    i = nomes.index(nome)
    linha = f"  {nome:20s}"
    for ano in (2023, 2024):
        m = meta_te["ano"].values == ano
        fora = ((Xte[m, i] < p1[i]) | (Xte[m, i] > p99[i])).mean() * 100
        linha += f" {fora:9.1f}%"
    for ano in (2023, 2024):
        m = meta_te["ano"].values == ano
        linha += f" {Xte[m, i].mean():+11.3f}"
    linha += f" {p99[i]:+11.3f}"
    print(linha)


# ---------------------------------------------------------------------------
sep("5. LEITURA")

print("""  Se temperatura (tna, a_t2, temperature_850) e os indices dominarem
  a lista de deslocamento E estiverem entre as mais importantes, a
  hipotese se confirma: o modelo extrapola em 2023/24 porque nunca viu
  temperaturas assim.

  Caminhos, do mais conservador ao mais agressivo:
    a) final = sub04 (sem recencia, sem SST): a menos dependente de
       temperatura e a melhor no placar
    b) clip das features do teste ao intervalo do treino: impede a
       arvore de cair em bins vazios, mas descarta informacao
    c) remover tendencia dos indices de TSM: transforma "recorde
       absoluto" em "anomalia relativa ao aquecimento", que o modelo
       ja viu
    d) manter so o dipolo (TNA - TSA), que eh menos afetado por
       aquecimento uniforme que os valores absolutos

  Nenhuma pode ser validada no CV atual: ele nao contem anos com
  temperatura recorde. O placar publico, com todos os defeitos, eh a
  unica medida disponivel desse regime.""")

df.to_csv(f"{SAIDA}/diagnostico_ood.csv", index=False)
print(f"\n  salvo: {SAIDA}/diagnostico_ood.csv")