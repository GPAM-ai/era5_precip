#!/usr/bin/env python
"""
Etapa 04b - Remocao de tendencia.

O PROBLEMA DIAGNOSTICADO EM 04:
  a_t2 tem media -0.220 no treino e +0.547 no teste. Quase 0.8 K de
  desvio, cerca de um desvio-padrao inteiro da variavel. Isso eh
  aquecimento global: 2023 e 2024 foram os anos mais quentes da serie, e
  a climatologia - mesmo ponderada no periodo recente - nao acompanha.

POR QUE ISSO QUEBRA UM MODELO DE ARVORE:
  Arvores particionam em degraus e NAO extrapolam. Se no teste a_t2 cai
  sempre no bin mais alto que o modelo viu no treino, toda previsao
  herda o que ele aprendeu para os meses mais quentes do historico. O
  erro deixa de ser aleatorio e vira vies sistematico.

A CORRECAO:
  Para cada feature, ajusta-se uma reta (valor medio por passo de tempo
  contra o ano) usando SO o treino, e subtrai-se essa reta de treino e
  teste. A tendencia extrapolada para 2023/2024 vem da reta ajustada.
  Preserva a variabilidade interanual, que eh o sinal util, e remove a
  deriva de longo prazo, que o modelo nao sabe tratar.

  O ALVO nao eh destendenciado: a anomalia de precipitacao ja tem media
  praticamente zero (-0.014) e nao apresenta deriva relevante.

VALIDACAO:
  Nao basta argumentar - a decisao eh testada. Treina-se em 1940-2012 e
  avalia-se em 2013-2022, com e sem remocao de tendencia. Esse desenho
  imita a extrapolacao real (ajustar ate um ano e prever anos seguintes).

Uso:
    python src/04b_detrend.py
"""

import json
import os

import numpy as np
import pandas as pd

SAIDA = "dados_processados"


def sep(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68, flush=True)


# ---------------------------------------------------------------------------
sep("1. CARREGANDO")

X = np.load(f"{SAIDA}/X_treino.npy", mmap_mode="r")
Xte = np.load(f"{SAIDA}/X_teste.npy", mmap_mode="r")
meta_tr = pd.read_parquet(f"{SAIDA}/meta_treino.parquet")
meta_te = pd.read_parquet(f"{SAIDA}/meta_teste.parquet")

with open(f"{SAIDA}/features.json") as f:
    spec = json.load(f)
nomes = spec["nomes"]

print(f"  treino: {X.shape}")
print(f"  teste : {Xte.shape}")

# colunas que NAO devem ser destendenciadas: estaticas e sazonais
FIXAS = {"lat", "lon", "clim_alvo", "clim_anual", "sin_mes", "cos_mes"}
alvo_cols = [i for i, n in enumerate(nomes) if n not in FIXAS]
print(f"  colunas a tratar: {len(alvo_cols)} de {len(nomes)}")


# ---------------------------------------------------------------------------
sep("2. DIAGNOSTICO: QUAIS FEATURES TEM DERIVA?")

anos_tr = meta_tr["ano_feature"].values
anos_te = meta_te["ano"].values

# media por ano, no treino (uma linha por ano, nao por ponto)
anos_unicos = np.unique(anos_tr)
print(f"  anos no treino: {anos_unicos[0]} a {anos_unicos[-1]} "
      f"({len(anos_unicos)} anos)\n")

# agrega por ano de forma barata: soma acumulada por grupo
ordem = np.argsort(anos_tr, kind="stable")
anos_ord = anos_tr[ordem]
bordas = np.searchsorted(anos_ord, anos_unicos, side="left")
bordas = np.append(bordas, len(anos_ord))

medias_ano = np.empty((len(anos_unicos), len(nomes)), dtype="float64")
for k in range(len(anos_unicos)):
    fatia = ordem[bordas[k]:bordas[k + 1]]
    medias_ano[k] = X[fatia].mean(axis=0)

# ajusta reta por coluna
coef = np.zeros(len(nomes))
intercepto = np.zeros(len(nomes))
x_ = anos_unicos.astype("float64")
x_c = x_ - x_.mean()
den = (x_c ** 2).sum()

for j in alvo_cols:
    y_ = medias_ano[:, j]
    coef[j] = (x_c * (y_ - y_.mean())).sum() / den
    intercepto[j] = y_.mean() - coef[j] * x_.mean()

# ranking por deriva acumulada ao longo de 83 anos
deriva = np.abs(coef) * (x_[-1] - x_[0])
desvios = np.array([X[::997, j].std() for j in range(len(nomes))])
razao = deriva / (desvios + 1e-9)

ranking = sorted(((razao[j], deriva[j], nomes[j]) for j in alvo_cols),
                 reverse=True)

print(f"  {'feature':24s} {'deriva 83 anos':>16s} {'em desvios':>12s}")
for r, d, n in ranking[:10]:
    marca = "  <-- forte" if r > 1.0 else ("  <-- moderada" if r > 0.5 else "")
    print(f"  {n:24s} {d:16.4f} {r:12.2f}{marca}")

fortes = [n for r, d, n in ranking if r > 0.5]
print(f"\n  features com deriva acima de 0.5 desvio: {len(fortes)}")


# ---------------------------------------------------------------------------
sep("3. APLICANDO A REMOCAO DE TENDENCIA")

print("  A reta eh ajustada no treino e extrapolada para 2023/2024.")
print("  Isso significa assumir que a tendencia continua - hipotese")
print("  razoavel para 2 anos, e melhor que ignorar a deriva.\n")

Xd = np.array(X, dtype="float32")        # copia em memoria
Xted = np.array(Xte, dtype="float32")

for j in alvo_cols:
    if coef[j] == 0:
        continue
    Xd[:, j] -= (intercepto[j] + coef[j] * anos_tr).astype("float32")
    Xted[:, j] -= (intercepto[j] + coef[j] * anos_te).astype("float32")

print("  conferencia das medias apos a correcao:\n")
print(f"  {'feature':24s} {'treino antes':>14s} {'teste antes':>13s}"
      f" {'treino apos':>13s} {'teste apos':>12s}")
for n in ["a_t2", "a_temperature_850", "a_geopotential_850",
          "pc_slp_2", "a_shum_850"]:
    if n not in nomes:
        continue
    j = nomes.index(n)
    print(f"  {n:24s} {X[::97, j].mean():14.3f} {Xte[::97, j].mean():13.3f}"
          f" {Xd[::97, j].mean():13.3f} {Xted[::97, j].mean():12.3f}")

print("""
  O que se espera: as colunas "apos" ficam proximas entre treino e teste.
  Se a diferenca persistir, a deriva nao eh linear e vale investigar.""")


# ---------------------------------------------------------------------------
sep("4. VALIDACAO: A CORRECAO AJUDA DE FATO?")

print("""  Teste de extrapolacao, imitando o problema real: treina em
  1940-2012, preve 2013-2022. Um modelo pequeno e rapido, so para
  comparar as duas versoes das features.
""")

try:
    import lightgbm as lgb
except ImportError:
    print("  lightgbm nao disponivel; pulando a validacao.")
    lgb = None

if lgb is not None:
    y = np.load(f"{SAIDA}/y_treino.npy")

    m_fit = anos_tr <= 2012
    m_val = anos_tr >= 2013
    print(f"  ajuste   : {int(m_fit.sum()):,} linhas (1940-2012)")
    print(f"  validacao: {int(m_val.sum()):,} linhas (2013-2022)\n")

    # subamostra as linhas de ajuste para o teste ser rapido
    rng = np.random.default_rng(42)
    idx_fit = np.where(m_fit)[0]
    idx_fit = rng.choice(idx_fit, size=min(2_000_000, len(idx_fit)),
                         replace=False)
    idx_val = np.where(m_val)[0]

    params = dict(objective="l2", num_leaves=63, learning_rate=0.05,
                  min_child_samples=200, feature_fraction=0.7,
                  bagging_fraction=0.7, bagging_freq=1, lambda_l2=5,
                  n_estimators=300, n_jobs=-1, verbose=-1, random_state=42)

    resultados = {}
    for rotulo, XX in (("sem correcao", X), ("com correcao", Xd)):
        mod = lgb.LGBMRegressor(**params)
        mod.fit(np.asarray(XX[idx_fit]), y[idx_fit])
        pred = mod.predict(np.asarray(XX[idx_val]))

        rmse_mod = float(np.sqrt(np.mean((y[idx_val] - pred) ** 2)))
        rmse_clim = float(np.sqrt(np.mean(y[idx_val] ** 2)))
        vies = float(np.mean(pred - y[idx_val]))

        # alfa otimo: minimiza RMSE de alfa*pred contra y
        a_ot = float(np.dot(pred, y[idx_val]) / (np.dot(pred, pred) + 1e-12))
        rmse_a = float(np.sqrt(np.mean((y[idx_val] - a_ot * pred) ** 2)))

        resultados[rotulo] = (rmse_mod, rmse_a, a_ot, vies)
        print(f"  {rotulo}:")
        print(f"    RMSE da climatologia : {rmse_clim:.4f}")
        print(f"    RMSE do modelo puro  : {rmse_mod:.4f}")
        print(f"    alfa otimo           : {a_ot:.3f}")
        print(f"    RMSE com encolhimento: {rmse_a:.4f}  "
              f"(ganho {100*(1-rmse_a/rmse_clim):.2f}%)")
        print(f"    vies medio           : {vies:+.4f}\n")

    a = resultados["sem correcao"][1]
    b = resultados["com correcao"][1]
    if b < a:
        print(f"  >> A REMOCAO DE TENDENCIA AJUDA: {a:.4f} -> {b:.4f} "
              f"({100*(a-b)/a:.2f}% melhor). Usar Xd.")
    else:
        print(f"  >> a correcao nao ajudou ({a:.4f} -> {b:.4f}). "
              "Manter as features originais e registrar o teste no README.")


# ---------------------------------------------------------------------------
sep("5. GRAVANDO")

np.save(f"{SAIDA}/X_treino_dt.npy", Xd)
np.save(f"{SAIDA}/X_teste_dt.npy", Xted)
np.savez(f"{SAIDA}/tendencias.npz", coef=coef, intercepto=intercepto,
         nomes=np.array(nomes))

print(f"  X_treino_dt.npy  {os.path.getsize(f'{SAIDA}/X_treino_dt.npy')/1e9:.2f} GB")
print(f"  X_teste_dt.npy   {os.path.getsize(f'{SAIDA}/X_teste_dt.npy')/1e9:.2f} GB")
print(f"  tendencias.npz   coeficientes de {len(alvo_cols)} colunas")

print("""
  Proximo passo: 05_modelo.py""")