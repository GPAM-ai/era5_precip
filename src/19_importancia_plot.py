#!/usr/bin/env python
"""
Etapa 19 - Grafico de importancia das 20 features mais importantes.

Exigido pelo Kaggle Winning Model Documentation Guidelines (A3).
Le o modelo salvo por 06_submissao.py e gera figuras/importancia_top20.png
e figuras/importancia_grupos.png.

Uso:
    python src/19_importancia_plot.py
"""

import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SAIDA = "dados_processados"
FEATURES = os.environ.get("FEATURES", "mme3")
os.makedirs("figuras", exist_ok=True)

lg = pickle.load(open(f"{SAIDA}/modelo_final_{FEATURES}.pkl", "rb"))
modelos = lg.get("modelos", [lg["modelo"]])
with open(f"{SAIDA}/features_{FEATURES}.json") as f:
    nomes = json.load(f)["nomes"]

# media da importancia (gain) entre as sementes, normalizada
imp = np.mean([m.booster_.feature_importance(importance_type="gain") for m in modelos], axis=0)
imp = 100 * imp / imp.sum()
ordem = np.argsort(imp)[::-1]

# --- top 20 -----------------------------------------------------------------
top = ordem[:20]
rotulos = {
    "clim_alvo": "climatologia do ponto (mes-alvo)", "clim_anual": "climatologia anual",
    "lat": "latitude", "lon": "longitude",
}
def nome_legivel(n):
    if n in rotulos:
        return rotulos[n]
    n = n.replace("_L05_anom", " (lead 0,5)").replace("_L15_anom", " (lead 1,5)")
    n = n.replace("mme05_anom", "media dos sistemas (lead 0,5)").replace("mme_anom", "media geral")
    n = n.replace("seas5", "SEAS5").replace("cansips", "CanSIPS").replace("cfsv2", "CFSv2")
    n = n.replace("ukmo", "UKMO").replace("cmcc", "CMCC").replace("spear", "GFDL-SPEAR")
    n = n.replace("nasa", "NASA").replace("meteo_france", "Meteo-France").replace("dwd", "DWD")
    return n

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(range(20)[::-1], imp[top], color="#2a6f97")
ax.set_yticks(range(20)[::-1])
ax.set_yticklabels([nome_legivel(nomes[i]) for i in top], fontsize=9)
ax.set_xlabel("% do ganho total (LightGBM, media de 5 sementes)")
ax.set_title("20 features mais importantes")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/importancia_top20.png", dpi=150)
print("  figuras/importancia_top20.png")

# --- por grupo --------------------------------------------------------------
def grupo(n):
    p = n.split("_")[0]
    if n in ("lat", "lon", "clim_alvo", "clim_anual"): return "estaticas"
    if n in ("sin_mes", "cos_mes"): return "sazonais"
    if n.startswith("a_"): return "locais (reanalise)"
    if n.startswith("pc_") or n.startswith("idx_"): return "grande escala (reanalise)"
    if p in ("nino12", "nino34", "nino4", "oni", "soi", "pdo", "dipolo", "nino"): return "indices de TSM"
    if n.startswith("nmme") or n.startswith("mme") or p in (
        "cfsv2", "spear", "nasa", "cansips", "seas5", "ukmo", "meteo", "dwd", "cmcc"):
        return "previsoes dinamicas"
    return "outros"

g = {}
for i, n in enumerate(nomes):
    g[grupo(n)] = g.get(grupo(n), 0) + imp[i]
g = dict(sorted(g.items(), key=lambda kv: -kv[1]))

fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(list(g.keys())[::-1], list(g.values())[::-1], color="#2a6f97")
ax.set_xlabel("% do ganho total")
ax.set_title("Importancia por grupo de features")
for i, (k, v) in enumerate(reversed(list(g.items()))):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("figuras/importancia_grupos.png", dpi=150)
print("  figuras/importancia_grupos.png")
for k, v in g.items():
    print(f"    {k:28s} {v:5.1f}%")
