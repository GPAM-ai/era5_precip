#!/usr/bin/env python
"""
prepare_data.py - ponto de entrada de preparacao de dados.

Executa a cadeia que transforma os dados brutos da competicao e os dados
externos ja baixados em matrizes de treino e teste:

    01_climatologia.py, 01d_climatologia_final.py   climatologia v3
    02_anomalias.py 3                               anomalias e campos para EOF
    03_eofs.py 8                                    EOFs
    04_features.py completo                         tabela de atributos
    09_indices_noaa.py (ROBUSTO=1)                  indices de TSM
    13d_mme_ext.py                                  sistemas dinamicos

PRE-REQUISITOS (fora do cluster, ver README secao 8):
    dados_processados/indices_brutos/ com os arquivos da NOAA, do IRI e do
    Copernicus. Os scripts baixar_seas5.py e baixar_c3s.py geram os do
    Copernicus; os comandos curl para NOAA e IRI estao documentados em
    09_indices_noaa.py, 13_nmme.py, 13b_mme.py, 13c_mme_leads.py.

Uso:
    python prepare_data.py

Tempo: ~25 min num no de 16 nucleos. Saida: dados_processados/X_treino_mme3.npy,
X_teste_mme3.npy, y_treino.npy, meta_*.parquet, features_mme3.json.
"""

import json
import os
import subprocess
import sys

cfg = json.load(open("settings.json"))
os.makedirs(cfg["PROCESSED_DATA_DIR"], exist_ok=True)
os.makedirs(cfg["LOGS_DIR"], exist_ok=True)

ETAPAS = [
    (["python", "src/01_climatologia.py"], {}),
    (["python", "src/01d_climatologia_final.py"], {}),
    (["python", "src/02_anomalias.py", "3"], {}),
    (["python", "src/03_eofs.py", "8"], {}),
    (["python", "src/04_features.py", "completo"], {}),
    (["python", "src/09_indices_noaa.py"], {"ROBUSTO": "1"}),
    (["python", "src/13d_mme_ext.py"], {}),
]

for cmd, env_extra in ETAPAS:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    env = {**os.environ, **env_extra}
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"\nFALHOU: {' '.join(cmd)}")
        sys.exit(r.returncode)

print("\nPreparacao concluida. Proximo: python train.py")
