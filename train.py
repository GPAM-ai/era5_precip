#!/usr/bin/env python
"""
train.py - ponto de entrada de treino.

    1. 06_submissao.py      LightGBM: walk-forward, alfa, 5 sementes,
                            gera submissao do LightGBM sozinho
    2. 14_avaliacao.py      previsoes fora da amostra (necessarias para o
                            peso da combinacao)
    3. 17_mos_pontual.py    ridge por ponto, peso validado, combinacao,
                            gera a submissao final

Modelos salvos em dados_processados/: modelo_final_mme3*.pkl (LightGBM),
ridge_pontual.pkl (coeficientes por ponto + peso).

Uso:
    python train.py

Tempo: ~45 min num no de 16 nucleos. Requer prepare_data.py concluido.
"""

import json
import os
import subprocess
import sys

cfg = json.load(open("settings.json"))
os.makedirs(cfg["SUBMISSION_DIR"], exist_ok=True)

env = {**os.environ,
       "FEATURES": cfg["FEATURES"], "LR": cfg["LR"], "N_ARVORES": cfg["N_ARVORES"],
       "N_SEMENTES": cfg["N_SEMENTES"], "RECENCIA": "0"}

tag = f"{cfg['FEATURES']}_norec_t{cfg['N_ARVORES']}l63_s{cfg['N_SEMENTES']}"
sub_lgbm = f"{cfg['SUBMISSION_DIR']}/sub07_{tag}.csv"

ETAPAS = [
    ["python", "src/06_submissao.py"],
    ["python", "src/14_avaliacao.py"],
    ["python", "src/17_mos_pontual.py", sub_lgbm],
]

for cmd in ETAPAS:
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        print(f"\nFALHOU: {' '.join(cmd)}")
        sys.exit(r.returncode)

print(f"\nTreino concluido. Submissao final: {sub_lgbm.replace('.csv', '_mos.csv')}")
