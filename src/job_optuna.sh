#!/bin/bash
#SBATCH -A cptec
#SBATCH -p lncc-cpu_amd
#SBATCH --job-name=optuna
#SBATCH --output=logs/optuna_%j.out
#SBATCH --time=05:00:00
#SBATCH -N 1 -n 1 -c 16

# Busca de hiperparametros em batch. Roda sozinho; colher o resultado em
# dados_processados/optuna_melhor_<features>.json quando terminar.
#
#   sbatch src/job_optuna.sh
#   tail -f logs/optuna_<id>.out

cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# optuna nao esta no ambiente base; instala no login antes de submeter:
#   pip install optuna
python -c "import optuna" || { echo "pip install optuna primeiro"; exit 1; }

export FEATURES=${FEATURES:-grade}
python -u src/12_optuna.py 40