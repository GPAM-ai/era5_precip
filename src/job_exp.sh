#!/bin/bash
#SBATCH -A cptec
#SBATCH -p lncc-cpu_amd
#SBATCH --job-name=exp_eofs
#SBATCH --output=logs/exp_%j.out
#SBATCH --time=02:00:00
#SBATCH -N 1 -n 1 -c 16

cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
python -u src/07_experimento_eofs.py
