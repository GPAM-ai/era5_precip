#!/bin/bash
#SBATCH -A cptec
#SBATCH -p lncc-cpu_amd
#SBATCH --exclude=sdumont2nd1002,sdumont2nd1003,sdumont2nd1004
#SBATCH --job-name=final
#SBATCH --output=logs/final_%j.out
#SBATCH --time=02:00:00
#SBATCH -N 1 -n 1 -c 16
cd "$SLURM_SUBMIT_DIR"
export PATH=/scratch/app/anaconda3/2024.10/bin:$PATH
LR=0.02 N_ARVORES=1200 FEATURES=mme3 RECENCIA=0 N_SEMENTES=5 python -u src/06_submissao.py
python -u src/17_mos_pontual.py saidas/sub07_mme3_norec_t1200l63_s5.csv
python src/19_importancia_plot.py
python predict.py saidas/teste_predict.csv
