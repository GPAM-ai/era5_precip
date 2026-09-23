#!/bin/bash
#SBATCH -A cptec
#SBATCH -p lncc-cpu_amd
#SBATCH --exclude=sdumont2nd1002,sdumont2nd1003,sdumont2nd1004
#SBATCH --job-name=cons
#SBATCH --output=logs/cons_%j.out
#SBATCH --time=00:40:00
#SBATCH -N 1 -n 1 -c 16
cd "$SLURM_SUBMIT_DIR"
export PATH=/scratch/app/anaconda3/2024.10/bin:$PATH
python -u src/17_mos_pontual.py saidas/sub07_mme3_norec_t1200l63_s5_conservadora.csv
