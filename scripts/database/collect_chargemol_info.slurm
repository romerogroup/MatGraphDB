#!/bin/bash
#SBATCH -J mp_database_chargemol
#SBATCH --nodes=1
#SBATCH -n 40
#SBATCH -p comm_small_day
#SBATCH -t 24:00:00

source ~/.bashrc

source /shared/software/conda/conda_init.sh
conda activate matgraphdb

python /users/lllang/SCRATCH/projects/MatGraphDB/scripts/database/collect_chargemol_info.py