#!/bin/bash
#SBATCH -J mp_database_wyckoff
#SBATCH --nodes=1
#SBATCH -c 40
#SBATCH -p comm_small_day
#SBATCH -t 24:00:00

source ~/.bashrc

source /shared/software/conda/conda_init.sh
conda activate matgraphdb

python /users/lllang/SCRATCH/projects/MatGraphDB/scripts/database/python_scripts/run_wyckoff_calc.py