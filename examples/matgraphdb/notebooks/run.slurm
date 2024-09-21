#!/bin/bash
#SBATCH -J mp_database
#SBATCH --nodes=1
#SBATCH -c 40
#SBATCH -p standby
#SBATCH -t 4:00:00

source ~/.bashrc

source /shared/software/conda/conda_init.sh
conda activate matgraphdb

python "/users/lllang/SCRATCH/projects/MatGraphDB/examples/matgraphdb/notebooks/Example 3 - Importing A Large Dataset.py" > /users/lllang/SCRATCH/projects/MatGraphDB/examples/matgraphdb/notebooks/output.log 2>&1