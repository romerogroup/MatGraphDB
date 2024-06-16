#!/bin/bash
#SBATCH -J mp_database_graph_generator
#SBATCH --nodes=2
#SBATCH -c 20
#SBATCH -p standby
#SBATCH -t 4:00:00

# SBATCH -o /users/lllang/SCRATCH/projects/MatGraphDB/scripts/testing_scripts/outputfile.txt

source ~/.bashrc

source /shared/software/conda/conda_init.sh
conda activate matgraphdb

python /users/lllang/SCRATCH/projects/MatGraphDB/scripts/testing_scripts/python_scripts/run_mp_test.py &> /users/lllang/SCRATCH/projects/MatGraphDB/scripts/testing_scripts/outputfile.txt