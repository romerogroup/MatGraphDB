#!/bin/bash
#SBATCH -J generate_material_graph
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH -p standby
#SBATCH -t 4:00:00

source ~/.bashrc

source /shared/software/conda/conda_init.sh
conda activate matgraphdb

python /users/lllang/SCRATCH/projects/MatGraphDB/scripts/database/python_scripts/generate_material_graph.py