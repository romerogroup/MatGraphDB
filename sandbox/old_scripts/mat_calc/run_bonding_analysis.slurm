#!/bin/bash

#SBATCH -J mp_database
#SBATCH --nodes=1
#SBATCH -c 20
#SBATCH -p standby
#SBATCH -t 4:00:00


export NUM_CORES=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_ON_NODE))
source ~/.bashrc

cd $SLURM_SUBMIT_DIR


echo $SLURM_SUBMIT_DIR > run_analysis_debug.txt
source venv/bin/activate

python poly_graphs_lib/database/json/bonding_analysis.py > run_bonding_analysis_debug.txt


