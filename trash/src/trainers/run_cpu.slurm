#!/bin/bash

#SBATCH -J Poly_Featurization
#SBATCH --nodes=40
#SBATCH -p comm_small_day
#SBATCH -t 1-00:00:00


export NUM_CORES=$((SLURM_JOB_NUM_NODES * SLURM_CPUS_ON_NODE))

source ~/.bashrc
module purge
module load atomistic/vasp/6.2.1_intel22_impi22

echo $SLURM_SUBMIT_DIR > log.out
cd $SLURM_SUBMIT_DIR

python data_pipeline.py