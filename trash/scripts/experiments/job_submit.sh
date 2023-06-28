#!/bin/bash

#PBS -N poly_graph_gpu
#PBS -q comm_gpu_week
#PBS -l walltime=168:00:00
#PBS -l nodes=1:ppn=4:gpus=3
#PBS -m ae
#PBS -M lllang@mix.wvu.edu
#PBS -j oe


source ~/.bashrc

use_torch

cd $PBS_O_WORKDIR

python single_test_run.py