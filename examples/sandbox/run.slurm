#!/bin/bash
#SBATCH -J mongodb
#SBATCH --nodes=1
#SBATCH -c 40
#SBATCH -p standby
#SBATCH -t 4:00:00

source ~/.bashrc

source /shared/software/conda/conda_init.sh
conda activate matgraphdb


numactl --interleave=all mongod --dbpath /users/lllang/SCRATCH/projects/MatGraphDB/data/db --port 27017 --bind_ip localhost --logpath /users/lllang/SCRATCH/projects/MatGraphDB/data/db/mongod.log &

python /users/lllang/SCRATCH/projects/MatGraphDB/examples/sandbox/pymongo_test.py

mongod --shutdown --dbpath /users/lllang/SCRATCH/projects/MatGraphDB/data/db