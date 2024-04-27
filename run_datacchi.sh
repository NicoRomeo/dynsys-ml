#!/bin/bash
#SBATCH -o logs/datagen.sh.log-%j
#SBATCH -c 16
#SBATCH --time=24:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-dinner

module load hdf5

NUM_SAMPLES=10000
NUM_IC=30
NUM_JOBS=16

python make_cchi_dynsys.py $NUM_SAMPLES $NUM_IC $NUM_JOBS
