#!/bin/bash
#SBATCH -o logs/datarand.sh.log-%j
#SBATCH -c 16
#SBATCH --time=05:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-dinner

module load hdf5

NUM_SAMPLES=200
NUM_IC=30
NUM_JOBS=16

python make_generic.py $NUM_SAMPLES $NUM_IC $NUM_JOBS
