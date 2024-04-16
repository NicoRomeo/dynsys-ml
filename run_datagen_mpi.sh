#!/bin/bash
#SBATCH -o logs/datagen.sh.log-%j
#SBATCH -n 16
#SBATCH --time=24:00:00
#SBATCH --partition=caslake
#SBATCH --account=pi-dinner

module load openmpi
module load hdf5
export OMP_NUM_THREADS=1

NUM_SAMPLES=16
NUM_IC=30

mpiexec -n 16 python make_fhn_systems_mpi.py $NUM_SAMPLES $NUM_IC
