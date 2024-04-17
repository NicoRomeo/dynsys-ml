#!/bin/bash
#SBATCH --job-name=dynsys    # create a short name for your job
#SBATCH -o logs/training_cpu.sh.log-%j
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --partition=caslake
#SBATCH -c 1                     # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=24:00:00
#SBATCH --account=pi-dinner

conda activate ./env/torch
module load hdf5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python train.py