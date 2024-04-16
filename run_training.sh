#!/bin/bash
#SBATCH --job-name=dynsys    # create a short name for your job
#SBATCH -o logs/training_gpu.sh.log-%j
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
# TO USE V100 specify --constraint=v100
# TO USE RTX600 specify --constraint=rtx6000
#SBATCH --constraint=v100   # constraint job runs on V100 GPU use
#SBATCH -c 1                     # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=24:00:00
#SBATCH --account=pi-dinner

conda activate ./env/torch
module load hdf5
module load cuda/10.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python train.py