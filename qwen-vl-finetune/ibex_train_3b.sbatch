#!/bin/bash --login
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2         # number of MP tasks
#SBATCH --gres=gpu:2                # number of GPUs per node
#SBATCH --cpus-per-task=15          # number of cores per tasks
#SBATCH --time=2:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --mem=256G 
#SBATCH --partition=batch 
#SBATCH --mail-type=ALL
#SBATCH --output=results/%x/%j-slurm.out
#SBATCH --error=results/%x/%j-slurm.err

module purge

module load cuda/12.2
module load gcc/12.2.0
#module load cudnn/8.8.1-cuda11.8.0

conda activate /ibex/user/antonic/CESOIA/env

bash train_3b.sh
# bash train_single_gpu.sh
