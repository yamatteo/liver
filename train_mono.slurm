#!/bin/bash

#SBATCH --job-name="ltd_train"          # name of job
#SBATCH -A otc@v100

#SBATCH -C v100-32g                 # uncomment to target only 32GB V100 GPU

# Here, reservation of 10 CPUs (for 1 task) and 1 GPU on a single node:
#SBATCH --nodes=1                    # we request one node
#SBATCH --ntasks-per-node=1          # with one task per node (= number of GPUs here)
#SBATCH --gres=gpu:4                 # number of GPUs per node (max 8 with gpu_p2, gpu_p4, gpu_p5)
#SBATCH --cpus-per-task=10           # number of cores per task (1/4 of the 4-GPUs node)

# /!\ Caution, "multithread" in Slurm vocabulary refers to hyperthreading.
#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=08:00:00              # maximum execution time requested (HH:MM:SS)
#SBATCH --output=/gpfswork/rech/otc/uiu95bi/runs/%j.out              # name of output file
#SBATCH --error=/gpfswork/rech/otc/uiu95bi/runs/%j.out               # name of error file (here, in common with the output file)

# Cleans out the modules loaded in interactive and inherited by default
module purge

# Uncomment the following module command if you are using the "gpu_p5" partition
# to have access to the modules compatible with this partition.
#module load cpuarch/amd

# Loading of modules
module load pytorch-gpu/py3/1.12.1

# Echo of launched commands
set -x

# For the "gpu_p5" partition, the code must be compiled with the compatible modules.
# Code execution
cd "/gpfswork/rech/otc/uiu95bi/runs"
source /gpfswork/rech/otc/uiu95bi/venv/bin/activate
wandb offline
PYTHONPATH=/gpfswork/rech/otc/uiu95bi/liver srun python "/gpfswork/rech/otc/uiu95bi/liver/train.py"