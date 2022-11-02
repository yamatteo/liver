#!/bin/bash

#SBATCH --job-name="pipelined_ltd_train"          # name of job
#SBATCH -A otc@v100
#SBATCH -C v100-32g

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10

#SBATCH --hint=nomultithread         # hyperthreading is deactivated
#SBATCH --time=8:00:00              # maximum execution time requested (HH:MM:SS)
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

# Code execution
cd "/gpfswork/rech/otc/uiu95bi/runs"
source /gpfswork/rech/otc/uiu95bi/hogwild_liver/venv/bin/activate
wandb offline
PYTHONPATH=/gpfswork/rech/otc/uiu95bi/hogwild_liver srun python "/gpfswork/rech/otc/uiu95bi/hogwild_liver/main.py"