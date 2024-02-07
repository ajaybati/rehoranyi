#!/bin/bash

#SBATCH --job-name=fullnoNL2
#SBATCH --output=fullnoNL2.out
#SBATCH --error=fullnoNL2.err
#SBATCH --partition=hoffman-lab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node="a40:2"
#SBATCH --qos="short"

export PYTHONUNBUFFERED=TRUE
source /nethome/abati7/.bashrc
. "/nethome/abati7/flash/miniconda3/etc/profile.d/conda.sh"
conda activate cv
cd /nethome/abati7/flash/Work/jat

hostname
nvidia-smi

srun -u python -u jatModel.py