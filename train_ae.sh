#!/bin/env bash

#SBATCH --partition=shared-gpu
#SBATCH --time=6:00:00
#SBATCH --gpus=nvidia_rtx_a6000:1
#SBATCH --mem=24GB
#SBATCH --job-name=ae_training

CONTAINER="$HOME/scratch/ctlearnenv.sif"

BIND="$HOME/scratch:/mnt,/srv/beegfs/scratch/shares/upeguipa:/mnt_data"

module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

apptainer exec --nv --bind $BIND --pwd /mnt $CONTAINER python3 /mnt/ae_train.py --config /mnt/config.yaml
