#!/bin/bash

#SBATCH --job-name=boltz_training_multi
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=scads-a100
#SBATCH --account=scads
#SBATCH --gres=gpu:2
#SBATCH --mem=192GB
#SBATCH --cpus-per-gpu=6
#SBATCH --time=3-00:00:00
#SBATCH --nice=0

hostname
nvidia-smi

# Set environment variables to avoid OOM issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Removed CUDA_VISIBLE_DEVICES to allow access to all 4 GPUs