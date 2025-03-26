#!/bin/bash
#SBATCH --job-name=boltz_training
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-4080
#SBATCH --account=scads
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --cpus-per-gpu=6
#SBATCH --time=12:00:00
#SBATCH --qos=normal
#SBATCH --nice=0

# Print out the hostname and GPU info for debugging purposes
hostname
nvidia-smi

# Load modules if needed
# module load cuda/11.7

# Activate the boltz environment
source ~/miniforge3/bin/activate boltz

# Set environment variables to avoid OOM issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# Print current working directory
echo "Current working directory: $(pwd)"

# Change to the boltz directory
cd ~/boltz

# Run the training with SLURM's srun
python scripts/train/train.py scripts/train/configs/confidence_example.yaml \
    trainer.precision=16

# Deactivate the environment
conda deactivate 