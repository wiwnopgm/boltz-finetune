#!/bin/bash

#SBATCH --job-name=boltz_inference
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-4080
#SBATCH --account=scads
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --time=1-0:0:0

hostname
nvidia-smi

# Activate conda environment
source ~/miniforge3/bin/activate boltz

# Set environment variables
export PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Print current working directory
echo "Current working directory: $(pwd)"

# Run the training with SLURM's srun using the multi-GPU config
boltz predict /ist-nas/users/bunditb/boltz/examples/merck_challenge.yaml \
  --out_dir /ist-nas/users/bunditb/boltz/scripts/merck_challenge/finetune_prediction_2 \
  --checkpoint /ist-nas/users/bunditb/.boltz/boltz1_finetune1.ckpt \
  --use_msa_server \
  --diffusion_samples 4 \
  
# Deactivate the environment
conda deactivate 

# --recycling_steps	