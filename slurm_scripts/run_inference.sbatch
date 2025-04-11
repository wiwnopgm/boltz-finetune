#!/bin/bash

#SBATCH --job-name=boltz_inference
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-4080
#SBATCH --account=scads
#SBATCH --gres=gpu:1
#SBATCH --mem=48GB
#SBATCH --cpus-per-gpu=6
#SBATCH --time=3-00:00:00
#SBATCH --qos=normal
#SBATCH --nice=0

hostname
nvidia-smi


source ~/miniforge3/bin/activate

source activate /ist-nas/users/bunditb/miniforge3/envs/boltz

# Set environment variables to avoid OOM issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the training with SLURM's srun using the multi-GPU config
boltz predict /ist-nas/users/bunditb/boltz/examples/merck_challenge.yaml \
  --out_dir /ist-nas/users/bunditb/boltz/test_results/large_finetune_1 \
  --checkpoint /ist-nas/users/bunditb/.boltz/finetune_lora.ckpt \
  --use_msa_server \
  --recycling_steps 5 \
  --diffusion_samples 250 \
  
# Deactivate the environment
conda deactivate 