#!/bin/bash

#SBATCH --job-name=${JOB_NAME:-boltz_training_array}
#SBATCH --output=${SLURM_OUTPUT_DIR:-slurm-%A_%a.out}
#SBATCH --error=${SLURM_ERROR_DIR:-slurm-%A_%a.err}
#SBATCH --nodes=1
#SBATCH --partition=${SLURM_PARTITION:-scads-a100}
#SBATCH --account=${SLURM_ACCOUNT:-scads}
#SBATCH --gres=gpu:${NUM_GPUS:-2}
#SBATCH --mem=${MEMORY:-192GB}
#SBATCH --cpus-per-gpu=${CPUS_PER_GPU:-6}
#SBATCH --time=${TIME_LIMIT:-3-00:00:00}
#SBATCH --nice=0
#SBATCH --array=0-${ARRAY_SIZE:-8}

# Define the parameter values - customize these arrays as needed
declare -a PARAM1_VALUES=(${PARAM1_VALUES:-0.3 0.5 0.7})
declare -a PARAM2_VALUES=(${PARAM2_VALUES:-1 2 4})
declare -a PARAM3_VALUES=(${PARAM3_VALUES:-0.001 0.0018 0.002})

# Calculate indices for the current array task
PARAM1_IDX=$((SLURM_ARRAY_TASK_ID % ${#PARAM1_VALUES[@]}))
PARAM2_IDX=$((SLURM_ARRAY_TASK_ID / ${#PARAM1_VALUES[@]} % ${#PARAM2_VALUES[@]}))
PARAM3_IDX=$((SLURM_ARRAY_TASK_ID / (${#PARAM1_VALUES[@]} * ${#PARAM2_VALUES[@]}) % ${#PARAM3_VALUES[@]}))

# Get the actual values
PARAM1=${PARAM1_VALUES[$PARAM1_IDX]}
PARAM2=${PARAM2_VALUES[$PARAM2_IDX]}
PARAM3=${PARAM3_VALUES[$PARAM3_IDX]}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Parameter 1: $PARAM1"
echo "Parameter 2: $PARAM2"
echo "Parameter 3: $PARAM3"

hostname
nvidia-smi

# Activate the conda environment
source ${CONDA_PATH:-~/mambaforge}/bin/activate ${CONDA_ENV:-boltz}

# Set environment variables to avoid OOM issues
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print current working directory
echo "Current working directory: $(pwd)"

# Change to the project directory
cd ${PROJECT_DIR:-.}

# Create output directory for this specific run
OUTPUT_DIR="${OUTPUT_BASE_DIR:-./output}/param1_${PARAM1}_param2_${PARAM2}_param3_${PARAM3}"
mkdir -p $OUTPUT_DIR

# Run the training with the specific parameters
python ${TRAIN_SCRIPT:-scripts/train/train.py} ${CONFIG_FILE:-scripts/train/configs/full_finetune.yaml} \
    ${PARAM1_PATH:-data.param1}=${PARAM1} \
    ${PARAM2_PATH:-data.param2}=${PARAM2} \
    ${PARAM3_PATH:-model.param3}=${PARAM3} \
    output=${OUTPUT_DIR} \
    debug=${DEBUG:-1}

# Deactivate the environment
conda deactivate 