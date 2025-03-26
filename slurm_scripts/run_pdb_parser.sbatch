#!/bin/bash

#SBATCH --job-name=PDB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --nodes=1
#SBATCH --partition=gpu-4080
#SBATCH --account=scads
#SBATCH --gres=gpu:4
#SBATCH --mem=64GB
#SBATCH --cpus-per-gpu=8
#SBATCH --time=1-0:0:0

echo "========================SLURM JOB SUBMISSION============================"
echo "Nodelist =" $SLURM_JOB_NODELIST
echo "Nnodes =" $SLURM_JOB_NUM_NODES
echo "NCPU per GPU =" $SLURM_CPUS_PER_GPU
echo "Memory allocated =" $SLURM_MEM_PER_NODE "MB"
echo "GPU type =" $(nvidia-smi -L | head -n 1)
echo "Num GPUs =" $(nvidia-smi -L | wc -l)
echo "GPU Memory =" $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)
echo "========================================================================"

# Load necessary modules
ml CUDA/12.1.1
ml GCC/12.3.0

# Source conda
source ~/miniforge3/bin/activate
conda activate boltz

# Correctly start Redis server with separate dir and filename parameters
redis-server --dir /ist-nas/users/bunditb/boltz --dbfilename ccd.rdb --port 7777 &
REDIS_PID=$!

# Wait a moment for Redis to start
sleep 5

# Set directories
PDB_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/complex_pdb_files"
OUTPUT_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/parsed_pdb_output"
PARSER_SCRIPT="/ist-nas/users/bunditb/boltz/scripts/process/test_pdb_parser.py"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Get a list of all PDB files
PDB_FILES=($(find $PDB_DIR -name "*.pdb"))

# Calculate total number of files
TOTAL_FILES=${#PDB_FILES[@]}
echo "Total PDB files found: $TOTAL_FILES"

# Check if running as a SLURM array job
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Running as array job task $SLURM_ARRAY_TASK_ID"
    
    # Calculate the index of the file to process (SLURM_ARRAY_TASK_ID is 1-based)
    INDEX=$((SLURM_ARRAY_TASK_ID - 1))
    
    # Check if index is valid
    if [ $INDEX -ge $TOTAL_FILES ]; then
        echo "Error: Array task ID $SLURM_ARRAY_TASK_ID exceeds number of files ($TOTAL_FILES)"
        # Shutdown Redis server
        redis-cli -p 7777 shutdown
        exit 1
    fi
    
    # Process single file
    PDB_FILE=${PDB_FILES[$INDEX]}
    
    # Extract the basename for organized output
    BASE_NAME=$(basename "$PDB_FILE" .pdb)
    FILE_OUTPUT_DIR="$OUTPUT_DIR/$BASE_NAME"
    
    echo "Processing file: $PDB_FILE"
    echo "Output directory: $FILE_OUTPUT_DIR"
    
    # Create output directory if it doesn't exist
    mkdir -p "$FILE_OUTPUT_DIR"
    
    # Run the PDB parser
    python $PARSER_SCRIPT \
      --pdb-file "$PDB_FILE" \
      --output-dir "$FILE_OUTPUT_DIR" \
      --min-length 50 \
      --redis-host localhost \
      --redis-port 7777 \
      --force
    
    echo "Completed processing: $PDB_FILE"
else
    # Not an array job, process all files sequentially
    echo "Running as a single job, processing all $TOTAL_FILES files sequentially"
    
    for PDB_FILE in "${PDB_FILES[@]}"; do
        # Extract the basename for organized output
        BASE_NAME=$(basename "$PDB_FILE" .pdb)
        FILE_OUTPUT_DIR="$OUTPUT_DIR/$BASE_NAME"
        
        echo "Processing file: $PDB_FILE"
        echo "Output directory: $FILE_OUTPUT_DIR"
        
        # Create output directory if it doesn't exist
        mkdir -p "$FILE_OUTPUT_DIR"
        
        # Run the PDB parser
        python $PARSER_SCRIPT \
          --pdb-file "$PDB_FILE" \
          --output-dir "$FILE_OUTPUT_DIR" \
          --min-length 50 \
          --redis-host localhost \
          --redis-port 7777 \
          --force
        
        echo "Completed processing: $PDB_FILE"
    done
fi

# Shutdown Redis server properly
redis-cli -p 7777 shutdown

echo "All processing completed!" 