#!/bin/bash

# Path to the parsed PDB output directory
OUTPUT_DIR="/ist-nas/users/bunditb/boltz/scripts/examples/parsed_pdb_output"

# Script to run finalize
FINALIZE_SCRIPT="/ist-nas/users/bunditb/boltz/scripts/command_scripts/run_finalize.py"

# Activate conda environment
source ~/miniforge3/bin/activate
conda activate boltz

echo "Running finalize on PDB output directories..."

# Run the finalize script
python $FINALIZE_SCRIPT --base-dir $OUTPUT_DIR

# Check if we want to run recursively (if --recursive flag is provided)
if [[ "$1" == "--recursive" ]]; then
  echo "Running finalize recursively on all subdirectories..."
  python $FINALIZE_SCRIPT --base-dir $OUTPUT_DIR --recursive
fi

echo "Done!" 