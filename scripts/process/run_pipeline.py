#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    print(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")
    
    print(f"{description} completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Run the processing pipeline")
    parser.add_argument("--data_dir", type=Path, required=True,
                      help="Directory containing CIF/PDB files")
    parser.add_argument("--msa_dir", type=Path, required=True,
                      help="Directory containing MSA files")
    parser.add_argument("--output_dir", type=Path, required=True,
                      help="Base directory for output")
    parser.add_argument("--redis_host", type=str, default="localhost",
                      help="Redis host (default: localhost)")
    parser.add_argument("--ccd_port", type=int, default=7777,
                      help="Port for CCD Redis server (default: 7777)")
    parser.add_argument("--taxonomy_port", type=int, default=7778,
                      help="Port for taxonomy Redis server (default: 7778)")
    parser.add_argument("--num_processes", type=int, default=4,
                      help="Number of processes to use (default: 4)")
    parser.add_argument("--max_seqs", type=int, default=1000,
                      help="Maximum number of sequences to process (default: 1000)")
    
    args = parser.parse_args()
    
    # Get the absolute path to the scripts directory
    script_dir = Path(__file__).parent.absolute()
    
    # Create output directories
    structures_output_dir = args.output_dir / "processed_structures"
    msa_output_dir = args.output_dir / "processed_msa"
    structures_output_dir.mkdir(parents=True, exist_ok=True)
    msa_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process structures using CCD Redis server
    run_command(
        ["python", str(script_dir / "rcsb.py"), 
         "--datadir", str(args.data_dir),
         "--outdir", str(structures_output_dir),
         "--redis-host", args.redis_host,
         "--redis-port", str(args.ccd_port)],
        "Processing CIF/PDB files"
    )
    
    # Process MSA files using taxonomy Redis server
    run_command(
        ["python", str(script_dir / "msa.py"), 
         "--msadir", str(args.msa_dir),
         "--outdir", str(msa_output_dir),
         "--redis-host", args.redis_host,
         "--redis-port", str(args.taxonomy_port),
         "--max-seqs", str(args.max_seqs)],
        "Processing MSA files"
    )
    
    print("\nPipeline completed successfully!")
    print(f"Processed structures saved to: {structures_output_dir}")
    print(f"Processed MSA files saved to: {msa_output_dir}")

if __name__ == "__main__":
    main()
