#!/usr/bin/env python3
"""
Script to run the finalize function on all subdirectories in the parsed_pdb_output directory.
This script directly implements the finalize logic to avoid dependency issues.
Now it accumulates all records into a single manifest.json file.
"""

import os
import sys
import json
from pathlib import Path
import argparse
from tqdm import tqdm

def process_and_accumulate_records(base_dir, recursive=False, limit=None):
    """
    Process all records in subdirectories and accumulate them into a single list.
    
    Parameters
    ----------
    base_dir : Path
        The base directory containing PDB output subdirectories
    recursive : bool
        Whether to process recursively
    limit : int, optional
        Limit the number of directories to process
    
    Returns
    -------
    list
        List of all processed records
    dict
        Statistics of processing
    """
    # Find all relevant directories
    if recursive:
        # Find all subdirectories that contain a 'records' folder
        processed_dirs = []
        for root, dirs, files in os.walk(base_dir):
            root_path = Path(root)
            records_dir = root_path / "records"
            if records_dir.exists() and records_dir.is_dir():
                processed_dirs.append(root_path)
                if limit and len(processed_dirs) >= limit:
                    break
    else:
        # Process only immediate subdirectories of the base directory
        processed_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        if limit:
            processed_dirs = processed_dirs[:limit]
    
    if not processed_dirs:
        print("No directories found to process.")
        return [], {"total_dirs": 0, "processed_dirs": 0, "total_records": 0, "failed_records": 0}
    
    print(f"Found {len(processed_dirs)} directories to process.")
    
    # Accumulate all records
    all_records = []
    total_failed = 0
    success_count = 0
    
    # Process each directory
    for directory in tqdm(processed_dirs, desc="Processing directories"):
        records_dir = directory / "records"
        if not records_dir.exists() or not records_dir.is_dir():
            print(f"No records directory found in {directory}")
            continue
        
        dir_failed = 0
        dir_records = []
        
        # Process each record in this directory
        for record_file in records_dir.iterdir():
            try:
                with record_file.open("r") as f:
                    record_data = json.load(f)
                    dir_records.append(record_data)
            except Exception as e:
                dir_failed += 1
                print(f"Failed to parse {record_file}: {str(e)}")
        
        # Add directory records to the global list
        all_records.extend(dir_records)
        total_failed += dir_failed
        
        # Report on this directory
        if dir_failed > 0:
            print(f"Directory {directory.name}: Failed to parse {dir_failed} entries, added {len(dir_records)} entries")
        else:
            success_count += 1
    
    stats = {
        "total_dirs": len(processed_dirs),
        "processed_dirs": success_count,
        "total_records": len(all_records),
        "failed_records": total_failed
    }
    
    return all_records, stats

def main():
    parser = argparse.ArgumentParser(description="Run finalize on parsed PDB output directories")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/ist-nas/users/bunditb/boltz/scripts/examples/parsed_pdb_output",
        help="Base directory containing PDB output subdirectories"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,  # Default to base_dir/manifest.json
        help="Path to the output manifest.json file"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process recursively (find all subdirectories containing 'records' folder)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of directories to process"
    )
    
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' does not exist.")
        return 1
    
    print(f"Processing directories in: {base_dir}")
    
    # Process and accumulate all records
    all_records, stats = process_and_accumulate_records(base_dir, args.recursive, args.limit)
    
    # Determine output file path
    if args.output_file:
        output_path = Path(args.output_file)
    else:
        output_path = base_dir / "manifest.json"
    
    # Save the accumulated records to a single manifest file
    with output_path.open("w") as f:
        json.dump(all_records, f)
    
    print(f"\nFinalize Summary:")
    print(f"  Total directories processed: {stats['total_dirs']}")
    print(f"  Directories with all records successful: {stats['processed_dirs']}")
    print(f"  Total records accumulated: {stats['total_records']}")
    print(f"  Failed records: {stats['failed_records']}")
    print(f"  Manifest saved to: {output_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 