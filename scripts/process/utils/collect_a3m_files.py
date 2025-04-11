#!/usr/bin/env python3

import os
import shutil
import argparse
from pathlib import Path

def collect_a3m_files(source_dir, output_dir):
    """
    Collect uniref.a3m files from source directory and its subdirectories.
    Rename them to lowercase PDB IDs.
    
    Args:
        source_dir (str): Source directory containing the PDB folders
        output_dir (str): Output directory for collected files
    """
    # Convert to Path objects
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Keep track of statistics
    total_files = 0
    processed_pdbs = set()
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(source_dir):
        # Get the PDB_id from the path
        path_parts = Path(root).parts
        # Find the PDB ID (it's the folder name that matches the pattern of 4 characters followed by a number)
        pdb_id = None
        for part in path_parts:
            if len(part) >= 4 and part[0].isdigit() and part[1:4].isalnum():
                pdb_id = part[:4]  # Take just the first 4 characters (e.g., "7P6Y")
                break
        
        if pdb_id and 'uniref.a3m' in files:
            # Source file path
            source_file = os.path.join(root, 'uniref.a3m')
            # Destination file path (using lowercase PDB_id)
            dest_file = os.path.join(output_dir, f"{pdb_id.lower()}.a3m")
            
            # Copy the file with the new name
            print(f"Copying {source_file} to {dest_file}")
            shutil.copy2(source_file, dest_file)
            total_files += 1
            processed_pdbs.add(pdb_id)
    
    print("\nCollection complete!")
    print(f"Total uniref.a3m files collected: {total_files}")
    print(f"Unique PDB IDs processed: {len(processed_pdbs)}")
    print(f"Output directory: {output_dir}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Collect uniref.a3m files and rename them to lowercase PDB IDs.')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Source directory containing the PDB folders')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for collected files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Collect the files
    collect_a3m_files(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main() 