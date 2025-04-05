#!/usr/bin/env python3

import os
import glob
import re

# Path to the ligand-posing directory (using absolute path)
base_dir = "/ist-nas/users/bunditb/boltz/scripts/examples/raw_data_package/ligand-posing"

# Output directory
output_dir = "/ist-nas/users/bunditb/boltz/scripts/examples/compiled_fasta_files"

# Find all protein.fasta files in subdirectories
fasta_files = glob.glob(os.path.join(base_dir, "**/protein.fasta"), recursive=True)
print(f"Found {len(fasta_files)} protein.fasta files")

# Process each fasta file
for fasta_file in fasta_files:
    print(f"Processing {fasta_file}")
    
    with open(fasta_file, 'r') as f:
        content = f.read()
    
    # Split the file content by '>' to get individual sequences
    sequences = content.split('>')
    sequences = [seq for seq in sequences if seq.strip()]  # Remove empty strings
    
    # Process each sequence
    for seq in sequences:
        lines = seq.strip().split('\n')
        header = lines[0]
        sequence_data = '\n'.join(lines[1:])
        
        # Extract just the identifier part for the filename
        # Example: from "Mpro-J0050_0B_A" we want "J0050_0B_A"
        parts = header.split('-')
        if len(parts) > 1:
            # If the header has a hyphen, use the part after the first hyphen
            identifier = parts[1]
            # Remove any unwanted characters
            identifier = re.sub(r'[^\w_]', '', identifier)
            filename = f"{identifier}.fasta"
        else:
            # If there's no hyphen, use the whole header but sanitize it
            identifier = re.sub(r'[^\w_]', '', header)
            filename = f"{identifier}.fasta"
        
        output_path = os.path.join(output_dir, filename)
        
        with open(output_path, 'w') as f:
            f.write(f">{header}\n{sequence_data}\n")
        print(f"Saved {header} to {output_path}")

print(f"Completed. All sequences have been saved to individual files.") 