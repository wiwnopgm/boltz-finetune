#!/usr/bin/env python
"""
Process CIF files to extract PDB IDs, fetch FASTA sequences, and save them in a single folder.

This script:
1. Reads all CIF files from the specified directory
2. Extracts the PDB ID from each filename
3. Fetches the FASTA sequence using the RCSB PDB API
4. Saves all FASTA sequences in a single folder with uppercase filenames
"""

import os
import re
import requests
import time
from pathlib import Path
from tqdm import tqdm
import argparse

def fetch_fasta(pdb_id):
    """Fetch FASTA sequence directly from RCSB PDB API."""
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error fetching FASTA for {pdb_id}: HTTP {response.status_code}")
        return None

def extract_pdb_id(filename):
    """Extract PDB ID from filename."""
    # Remove file extension and any additional text in parentheses
    base_name = os.path.splitext(filename)[0]
    # Remove any text in parentheses
    base_name = re.sub(r'\s*\([^)]*\)', '', base_name)
    # Convert to uppercase
    return base_name.upper()

def process_cif_files(cif_dir, output_dir):
    """Process all CIF files in the directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all CIF files
    cif_files = [f for f in os.listdir(cif_dir) if f.endswith('.cif')]
    
    print(f"Found {len(cif_files)} CIF files to process")
    
    # Process each CIF file
    for cif_file in tqdm(cif_files, desc="Processing CIF files"):
        # Extract PDB ID
        pdb_id = extract_pdb_id(cif_file)
        
        # Fetch FASTA sequence
        fasta_content = fetch_fasta(pdb_id)
        
        if fasta_content:
            # Save FASTA content to file in the output directory
            fasta_file = os.path.join(output_dir, f"{pdb_id}.fasta")
            with open(fasta_file, 'w') as f:
                f.write(fasta_content)
        
        # Add a small delay to avoid overwhelming the API
        time.sleep(0.5)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process CIF files to FASTA format')
    parser.add_argument('--cif_dir', type=str, required=True,
                      help='Directory containing CIF files')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save FASTA files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process CIF files
    process_cif_files(args.cif_dir, args.output_dir)
    
    print("Processing complete!") 