#!/usr/bin/env python
"""
Example script demonstrating how to use the Stanford RNA dataset parsing functions.

This script shows how to:
1. Convert Stanford RNA dataset entries to PDB format
2. Parse these PDB files into ParsedStructure objects for further processing
"""

import os
import sys
import argparse
from typing import Dict, List

from rdkit.Chem import AllChem
import numpy as np

# Fix the import path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

# Import the necessary modules with direct relative import
from pdb import prepare_rna_from_csv, parse_stanford_rna_structure


def load_components() -> Dict:
    """Load the PDB components dictionary.
    
    This is a simplified version that loads a minimal set of components.
    
    Returns
    -------
    Dict
        Dictionary of PDB components
    """
    # Create a minimal components dictionary with RNA nucleotides
    components = {}
    
    # Define the standard RNA nucleotides
    nucleotides = ['A', 'C', 'G', 'U']
    
    for nuc in nucleotides:
        # Create a simple molecule for each nucleotide
        mol = AllChem.MolFromSmiles('P')  # Phosphate atom
        mol.SetProp("name", nuc)
        
        # Set atom properties
        for atom in mol.GetAtoms():
            atom.SetProp("name", "P")  # Set atom name
        
        # Store as single letter key
        components[nuc] = mol
    
    print(f"Loaded {len(components)} components")
    return components


def main(args):
    """Main function to demonstrate RNA structure parsing."""
    print("Demonstrating Stanford RNA dataset parsing")
    
    # Load RNA components
    components = load_components()
    
    # Set paths to the CSV files
    csv_path = args.labels_csv or "/ist-nas/users/bunditb/boltz/stanford-rna/train_labels.csv"
    seq_csv_path = args.sequences_csv or "/ist-nas/users/bunditb/boltz/stanford-rna/train_sequences.csv"
    
    # Create output directory for PDB files
    output_dir = args.output_dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate PDB files for all RNA structures
    if args.generate_all:
        print(f"Generating PDB files for all RNA structures in {csv_path}")
        pdb_paths = prepare_rna_from_csv(csv_path, seq_csv_path, output_dir)
        print(f"Generated {len(pdb_paths)} PDB files")
        
        # Print the first few RNA IDs
        print("First few RNA IDs:")
        for i, rna_id in enumerate(list(pdb_paths.keys())[:5]):
            print(f"  {i+1}. {rna_id}: {pdb_paths[rna_id]}")
    
    # Parse specific RNA structure
    if args.rna_id:
        print(f"Parsing RNA structure: {args.rna_id}")
        try:
            parsed_structure = parse_stanford_rna_structure(
                args.rna_id,
                components,
                csv_path,
                seq_csv_path,
                output_dir
            )
            
            # Print structure info
            print("Structure info:")
            print(f"  Number of chains: {parsed_structure.data.chains.shape[0]}")
            print(f"  Number of residues: {parsed_structure.data.residues.shape[0]}")
            print(f"  Number of atoms: {parsed_structure.data.atoms.shape[0]}")
            
            # Print first few atoms
            if parsed_structure.data.atoms.shape[0] > 0:
                print("First few atoms:")
                for i in range(min(5, parsed_structure.data.atoms.shape[0])):
                    atom = parsed_structure.data.atoms[i]
                    print(f"  Atom {i+1}: Element={atom['element']}, Coords={atom['coords']}")
            
        except Exception as e:
            print(f"Error parsing RNA structure {args.rna_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demonstrate Stanford RNA dataset parsing")
    parser.add_argument("--rna_id", type=str, help="RNA structure ID to parse (e.g., '1SCL_A')")
    parser.add_argument("--labels_csv", type=str, help="Path to the labels CSV file")
    parser.add_argument("--sequences_csv", type=str, help="Path to the sequences CSV file")
    parser.add_argument("--output_dir", type=str, help="Directory to save generated PDB files")
    parser.add_argument("--generate_all", action="store_true", help="Generate PDB files for all RNA structures")
    
    args = parser.parse_args()
    
    if not (args.rna_id or args.generate_all):
        parser.error("At least one of --rna_id or --generate_all must be specified")
    
    main(args) 