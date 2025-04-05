import os
import random
import argparse
from pathlib import Path

def generate_validation_ids(data_dir, output_file, prob=0.2):
    """
    Generate validation IDs from .cif files in the data directory.
    
    Args:
        data_dir (str): Directory containing .cif files
        output_file (str): Path to save validation IDs
        prob (float): Probability of selecting each ID (default: 0.2)
    """
    # Convert to Path objects
    data_dir = Path(data_dir)
    output_file = Path(output_file)
    
    # Get all .cif files
    cif_files = [f for f in os.listdir(data_dir) if f.endswith('.cif')]
    
    # Extract PDB IDs (remove .cif extension)
    pdb_ids = [os.path.splitext(f)[0] for f in cif_files]
    
    # Select IDs based on probability
    selected_ids = [pdb_id for pdb_id in pdb_ids if random.random() < prob]
    
    # Sort the IDs for consistency
    selected_ids.sort()
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to file
    with open(output_file, 'w') as f:
        for pdb_id in selected_ids:
            f.write(f"{pdb_id}\n")
    
    print(f"Generated {len(selected_ids)} validation IDs in {output_file}")
    print(f"Total available IDs: {len(pdb_ids)}")
    print(f"Selection probability: {prob}")

def main():
    parser = argparse.ArgumentParser(description='Generate validation IDs from .cif files')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing .cif files')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to save validation IDs')
    parser.add_argument('--prob', type=float, default=0.2,
                      help='Probability of selecting each ID (default: 0.2)')
    
    args = parser.parse_args()
    generate_validation_ids(args.data_dir, args.output_file, args.prob)

if __name__ == '__main__':
    main() 