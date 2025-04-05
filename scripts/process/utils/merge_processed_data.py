import os
import json
import shutil
from pathlib import Path

def merge_processed_data(brd4_dir, fkbp_dir, output_dir):
    """
    Merge processed data from BRD4 and FKBP binders into a single directory.
    
    Args:
        brd4_dir (str): Path to BRD4_binder_processed directory
        fkbp_dir (str): Path to FKBP_binder_processed directory
        output_dir (str): Path to output directory
    """
    # Convert to Path objects
    brd4_dir = Path(brd4_dir)
    fkbp_dir = Path(fkbp_dir)
    output_dir = Path(output_dir)
    
    # Create output directory structure
    output_records = output_dir / "records"
    output_structures = output_dir / "structures"
    output_records.mkdir(parents=True, exist_ok=True)
    output_structures.mkdir(parents=True, exist_ok=True)
    
    # Load manifests
    with open(brd4_dir / "manifest.json", 'r') as f:
        brd4_manifest = json.load(f)
    
    with open(fkbp_dir / "manifest.json", 'r') as f:
        fkbp_manifest = json.load(f)
    
    # Combine manifests (they are lists)
    combined_manifest = brd4_manifest + fkbp_manifest
    
    # Create a set of record IDs for quick lookup
    brd4_ids = {record['id'] for record in brd4_manifest}
    fkbp_ids = {record['id'] for record in fkbp_manifest}
    
    # Copy files from BRD4
    print("Copying BRD4 files...")
    for record in brd4_manifest:
        record_id = record['id']
        # Copy record file
        src_record = brd4_dir / "records" / f"{record_id}.json"
        dst_record = output_records / f"{record_id}.json"
        shutil.copy2(src_record, dst_record)
        
        # Copy structure file if it exists
        src_structure = brd4_dir / "structures" / f"{record_id}.npz"
        if src_structure.exists():
            dst_structure = output_structures / f"{record_id}.npz"
            shutil.copy2(src_structure, dst_structure)
    
    # Copy files from FKBP
    print("Copying FKBP files...")
    for record in fkbp_manifest:
        record_id = record['id']
        # Copy record file
        src_record = fkbp_dir / "records" / f"{record_id}.json"
        dst_record = output_records / f"{record_id}.json"
        shutil.copy2(src_record, dst_record)
        
        # Copy structure file if it exists
        src_structure = fkbp_dir / "structures" / f"{record_id}.npz"
        if src_structure.exists():
            dst_structure = output_structures / f"{record_id}.npz"
            shutil.copy2(src_structure, dst_structure)
    
    # Save combined manifest
    with open(output_dir / "manifest.json", 'w') as f:
        json.dump(combined_manifest, f, indent=2)
    
    print(f"\nMerge completed successfully!")
    print(f"Total records in combined manifest: {len(combined_manifest)}")
    print(f"BRD4 records: {len(brd4_manifest)}")
    print(f"FKBP records: {len(fkbp_manifest)}")
    print(f"Unique BRD4 IDs: {len(brd4_ids)}")
    print(f"Unique FKBP IDs: {len(fkbp_ids)}")
    print(f"Total unique IDs: {len(brd4_ids | fkbp_ids)}")
    print(f"\nOutput directory: {output_dir}")

def main():
    # Define paths
    base_dir = Path("/ist-nas/users/bunditb/boltz/scripts/merk_challenge")
    brd4_dir = base_dir / "BRD4_binder/BRD4_binder_processed"
    fkbp_dir = base_dir / "FKBP_binder/FKBP_binder_processed"
    output_dir = base_dir / "combined_processed"
    
    # Merge the data
    merge_processed_data(brd4_dir, fkbp_dir, output_dir)

if __name__ == '__main__':
    main() 