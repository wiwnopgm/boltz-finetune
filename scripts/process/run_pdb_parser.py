#!/usr/bin/env python
"""Test script for PDB parser implementation."""

import argparse
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path
import pickle
import sys
import os

import numpy as np

# Add script directory to path so imports work when run as a script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

# Import the custom PDB parser
from pdb import parse_pdb
from rcsb import Resource, PDB, parse

from boltz.data.filter.static.filter import StaticFilter
from boltz.data.filter.static.ligand import ExcludedLigands
from boltz.data.filter.static.polymer import (
    ClashingChainsFilter,
    ConsecutiveCA,
    MinimumLengthFilter,
    UnknownFilter,
)

def main():
    """Run the test script."""
    parser = argparse.ArgumentParser(description="Test PDB parser")
    parser.add_argument("--pdb-file", type=str, required=True, help="Path to PDB file to parse")
    parser.add_argument("--output-dir", type=Path, default=Path("./output"), help="Output directory")
    parser.add_argument("--cluster-file", type=str, help="Path to cluster information JSON file")
    parser.add_argument("--min-length", type=int, default=30, help="Min chain length")
    parser.add_argument("--excluded-ligands", type=str, help="Excluded ligands")
    parser.add_argument("--redis-host", type=str, default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=7777, help="Redis port")
    parser.add_argument("--force", action="store_true", help="Force reprocessing even if output files exist")
    
    args = parser.parse_args()
    
    # Create output directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = args.output_dir / "structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup resource using the provided Redis connection parameters
    resource = Resource(host=args.redis_host, port=args.redis_port)
    
    # Load clusters from JSON file if provided
    clusters = {}
    if args.cluster_file is not None:
        try:
            with open(args.cluster_file, "r") as f:
                cluster_data = json.load(f)["clusters"]
            for entity_id, cluster_id in cluster_data.items():
                clusters[entity_id] = int(cluster_id)
            print(f"Loaded {len(clusters)} clusters from {args.cluster_file}")
        except Exception as e:
            print(f"Warning: Failed to load cluster file: {e}")
    
    # Create PDB object
    pdb_file = Path(args.pdb_file)
    file_format = "pdb" if pdb_file.suffix.lower() == ".pdb" else "mmcif"
    data = PDB(id=pdb_file.stem.lower(), path=str(pdb_file), format=file_format)
    
    # Create filters
    filters = [
        UnknownFilter(),
        MinimumLengthFilter(min_len=args.min_length),
        ConsecutiveCA(),
        ClashingChainsFilter(),
    ]
    
    if args.excluded_ligands:
        try:
            excluded_file = Path(args.excluded_ligands)
            with excluded_file.open("r") as f:
                excluded = json.load(f)
            filters.append(ExcludedLigands(excluded=excluded))
            print(f"Loaded excluded ligands from {args.excluded_ligands}")
        except Exception as e:
            print(f"Warning: Failed to load excluded ligands: {e}")
    
    # Check if we need to process
    struct_path = structure_dir / f"{data.id}.npz"
    record_path = records_dir / f"{data.id}.json"
    
    if struct_path.exists() and record_path.exists() and not args.force:
        print(f"Files already exist: {struct_path} and {record_path}")
        print("Use --force to reprocess")
        return
    
    print(f"Processing file: {args.pdb_file}")
    
    try:
        # Parse the target
        target = parse(data, resource, clusters)
        structure = target.structure
        
        # Apply the filters
        mask = structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
    except Exception:
        traceback.print_exc()
        print(f"Failed to parse {data.id}")
        return
    
    # Replace chains and interfaces
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))
    
    interfaces = []
    for interface in target.record.interfaces:
        chain_1 = bool(mask[interface.chain_1])
        chain_2 = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1 and chain_2)))
    
    # Replace structure and record
    structure = replace(structure, mask=mask)
    record = replace(target.record, chains=chains, interfaces=interfaces)
    target = replace(target, structure=structure, record=record)
    
    # Dump structure
    np.savez_compressed(struct_path, **asdict(structure))
    
    # Dump record
    with record_path.open("w") as f:
        json.dump(asdict(record), f)
    
    print(f"Successfully processed file: {args.pdb_file}")
    print(f"Structure saved to: {struct_path}")
    print(f"Record saved to: {record_path}")
    
if __name__ == "__main__":
    main() 