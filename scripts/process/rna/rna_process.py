#!/usr/bin/env python
"""
RNA structure processing script.

This script processes RNA structures from:
- Pre-parsed PDB files in a directory (/ist-nas/users/bunditb/boltz/scripts/process/rna_example_parsed)

The processed structures are stored in NPZ and JSON format.

Example usage:
-------------
# Process RNA PDB files in a directory
python rna_process.py --datadir /ist-nas/users/bunditb/boltz/scripts/process/rna_example_parsed --outdir /path/to/output/dir \
    --min-length 5 --num-workers 4
"""

import argparse
import json
import multiprocessing
import os
import pickle
import traceback
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, NamedTuple, Tuple

import numpy as np
import rdkit
import sys

# Try to import tqdm
try:
    from tqdm import tqdm
except ImportError:
    # Simple fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    # First try importing as if the script is run from the boltz module
    from boltz.scripts.process.pdb import parse_pdb
    from boltz.data.filter.static.filter import StaticFilter
    from boltz.data.filter.static.ligand import ExcludedLigands
    from boltz.data.filter.static.polymer import (
        MinimumLengthFilter,
        UnknownFilter,
    )
    from boltz.data.types import ChainInfo, InterfaceInfo, Record, Structure, Target
    from boltz.data.types import Atom, Bond, Chain, Residue, Connection, Interface
    
    try:
        from p_tqdm import p_umap
    except ImportError:
        # If p_tqdm is not available, use a simple wrapper
        def p_umap(func, *args, **kwargs):
            return list(map(func, *args))
except ImportError:
    # Then try relative imports if run from the scripts directory
    from pdb import parse_pdb
    
    # Define necessary data structures if we can't import them
    @dataclass
    class Atom:
        atom_id: int
        atom_name: str
        element: str
        entity_id: int
        res_id: int
        ins_code: str
        chain_id: str
        pos: np.ndarray
        occupancy: float = 1.0
        b_factor: float = 0.0
        is_backbone: bool = True
    
    @dataclass
    class Bond:
        atom_1: int
        atom_2: int
        type: int
    
    @dataclass
    class Residue:
        res_id: int
        ins_code: str
        res_name: str
        entity_id: int
        atom_ids: List[int]
        chain_id: str
    
    @dataclass
    class Chain:
        chain_id: str
        chain_type: int
        entity_id: int
    
    @dataclass
    class Connection:
        chain_1: int
        chain_2: int
        residue_1: int
        residue_2: int
        atom_1: int
        atom_2: int
    
    @dataclass
    class Interface:
        chain_1: int
        chain_2: int
    
    @dataclass
    class Structure:
        atoms: List[Atom]
        residues: List[Residue]
        chains: List[Chain]
        connections: List[Connection] = field(default_factory=list)
        bonds: List[Bond] = field(default_factory=list)
        interfaces: List[Interface] = field(default_factory=list)
        mask: np.ndarray = None
        assemblies: Dict = field(default_factory=dict)
        entities: Dict = field(default_factory=dict)
        modified_residues: Dict = field(default_factory=dict)
    
    @dataclass
    class ChainInfo:
        chain_id: int
        chain_name: str
        mol_type: int
        msa_id: str
        cluster_id: int
        num_residues: int
        valid: bool = True
    
    @dataclass
    class InterfaceInfo:
        interface_id: int
        chain_1: int
        chain_2: int
        valid: bool = True
    
    @dataclass
    class StructureInfo:
        id: str
        type: str
        resolution: float
        num_chains: int
        num_residues: int
        num_atoms: int
        num_interfaces: int
    
    @dataclass
    class Record:
        id: str
        structure: Dict
        chains: List[ChainInfo]
        interfaces: List[InterfaceInfo]
        inference_options: Dict = None
    
    @dataclass
    class Target:
        structure: Structure
        record: Record
    
    # The rest can't be imported relatively - the script must be run with proper PYTHONPATH
    # Simplified implementations
    class StaticFilter:
        def filter(self, structure: Structure) -> np.ndarray:
            return np.ones(len(structure.chains), dtype=bool)
    
    class ExcludedLigands(StaticFilter):
        def __init__(self, excluded):
            self.excluded = excluded
    
    class MinimumLengthFilter(StaticFilter):
        def __init__(self, min_len):
            self.min_len = min_len
    
    class UnknownFilter(StaticFilter):
        pass
    
    try:
        from p_tqdm import p_umap
    except ImportError:
        # If p_tqdm is not available, use a simple wrapper
        def p_umap(func, *args, **kwargs):
            return list(map(func, *args))


@dataclass(frozen=True)
class RNASource:
    """A source RNA structure file or identifier."""
    id: str
    path: str
    
# Define ParsedStructure
class ParsedStructure(NamedTuple):
    """Structure after parsing."""
    atoms: List[Atom]
    residues: List[Residue]
    chains: List[Chain]
    connections: List[Connection] = []
    bonds: List[Bond] = []
    interfaces: List[Interface] = []
    assemblies: Dict = {}
    entities: Dict = {}
    modified_residues: Dict = {}
    raw_string: str = ""


def fetch_rna_structures(
    datadir: Path,
    max_file_size: Optional[int] = None
) -> List[RNASource]:
    """Fetch RNA structure files from a directory.
    
    Parameters
    ----------
    datadir : Path
        Directory containing pre-parsed PDB files.
    max_file_size : Optional[int]
        Maximum file size to process.
        
    Returns
    -------
    List[RNASource]
        List of RNA structure sources.
    """
    data = []
    excluded = 0
    
    # Process PDB files
    for file in datadir.rglob("*.pdb"):
        # Skip if too large
        if max_file_size is not None and (file.stat().st_size > max_file_size):
            excluded += 1
            continue
            
        pdb_id = str(file.stem)
        source = RNASource(id=pdb_id, path=str(file))
        data.append(source)
        
    print(f"Found {len(data)} pre-parsed RNA PDB files")
    print(f"Excluded {excluded} files due to size")
    
    return data


def parse_rna_structure_direct(source: RNASource) -> ParsedStructure:
    """
    Parse RNA structure using direct extraction of P atoms. 
    This is used when standard parsers fail.
    
    Args:
        source: The RNA structure source to parse

    Returns:
        A parsed structure containing only P atoms, chain and residue information
    """
    print(f"Direct parsing of {source.id}")
    
    from collections import defaultdict
    import re
    
    try:
        with open(source.path, 'r') as f:
            content = f.read()
            
        # Extract sequence if available
        sequence_match = re.search(r'REMARK\s+999\s+SEQUENCE\s*:?\s*(.+)', content)
        if sequence_match:
            sequence = sequence_match.group(1).strip()
            print(f"Found sequence: {sequence[:20]}...")
        else:
            sequence = None
            
        # Extract all ATOM lines for P atoms (using flexible regex matching)
        atom_lines = []
        for line in content.splitlines():
            # Match ATOM lines and check if they contain P atom information
            if line.startswith('ATOM') and 'P' in line:
                atom_lines.append(line)
        
        print(f"Found {len(atom_lines)} P atoms")
        
        if not atom_lines:
            raise ValueError(f"Failed to parse any atoms from {source.path}")
            
        atoms = []
        residues = defaultdict(list)
        chains = {}
        
        # Regular expression for flexible PDB ATOM line parsing
        # This pattern matches the important components without relying on fixed column positions
        atom_pattern = re.compile(
            r'ATOM\s+(\d+)\s+(\S+)(?:\s+|:)(\S+)(?:\s+|:)([A-Za-z0-9]?)(?:\s+|:)(\d+|[-+]?\d*\.\d+)(?:\s+|:)?' +
            r'([-+]?\d*\.\d+|[-+]?\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)\s+([-+]?\d*\.\d+|[-+]?\d+)'
        )
        
        # Alternate pattern with more flexibility
        alt_pattern = re.compile(
            r'ATOM\s+(\d+)[\s\S]+?'  # ATOM and atom number
            r'([-+]?\d*\.?\d+)\s+'   # X coordinate
            r'([-+]?\d*\.?\d+)\s+'   # Y coordinate
            r'([-+]?\d*\.?\d+)'      # Z coordinate
        )
        
        valid_atom_count = 0
        for i, line in enumerate(atom_lines):
            try:
                # Try primary pattern first
                match = atom_pattern.search(line)
                
                if match:
                    # Extract data from regex groups
                    atom_num = int(match.group(1))
                    atom_name = match.group(2).strip()
                    res_name = match.group(3).strip()
                    chain_id = match.group(4).strip() if match.group(4) else "A"  # Default to chain A if missing
                    res_id = 0
                    
                    # Attempt to parse residue ID as integer
                    try:
                        res_id = int(match.group(5))
                    except ValueError:
                        # If conversion fails, just use the index
                        res_id = i + 1
                    
                    # Parse coordinates
                    x = float(match.group(6))
                    y = float(match.group(7))
                    z = float(match.group(8))
                else:
                    # Try extracting just the atom number and coordinates with alternate pattern
                    alt_match = alt_pattern.search(line)
                    if not alt_match:
                        # If we can't extract this data, skip this line
                        print(f"Skipping atom line {i}, could not parse with regex: {line}")
                        continue
                    
                    # Extract the atom number and coordinates
                    atom_id = int(alt_match.group(1))
                    
                    # Extract best-guess atom name, residue name, chain_id from the line
                    # Look for P atom name
                    atom_name_match = re.search(r'\s+([A-Z0-9\'\*]+)\s+', line[10:20])
                    atom_name = atom_name_match.group(1) if atom_name_match else "P"
                    
                    # Try to extract residue name
                    res_name_match = re.search(r'\s+([A-Z]+)\s+', line[15:25])
                    res_name = res_name_match.group(1) if res_name_match else "X"
                    
                    # Try to extract chain ID - typically a single character
                    chain_id_match = re.search(r'\s([A-Za-z0-9])\s', line[20:25])
                    chain_id = chain_id_match.group(1) if chain_id_match else "A"
                    
                    # Try to extract residue ID
                    res_id_match = re.search(r'\s(\d+)\s', line[22:30])
                    res_id = int(res_id_match.group(1)) if res_id_match else (i + 1)
                    
                    # Get coordinates
                    x = float(alt_match.group(2))
                    y = float(alt_match.group(3))
                    z = float(alt_match.group(4))
                
                # Final fallback - try to extract coordinates using a very flexible approach
                if not (isinstance(x, float) and isinstance(y, float) and isinstance(z, float)):
                    # Try to find any three consecutive floating point numbers
                    numbers = re.findall(r'([-+]?\d*\.\d+|[-+]?\d+)', line)
                    if len(numbers) >= 3:
                        # Skip the first few numbers which might be atom IDs, residue IDs, etc.
                        # Typically coordinates appear after these
                        start_idx = min(3, len(numbers) - 3)  # Start from index 3 or earlier if needed
                        try:
                            x = float(numbers[start_idx])
                            y = float(numbers[start_idx + 1])
                            z = float(numbers[start_idx + 2])
                        except (ValueError, IndexError):
                            print(f"Couldn't extract coordinates from numbers: {numbers}")
                            continue
                    else:
                        print(f"Couldn't find enough numbers in line: {line}")
                        continue
                
                # Create atom object
                atom_id = valid_atom_count + 1  # Generate sequential IDs
                atom = Atom(
                    atom_id=atom_id,
                    atom_name=atom_name,
                    element="P",
                    entity_id=0,
                    res_id=res_id,
                    ins_code="",
                    chain_id=chain_id,
                    pos=np.array([x, y, z], dtype=np.float32),
                    occupancy=1.0,
                    b_factor=0.0,
                    is_backbone=True,
                )
                atoms.append(atom)
                valid_atom_count += 1
                
                # Group atoms by residue
                residue_key = (chain_id, res_id)
                residues[residue_key].append(atom)
                
                # Add chain if not already present
                if chain_id not in chains:
                    chains[chain_id] = Chain(
                        chain_id=chain_id,
                        chain_type=2,  # RNA type
                        entity_id=0,
                    )
            except Exception as e:
                print(f"Error parsing atom line {i}: {e}")
                print(f"Problematic line: {line}")
                # Continue with the next line, don't abort on parsing errors
        
        if not atoms:
            raise ValueError(f"Failed to parse any atoms from {source.path}")
            
        # Sort residues by residue ID
        parsed_residues = []
        for (chain_id, res_id), res_atoms in sorted(residues.items(), key=lambda x: (x[0][0], x[0][1])):
            # Get the residue name from the first atom
            res_name = None
            if res_atoms:
                if hasattr(res_atoms[0], 'res_name'):
                    res_name = res_atoms[0].res_name
                elif hasattr(res_atoms[0], 'atom_name'):
                    res_name = res_atoms[0].atom_name[0]
            
            if not res_name:
                res_name = "X"  # Default to X if we can't determine the residue name
            
            parsed_residues.append(Residue(
                res_id=res_id,
                ins_code="",
                res_name=res_name,
                entity_id=0,
                atom_ids=[atom.atom_id for atom in res_atoms],
                chain_id=chain_id,
            ))
        
        chain_obj_list = list(chains.values())
        print(f"Created structure with {len(parsed_residues)} residues and chain ID {chain_obj_list[0].chain_id if chain_obj_list else 'unknown'}")
            
        return ParsedStructure(
            atoms=atoms,
            residues=parsed_residues,
            chains=chain_obj_list,
            connections=[],
            bonds=[],
            assemblies={},
            entities={},
            modified_residues={},
            raw_string=content,
        )
    except Exception as e:
        raise ValueError(f"Failed to parse {source.id}: {str(e)}")


def parse_rna_structure(source: RNASource, clusters: Dict) -> Target:
    """Parse an RNA structure.
    
    Parameters
    ----------
    source : RNASource
        The RNA structure source.
    resource : Resource
        The shared resource.
    clusters : Dict
        Cluster information.
        
    Returns
    -------
    Target
        The processed structure.
    """
    # Get the structure ID
    structure_id = source.id
    
    # Direct parsing for RNA structures
    try:
        parsed = parse_rna_structure_direct(source)
    except Exception as e:
        print(f"Failed to parse {structure_id}: {e}")
        raise
    
    # Create a structure from the parsed data
    structure = Structure(
        atoms=parsed.atoms,
        residues=parsed.residues,
        chains=parsed.chains,
        connections=parsed.connections,
        bonds=parsed.bonds,
        interfaces=[],
        mask=np.ones(len(parsed.chains), dtype=bool),
        assemblies=parsed.assemblies,
        entities=parsed.entities,
        modified_residues=parsed.modified_residues,
    )
    
    # Create structure info
    structure_info = {
        "id": structure_id,
        "type": "rna",
        "resolution": 0.0,
        "num_chains": len(parsed.chains),
        "num_residues": len(parsed.residues),
        "num_atoms": len(parsed.atoms),
        "num_interfaces": 0,
    }
    
    # Create chain metadata
    chain_info = []
    for i, chain in enumerate(parsed.chains):
        chain_name = chain.chain_id        
        chain_info.append(
            ChainInfo(
                chain_id=i,
                chain_name=chain_name,
                msa_id=structure_id,
                mol_type=2,  # RNA
                cluster_id=clusters.get(structure_id, -1),
                num_residues=sum(1 for r in parsed.residues if r.chain_id == chain_name),
                valid=True
            )
        )
    
    # No interfaces for RNA structures
    interface_info = []
    
    # Create record
    record = Record(
        id=source.id,
        structure=structure_info,
        chains=chain_info,
        interfaces=interface_info,
        inference_options=None
    )
    
    return Target(structure=structure, record=record)


def process_rna_structure(
    source: RNASource,
    outdir: Path,
    filters: List[StaticFilter],
    clusters: Dict,
) -> None:
    """Process an RNA structure.
    
    Parameters
    ----------
    source : RNASource
        The RNA structure source.
    outdir : Path
        The output directory.
    filters : List[StaticFilter]
        Filters to apply.
    clusters : Dict
        Cluster information.
    """
    # Check if already processed
    struct_path = outdir / "structures" / f"{source.id}.npz"
    record_path = outdir / "records" / f"{source.id}.json"
    
    if struct_path.exists() and record_path.exists():
        return
    
    try:
        # Parse the structure
        target = parse_rna_structure(source, clusters)
        structure = target.structure
        
        # Apply filters
        mask = np.ones(len(structure.chains), dtype=bool) if structure.mask is None else structure.mask
        if filters is not None:
            for f in filters:
                filter_mask = f.filter(structure)
                mask = mask & filter_mask
        
        structure.mask = mask
    except Exception as e:
        print(f"Failed to parse {source.id}: {e}")
        traceback.print_exc()
        return
    
    # Update chain and interface validity
    chains = []
    for i, chain in enumerate(target.record.chains):
        chains.append(replace(chain, valid=bool(mask[i])))
    
    interfaces = []
    for interface in target.record.interfaces:
        chain_1_valid = bool(mask[interface.chain_1])
        chain_2_valid = bool(mask[interface.chain_2])
        interfaces.append(replace(interface, valid=(chain_1_valid and chain_2_valid)))
    
    # Update structure and record
    record = replace(target.record, chains=chains, interfaces=interfaces)
    target = replace(target, structure=structure, record=record)
    
    # Save structure as NPZ
    os.makedirs(os.path.dirname(struct_path), exist_ok=True)
    
    # Convert structure to dictionary of numpy arrays
    structure_dict = {
        "atoms": np.array([vars(atom) for atom in structure.atoms]),
        "residues": np.array([vars(res) for res in structure.residues]),
        "chains": np.array([vars(chain) for chain in structure.chains]),
        "mask": structure.mask
    }
    
    np.savez_compressed(struct_path, **structure_dict)
    
    # Save record as JSON
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    with record_path.open("w") as f:
        json.dump(asdict(record), f)


def create_manifest(outdir: Path) -> None:
    """Create a manifest of all processed structures.
    
    Parameters
    ----------
    outdir : Path
        The output directory.
    """
    records_dir = outdir / "records"
    
    # Collect all records
    failed_count = 0
    records = []
    
    for record_file in records_dir.iterdir():
        try:
            with record_file.open("r") as f:
                records.append(json.load(f))
        except Exception:
            failed_count += 1
            print(f"Failed to parse {record_file}")
    
    # Print statistics
    print(f"Successfully processed {len(records)} RNA structures")
    if failed_count > 0:
        print(f"Failed to parse {failed_count} records")
    
    # Save manifest
    manifest_path = outdir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(records, f)
    
    print(f"Saved manifest to {manifest_path}")


def process_rna_dataset(args) -> None:
    """Process the RNA structures.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    # Create output directories
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "records").mkdir(parents=True, exist_ok=True)
    (args.outdir / "structures").mkdir(parents=True, exist_ok=True)
    
    # Create filters
    filters = [
        UnknownFilter(),
        MinimumLengthFilter(min_len=args.min_length),
    ]
    
    # Load excluded ligands if specified
    if args.excluded_ligands:
        with open(args.excluded_ligands, "r") as f:
            excluded = json.load(f)
        filters.append(ExcludedLigands(excluded=excluded))
    
    # Load clustering information if provided
    clusters = {}
    if args.cluster_file is not None:
        with open(args.cluster_file, "r") as f:
            cluster_data = json.load(f)["clusters"]
        for entity_id, cluster_id in cluster_data.items():
            clusters[entity_id] = int(cluster_id)
    
    # Fetch RNA structures
    rna_sources = fetch_rna_structures(
        datadir=args.datadir,
        max_file_size=args.max_file_size
    )
    
    # Process structures
    print(f"Processing {len(rna_sources)} RNA structures")
    
    process_func = partial(
        process_rna_structure,
        outdir=args.outdir,
        filters=filters,
        clusters=clusters,
    )
    
    # Process sequentially or in parallel
    if args.num_workers <= 1:
        for i, source in enumerate(tqdm(rna_sources)):
            process_func(source)
            if args.num and (i + 1) >= args.num:
                break
    else:
        # Process in parallel
        _ = p_umap(
            process_func,
            rna_sources[:args.num] if args.num else rna_sources,
            num_cpus=args.num_workers,
            disable=False,
        )
    
    # Create manifest
    create_manifest(args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RNA structures")
    
    # Input/output options
    parser.add_argument("--datadir", type=Path, required=True, help="Directory containing pre-parsed RNA PDB files")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory")
    
    # Filtering options
    parser.add_argument("--max-file-size", type=int, help="Maximum file size to process")
    parser.add_argument("--min-length", type=int, default=5, help="Minimum RNA chain length")
    parser.add_argument("--cluster-file", type=str, help="Path to cluster information")
    parser.add_argument("--excluded-ligands", type=str, help="Path to excluded ligands list")
    
    # Processing options
    parser.add_argument("--num-workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--num", type=int, help="Maximum number of structures to process")
    
    args = parser.parse_args()
    process_rna_dataset(args) 