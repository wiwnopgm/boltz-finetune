#!/usr/bin/env python3
"""Process ligand structures from .sdf files and compute conformers and symmetries."""

import argparse
import multiprocessing
import pickle
import sys
import os
import re
from functools import partial
from pathlib import Path

import pandas as pd
from p_tqdm import p_uimap
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer, Mol
from tqdm import tqdm


def load_molecules_from_sdf(directory: str) -> dict[str, Mol]:
    """Load ligand molecules from .sdf files in subdirectories.

    Parameters
    ----------
    directory : str
        Path to the directory containing the ligand structures.

    Returns
    -------
    dict[str, Mol]
        Dictionary mapping ligand names to RDKit molecules.
    """
    molecules = {}
    
    # Walk through all the directories
    for root, dirs, files in os.walk(directory):
        # Find all .sdf files
        for file in files:
            if file.endswith('.sdf'):
                sdf_path = os.path.join(root, file)
                
                # Extract the ligand name from the directory path
                # The directory format is SARS-CoV-2_Mpro-x1374_0A_CONFIDENTIAL
                dir_name = os.path.basename(root)
                
                # Use regex to extract the ID after "Mpro-" and before "_"
                match = re.search(r'Mpro-([^_]+)', dir_name)
                if match:
                    ligand_id = match.group(1)
                else:
                    # Fallback using the directory name
                    ligand_id = dir_name
                
                # Load the molecule
                try:
                    mol = Chem.SDMolSupplier(sdf_path)[0]
                    if mol is not None:
                        # Set the ligand ID as a property
                        mol.SetProp("PDB_NAME", ligand_id)
                        molecules[ligand_id] = mol
                        print(f"Loaded {ligand_id} from {sdf_path}")
                except Exception as e:
                    print(f"Error loading {sdf_path}: {e}")
    
    return molecules


def compute_3d(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults to v3

    Returns
    -------
    bool
        Whether computation was successful.
    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)
        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)
    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")
        return True

    return False


def mark_leaving_atoms(mol: Mol) -> Mol:
    """Mark leaving atoms in the molecule.
    
    Since we don't have specific information about leaving atoms,
    we'll mark all atoms as non-leaving atoms.
    
    Parameters
    ----------
    mol : Mol
        The molecule to mark
        
    Returns
    -------
    Mol
        The input molecule with atoms marked
    """
    # Mark all atoms as non-leaving
    for atom in mol.GetAtoms():
        atom.SetProp("leaving_atom", "0")
    
    return mol


def compute_symmetries(mol: Mol) -> list[list[int]]:
    """Compute the symmetries of a molecule.

    Parameters
    ----------
    mol : Mol
        The molecule to process

    Returns
    -------
    list[list[int]]
        The symmetries as a list of index permutations
    """
    mol = AllChem.RemoveHs(mol)
    # First ensure all atoms have the leaving_atom property
    mol = mark_leaving_atoms(mol)
    
    idx_map = {}
    atom_idx = 0
    for i, atom in enumerate(mol.GetAtoms()):
        # Skip if leaving atoms
        if int(atom.GetProp("leaving_atom")):
            continue
        idx_map[i] = atom_idx
        atom_idx += 1

    # Calculate self permutations
    permutations = []
    raw_permutations = mol.GetSubstructMatches(mol, uniquify=False)
    for raw_permutation in raw_permutations:
        # Filter out permutations with leaving atoms
        try:
            if {raw_permutation[idx] for idx in idx_map} == set(idx_map.keys()):
                permutation = [
                    idx_map[idx] for idx in raw_permutation if idx in idx_map
                ]
                permutations.append(permutation)
        except Exception:  # noqa: S110, PERF203, BLE001
            pass
    serialized_permutations = pickle.dumps(permutations)
    mol.SetProp("symmetries", serialized_permutations.hex())
    return permutations


def process(mol: Mol, output: str) -> tuple[str, str]:
    """Process a ligand molecule.

    Parameters
    ----------
    mol : Mol
        The molecule to process
    output : str
        The directory to save the molecules

    Returns
    -------
    str
        The name of the ligand
    str
        The result of the conformer generation
    """
    # Get name
    name = mol.GetProp("PDB_NAME")

    # Check if single atom
    if mol.GetNumAtoms() == 1:
        result = "single"
    else:
        # Get the 3D conformer - for SDF files, we already have 3D coordinates
        # but we'll try to regenerate them with ETKDG for consistency
        try:
            # Try to generate a 3D conformer with RDKit
            success = compute_3d(mol, version="v3")
            if success:
                result = "computed"
            else:
                result = "original" # Use the original coordinates from the SDF
        except ValueError:
            result = "failed"

    # Compute symmetries
    try:
        compute_symmetries(mol)
    except Exception as e:
        print(f"Error computing symmetries for {name}: {e}")

    # Dump the molecule
    path = Path(output) / f"{name}.pkl"
    with path.open("wb") as f:
        pickle.dump(mol, f)

    # Output the results
    return name, result


def main(args: argparse.Namespace) -> None:
    """Process ligand structures."""
    # Set property saving
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Load molecules
    print("Loading molecules from SDF files")
    molecules = load_molecules_from_sdf(args.input_dir)
    print(f"Loaded {len(molecules)} molecules")

    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # Setup processing function
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mol_output = outdir / "mols"
    mol_output.mkdir(parents=True, exist_ok=True)
    process_fn = partial(process, output=str(mol_output))

    # Process the files in parallel
    print("Processing molecules")
    metadata = []

    # Check if we can run in parallel
    max_processes = multiprocessing.cpu_count()
    num_processes = max(1, min(args.num_processes, max_processes, len(molecules)))
    parallel = num_processes > 1

    if parallel:
        for name, result in p_uimap(
            process_fn,
            molecules.values(),
            num_cpus=num_processes,
        ):
            metadata.append({"name": name, "result": result})
    else:
        for mol in tqdm(molecules.values()):
            name, result = process_fn(mol)
            metadata.append({"name": name, "result": result})

    # Load and group outputs
    processed_molecules = {}
    for item in metadata:
        if item["result"] == "failed":
            continue

        # Load the mol file
        path = mol_output / f"{item['name']}.pkl"
        with path.open("rb") as f:
            mol = pickle.load(f)  # noqa: S301
            processed_molecules[item["name"]] = mol

    # Dump metadata
    path = outdir / "results.csv"
    metadata = pd.DataFrame(metadata)
    metadata.to_csv(path)

    # Dump the final dictionary in the same format as symmetry.pkl
    path = outdir / "ligands.pkl"
    with path.open("wb") as f:
        pickle.dump(processed_molecules, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory containing ligand structures")
    parser.add_argument("--outdir", type=str, help="Output directory")
    parser.add_argument(
        "--num_processes",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use",
    )
    args = parser.parse_args()
    main(args) 