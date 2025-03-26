import contextlib
from dataclasses import dataclass, replace
from typing import Optional, Any, List, Tuple, Dict, Set

import gemmi
import numpy as np
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import Conformer, Mol
from sklearn.neighbors import KDTree

from boltz.data import const
from boltz.data.types import (
    Atom,
    Bond,
    Chain,
    Connection,
    Interface,
    Residue,
    Structure,
    StructureInfo,
)

# Define compatible versions of the mmcif classes for Python 3.9
# instead of importing directly from mmcif

@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""
    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int

@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""
    atom_1: int
    atom_2: int
    type: int

@dataclass(frozen=True)
class ParsedResidue:
    """A parsed residue object."""
    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool

@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""
    name: str
    entity: str
    type: str
    residues: list[ParsedResidue]
    sequence: list[str]

@dataclass(frozen=True)
class ParsedConnection:
    """A parsed connection object."""
    chain_1: str
    chain_2: str
    residue_index_1: int
    residue_index_2: int
    atom_index_1: str
    atom_index_2: str

@dataclass(frozen=True)
class ParsedStructure:
    """A parsed structure object."""
    data: Structure
    info: StructureInfo
    covalents: list[int]

# Import important functions from mmcif.py here for direct use
# Define the needed functions to avoid importing from mmcif.py

def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    msg = "Conformer does not exist."
    raise ValueError(msg)


def get_unk_token(dtype: gemmi.PolymerType) -> str:
    """Get the unknown token for a given entity type.

    Parameters
    ----------
    dtype : gemmi.EntityType
        The entity type.

    Returns
    -------
    str
        The unknown token.

    """
    if dtype == gemmi.PolymerType.PeptideL:
        unk = const.unk_token["PROTEIN"]
    elif dtype == gemmi.PolymerType.Dna:
        unk = const.unk_token["DNA"]
    elif dtype == gemmi.PolymerType.Rna:
        unk = const.unk_token["RNA"]
    else:
        msg = f"Unknown polymer type: {dtype}"
        raise ValueError(msg)

    return unk


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def compute_interfaces(atom_data: np.ndarray, chain_data: np.ndarray) -> np.ndarray:
    """Compute the chain-chain interfaces from a gemmi structure.

    Parameters
    ----------
    atom_data : List[tuple]
        The atom data.
    chain_data : List[tuple]
        The chain data.

    Returns
    -------
    List[tuple[int, int]]
        The interfaces.

    """
    # Compute chain_id per atom
    chain_ids = []
    for idx, chain in enumerate(chain_data):
        chain_ids.extend([idx] * chain["atom_num"])
    chain_ids = np.array(chain_ids)

    # Filter to present atoms
    coords = atom_data["coords"]
    mask = atom_data["is_present"]

    coords = coords[mask]
    chain_ids = chain_ids[mask]

    # Compute the distance matrix
    tree = KDTree(coords, metric="euclidean")
    query = tree.query_radius(coords, const.atom_interface_cutoff)

    # Get unique chain pairs
    interfaces = set()
    for c1, pairs in zip(chain_ids, query):
        chains = np.unique(chain_ids[pairs])
        chains = chains[chains != c1]
        interfaces.update((c1, c2) for c2 in chains)

    # Get unique chain pairs
    interfaces = [(min(i, j), max(i, j)) for i, j in interfaces]
    interfaces = list({(int(i), int(j)) for i, j in interfaces})
    interfaces = np.array(interfaces, dtype=Interface)
    return interfaces


# Import these functions from mmcif.py
# This is a simplified approach - for a full solution, we would need to
# directly implement all mmcif parsing functions used
from mmcif import (
    parse_ccd_residue,
    parse_polymer,
    parse_connection,
    compute_covalent_ligands,
)


def parse_pdb(
    path: str,
    components: Dict[str, Mol],
    use_assembly: bool = True,
    test_mode: bool = False,  # Added for testing purpose
) -> ParsedStructure:
    """Parse a structure in PDB format.

    Parameters
    ----------
    path : str
        Path to the PDB file.
    components: Dict[str, Mol]
        The preprocessed PDB components dictionary.
    use_assembly: bool
        Whether to use the first assembly.
    test_mode: bool
        If True, don't raise errors for empty structures, useful for testing.

    Returns
    -------
    ParsedStructure
        The parsed structure.

    """
    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # Parse PDB input file
    print(f"Reading PDB file: {path}")
    structure = gemmi.read_structure(str(path))
    
    # Set up entities for the structure (important for PDB format)
    structure.setup_entities()
    
    # Print detailed entity info for debugging
    print("\n=== ENTITY INFORMATION ===")
    try:
        entity_count = len(structure.entities)
        print(f"PDB structure loaded. Contains {entity_count} entities.")
        for i, entity in enumerate(structure.entities):
            print(f"Entity {i}: Type={entity.entity_type.name}, Name={entity.name}, Polymer={entity.polymer_type.name if hasattr(entity, 'polymer_type') and entity.entity_type.name == 'Polymer' else 'N/A'}")
            print(f"  Subchains: {entity.subchains}")
            if entity.entity_type.name == "Polymer" and hasattr(entity, "full_sequence"):
                print(f"  Sequence length: {len(entity.full_sequence)}")
                print(f"  First 10 residues: {entity.full_sequence[:10]}")
    except (AttributeError, TypeError) as e:
        print(f"Warning: Could not determine entity count: {e}")
        print(f"PDB structure loaded successfully.")
    
    # Print detailed model information
    print("\n=== MODEL INFORMATION ===")
    for model_idx, model in enumerate(structure):
        print(f"Model {model_idx}: {len(model)} chains")
        for chain_idx, chain in enumerate(model):
            print(f"  Chain {chain_idx}: {chain.name}, {len(chain)} residues")
            try:
                polymer_residues = list(chain.get_polymer())
                print(f"    Polymer: {len(polymer_residues)} residues")
                if polymer_residues:
                    print(f"    First 5 residue names: {[res.name for res in polymer_residues[:5]]}")
                    print(f"    First 5 residue sequence numbers: {[str(res.seqid.num) for res in polymer_residues[:5]]}")
                
                # Count ligands and other residues
                ligand_residues = [res for res in chain if is_ligand(res)]
                print(f"    Ligands: {len(ligand_residues)} residues")
                if ligand_residues:
                    print(f"    Ligand names: {[res.name for res in ligand_residues]}")
            except Exception as e:
                print(f"    Error processing chain {chain.name}: {e}")
    
    # Print subchain information
    print("\n=== SUBCHAIN INFORMATION ===")
    for subchain_idx, subchain in enumerate(structure[0].subchains()):
        subchain_id = subchain.subchain_id()
        print(f"  Subchain {subchain_idx}: ID={subchain_id}")
        try:
            residue_count = len(list(subchain))
            print(f"    Residues: {residue_count}")
            if residue_count > 0:
                first_residues = list(subchain)[:5]
                print(f"    First 5 residue names: {[res.name for res in first_residues]}")
        except Exception as e:
            print(f"    Error processing subchain {subchain_id}: {e}")

    # Extract metadata - for PDB format, this is different from MMCIF
    # We'll have to extract this information from the HEADER and REMARK records
    deposit_date = ""
    release_date = ""
    revision_date = ""
    resolution = 0.0
    method = ""

    # Try to extract metadata from the file itself
    # PDB files store the deposition date in the HEADER record
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Extract metadata from PDB REMARK records if available
                    if hasattr(atom, 'remark'):
                        if 'DEPOSITION DATE' in atom.remark:
                            deposit_date = atom.remark.split('DEPOSITION DATE')[1].strip()
                        if 'RESOLUTION' in atom.remark:
                            try:
                                resolution = float(atom.remark.split('RESOLUTION')[1].strip().split()[0])
                            except (ValueError, IndexError):
                                pass
                        if 'EXPERIMENTAL METHOD' in atom.remark:
                            method = atom.remark.split('EXPERIMENTAL METHOD')[1].strip().lower()
                    break
                break
            break
        break

    # Clean up the structure
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

    # Expand assembly 1 if requested
    if use_assembly and structure.assemblies:
        print(f"Transforming to assembly: {structure.assemblies[0].name}")
        how = gemmi.HowToNameCopiedChain.AddNumber
        assembly_name = structure.assemblies[0].name
        structure.transform_to_assembly(assembly_name, how=how)

    # Parse entities
    # Create mapping from subchain id to entity
    entities: Dict[str, gemmi.Entity] = {}
    entity_ids: Dict[str, int] = {}
    for entity_id, entity in enumerate(structure.entities):
        entity: gemmi.Entity
        if entity.entity_type.name == "Water":
            continue
        print(f"Entity {entity_id}: Type={entity.entity_type.name}, Name={entity.name}, Subchains={entity.subchains}")
        for subchain_id in entity.subchains:
            entities[subchain_id] = entity
            entity_ids[subchain_id] = entity_id

    # Create mapping from chain name to entity
    chain_to_entity_id = {}
    for entity_id, entity in enumerate(structure.entities):
        for subchain in entity.subchains:
            chain_name = subchain[0]  # First character is chain name
            chain_to_entity_id[chain_name] = entity_id

    # Create mapping from chain, residue to subchains
    # since a Connection uses the chains and not subchins
    subchain_map = {}
    for chain in structure[0]:
        for residue in chain:
            seq_id = residue.seqid
            seq_id = str(seq_id.num) + str(seq_id.icode).strip()
            subchain_map[(chain.name, seq_id)] = residue.subchain

    # Find covalent ligands
    covalent_chain_ids = compute_covalent_ligands(
        connections=structure.connections,
        subchain_map=subchain_map,
        entities=entities,
    )

    # Parse chains
    chains: List[ParsedChain] = []
    chain_seqs = []
    
    # First, process chains directly using model[0] access method
    print(f"\n=== PROCESSING CHAINS ===")
    print(f"Structure has {len(structure[0])} chains")
    for chain_id, chain in enumerate(structure[0]):
        chain_name = chain.name
        print(f"\nProcessing chain {chain_name} ({chain_id}/{len(structure[0])})")
        
        # Check if this chain has a polymer
        try:
            polymer_residues = list(chain.get_polymer())
            if polymer_residues:
                print(f"Chain {chain_name} has polymer with {len(polymer_residues)} residues")
                print(f"Residue names sample: {[res.name for res in polymer_residues[:5]]}...")
                
                # Find corresponding entity
                entity = None
                for e in structure.entities:
                    if e.entity_type.name == "Polymer" and chain_name in [sc[0] for sc in e.subchains]:
                        entity = e
                        break
                
                print(f"Found entity for chain {chain_name}: {entity.name if entity else 'None'}")
                print(f"Entity polymer type: {entity.polymer_type.name if entity and hasattr(entity, 'polymer_type') else 'Unknown'}")
                
                if entity and entity.polymer_type.name in {"PeptideL", "Dna", "Rna"}:
                    # Get the subchain ID - should be the first character of chain name in most cases
                    subchain_id = chain_name
                    entity_name = entity.name if entity else "Unknown"
                    
                    # Use the simplified polymer parser for PDB files
                    print(f"Parsing polymer chain {chain_name} with {len(polymer_residues)} residues")
                    parsed_polymer = parse_polymer_simple(
                        polymer_chain=chain,
                        components=components,
                        chain_id=subchain_id,
                        entity_name=entity_name,
                    )
                    
                    if parsed_polymer is not None:
                        chains.append(parsed_polymer)
                        chain_seqs.append(parsed_polymer.sequence)
                        print(f"Added polymer chain: {chain_name}")
                        print(f"Chain has {len(parsed_polymer.residues)} residues")
                        print(f"Sequence (first 20): {''.join(parsed_polymer.sequence[:20])}")
                    else:
                        print(f"Failed to parse polymer chain {chain_name}")
            else:
                print(f"Chain {chain_name} has no polymer residues")
                
            # Also check for ligands (non-polymer residues)
            ligand_residues = [res for res in chain if is_ligand(res)]
            if ligand_residues:
                print(f"Chain {chain_name} has {len(ligand_residues)} ligand residues")
                
                # Skip UNL or other missing ligands
                valid_ligands = [lig for lig in ligand_residues if components.get(lig.name) is not None]
                if not valid_ligands:
                    print(f"No valid ligands found in chain {chain_name}")
                    continue
                
                residues = []
                for lig_idx, ligand in enumerate(valid_ligands):
                    # Always treat as non-covalent for simplicity
                    is_covalent = False
                    
                    residue = parse_ccd_residue(
                        name=ligand.name,
                        components=components,
                        res_idx=lig_idx,
                        gemmi_mol=ligand,
                        is_covalent=is_covalent,
                    )
                    if residue:
                        residues.append(residue)
                        print(f"Added ligand: {ligand.name}")
                
                if residues:
                    chains.append(
                        ParsedChain(
                            name=chain_name,
                            entity="",  # No entity for ligands
                            residues=residues,
                            type=const.chain_type_ids["NONPOLYMER"],
                            sequence=None
                        )
                    )
                    print(f"Added non-polymer chain: {chain_name}")
                
        except Exception as e:
            print(f"Error processing chain {chain_name}: {e}")
            continue
    
    # If no chains parsed from direct chain access, fall back to the subchain method
    if not chains:
        print("No chains found using direct chain access, falling back to subchain method...")
        print(f"Processing {len(list(structure[0].subchains()))} subchains")
        for raw_chain in structure[0].subchains():
            # Check chain type
            subchain_id = raw_chain.subchain_id()
            
            # Skip this chain if entity not found
            if subchain_id not in entities:
                print(f"Warning: Entity for subchain '{subchain_id}' not found, skipping.")
                continue
                
            entity: gemmi.Entity = entities[subchain_id]
            entity_type = entity.entity_type.name
            print(f"Processing subchain {subchain_id}: Type={entity_type}")

            # Parse a polymer
            if entity_type == "Polymer":
                # Skip PeptideD, DnaRnaHybrid, Pna, Other
                if entity.polymer_type.name not in {
                    "PeptideL",
                    "Dna",
                    "Rna",
                }:
                    print(f"Skipping polymer type: {entity.polymer_type.name}")
                    continue

                # Add polymer if successful
                parsed_polymer = parse_polymer(
                    polymer=raw_chain,
                    polymer_type=entity.polymer_type,
                    sequence=entity.full_sequence,
                    chain_id=subchain_id,
                    entity=entity.name,
                    components=components,
                )
                if parsed_polymer is not None:
                    chains.append(parsed_polymer)
                    chain_seqs.append(parsed_polymer.sequence)
                    print(f"Added polymer: {subchain_id}")
                else:
                    print(f"Failed to parse polymer: {subchain_id}")

            # Parse a non-polymer
            elif entity_type in {"NonPolymer", "Branched"}:
                # Skip UNL or other missing ligands
                missing_ligands = [lig.name for lig in raw_chain if components.get(lig.name) is None]
                if missing_ligands:
                    print(f"Skipping ligands not in components: {missing_ligands}")
                    continue

                residues = []
                for lig_idx, ligand in enumerate(raw_chain):
                    # Check if ligand is covalent
                    if entity_type == "Branched":
                        is_covalent = True
                    else:
                        is_covalent = subchain_id in covalent_chain_ids

                    ligand: gemmi.Residue
                    residue = parse_ccd_residue(
                        name=ligand.name,
                        components=components,
                        res_idx=lig_idx,
                        gemmi_mol=ligand,
                        is_covalent=is_covalent,
                    )
                    residues.append(residue)
                    print(f"Added ligand: {ligand.name}")

                if residues:
                    chains.append(
                        ParsedChain(
                            name=subchain_id,
                            entity=entity.name,
                            residues=residues,
                            type=const.chain_type_ids["NONPOLYMER"],
                            sequence=None
                        )
                    )
                    print(f"Added non-polymer: {subchain_id}")

    # If no chains parsed, fail or return empty structure based on test_mode
    if not chains:
        msg = "No chains parsed!"
        print(msg)
        if test_mode:
            # Create empty structure for testing
            info = StructureInfo(
                deposited=deposit_date,
                revised=revision_date,
                released=release_date,
                resolution=resolution,
                method=method,
                num_chains=0,
                num_interfaces=0,
            )
            
            # Create empty arrays
            atoms = np.array([], dtype=Atom)
            bonds = np.array([], dtype=Bond)
            residues = np.array([], dtype=Residue)
            chains = np.array([], dtype=Chain)
            connections = np.array([], dtype=Connection)
            interfaces = np.array([], dtype=Interface)
            mask = np.array([], dtype=bool)
            
            data = Structure(
                atoms=atoms,
                bonds=bonds,
                residues=residues,
                chains=chains,
                connections=connections,
                interfaces=interfaces,
                mask=mask,
            )
            
            return ParsedStructure(data=data, info=info, covalents=[])
        else:
            raise ValueError(msg)

    # Parse covalent connections
    connections: List[ParsedConnection] = []
    for connection in structure.connections:
        # Skip non-covalent connections
        connection: gemmi.Connection
        if connection.type.name != "Covale":
            continue

        parsed_connection = parse_connection(
            connection=connection,
            chains=chains,
            subchain_map=subchain_map,
        )
        connections.append(parsed_connection)

    # Create tables
    atom_data = []
    bond_data = []
    res_data = []
    chain_data = []
    connection_data = []

    # Convert parsed chains to tables
    atom_idx = 0
    res_idx = 0
    asym_id = 0
    sym_count = {}
    chain_to_idx = {}
    res_to_idx = {}

    for asym_id, chain in enumerate(chains):
        # Compute number of atoms and residues
        res_num = len(chain.residues)
        atom_num = sum(len(res.atoms) for res in chain.residues)

        # Find all copies of this chain in the assembly
        # Use a default entity_id if not found
        chain_name = chain.name or ""  # Handle empty chain names
        entity_id = chain_to_entity_id.get(chain_name, 0)
        sym_id = sym_count.get(entity_id, 0)
        chain_data.append(
            (
                chain.name,
                chain.type,
                entity_id,
                sym_id,
                asym_id,
                atom_idx,
                atom_num,
                res_idx,
                res_num,
            )
        )
        chain_to_idx[chain.name] = asym_id
        sym_count[entity_id] = sym_id + 1

        # Add residue, atom, bond, data
        for i, res in enumerate(chain.residues):
            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append(
                (
                    res.name,
                    res.type,
                    res.idx,
                    atom_idx,
                    len(res.atoms),
                    atom_center,
                    atom_disto,
                    res.is_standard,
                    res.is_present,
                )
            )
            res_to_idx[(chain.name, i)] = (res_idx, atom_idx)

            for bond in res.bonds:
                atom_1 = atom_idx + bond.atom_1
                atom_2 = atom_idx + bond.atom_2
                bond_data.append((atom_1, atom_2, bond.type))

            for atom in res.atoms:
                atom_data.append(
                    (
                        convert_atom_name(atom.name),
                        atom.element,
                        atom.charge,
                        atom.coords,
                        atom.conformer,
                        atom.is_present,
                        atom.chirality,
                    )
                )
                atom_idx += 1

            res_idx += 1

    # Convert connections to tables
    for conn in connections:
        chain_1_idx = chain_to_idx[conn.chain_1]
        chain_2_idx = chain_to_idx[conn.chain_2]
        res_1_idx, atom_1_offset = res_to_idx[(conn.chain_1, conn.residue_index_1)]
        res_2_idx, atom_2_offset = res_to_idx[(conn.chain_2, conn.residue_index_2)]
        atom_1_idx = atom_1_offset + conn.atom_index_1
        atom_2_idx = atom_2_offset + conn.atom_index_2
        connection_data.append(
            (
                chain_1_idx,
                chain_2_idx,
                res_1_idx,
                res_2_idx,
                atom_1_idx,
                atom_2_idx,
            )
        )

    # Convert into datatypes
    atoms = np.array(atom_data, dtype=Atom)
    bonds = np.array(bond_data, dtype=Bond)
    residues = np.array(res_data, dtype=Residue)
    chains = np.array(chain_data, dtype=Chain)
    connections = np.array(connection_data, dtype=Connection)
    mask = np.ones(len(chain_data), dtype=bool)

    # Compute interface chains (find chains with a heavy atom within 5A)
    interfaces = compute_interfaces(atoms, chains)

    # Return parsed structure
    info = StructureInfo(
        deposited=deposit_date,
        revised=revision_date,
        released=release_date,
        resolution=resolution,
        method=method,
        num_chains=len(chains),
        num_interfaces=len(interfaces),
    )

    data = Structure(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        connections=connections,
        interfaces=interfaces,
        mask=mask,
    )

    return ParsedStructure(data=data, info=info, covalents=[]) 

def parse_polymer_simple(
    polymer_chain,
    components: Dict[str, Mol],
    chain_id: str,
    entity_name: str,
) -> Optional[ParsedChain]:
    """Simplified polymer parsing for PDB files.
    
    Parameters
    ----------
    polymer_chain : gemmi.Chain
        The chain containing polymer residues
    components : Dict[str, Mol]
        Components dictionary
    chain_id : str
        Chain identifier
    entity_name : str
        Entity name
        
    Returns
    -------
    ParsedChain, optional
        The parsed polymer chain
    """
    try:
        # Get polymer residues
        polymer_residues = list(polymer_chain.get_polymer())
        if not polymer_residues:
            print(f"No polymer residues in chain {chain_id}")
            return None
            
        # Get sequence as residue names
        sequence = [res.name for res in polymer_residues]
        sequence_1letter = []
        
        # Process each residue
        parsed_residues = []
        
        for res_idx, residue in enumerate(polymer_residues):
            res_name = residue.name
            sequence_1letter.append(get_one_letter_code(res_name))
            
            # Standard amino acid or nucleotide
            if res_name in const.tokens:
                # Get ref molecule
                ref_mol = components.get(res_name)
                if ref_mol is None:
                    print(f"Missing component for standard residue {res_name}")
                    continue
                    
                ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
                try:
                    ref_conformer = get_conformer(ref_mol)
                except ValueError as e:
                    print(f"Error getting conformer for {res_name}: {e}")
                    continue
                
                # Get atom info
                ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
                ref_atoms = [ref_name_to_atom.get(a) for a in const.ref_atoms.get(res_name, [])]
                ref_atoms = [a for a in ref_atoms if a is not None]
                
                atoms = []
                for ref_atom in ref_atoms:
                    atom_name = ref_atom.GetProp("name")
                    idx = ref_atom.GetIdx()
                    
                    # Get coordinates from conformer
                    ref_coords = ref_conformer.GetAtomPosition(idx)
                    ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
                    
                    # Find corresponding atom in PDB
                    atom_is_present = False
                    coords = (0, 0, 0)
                    
                    for atom in residue:
                        if atom.name.strip() == atom_name.strip():
                            atom_is_present = True
                            coords = (atom.pos.x, atom.pos.y, atom.pos.z)
                            break
                    
                    # Add atom
                    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]
                    atoms.append(
                        ParsedAtom(
                            name=atom_name,
                            element=ref_atom.GetAtomicNum(),
                            charge=ref_atom.GetFormalCharge(),
                            coords=coords,
                            conformer=ref_coords,
                            is_present=atom_is_present,
                            chirality=const.chirality_type_ids.get(
                                ref_atom.GetChiralTag(), unk_chirality
                            ),
                        )
                    )
                
                # Add residue
                atom_center = const.res_to_center_atom_id.get(res_name, 0)
                atom_disto = const.res_to_disto_atom_id.get(res_name, 0)
                
                # Avoid index errors
                if atom_center >= len(atoms):
                    atom_center = 0
                if atom_disto >= len(atoms):
                    atom_disto = 0
                    
                parsed_residues.append(
                    ParsedResidue(
                        name=res_name,
                        type=const.token_ids.get(res_name, 0),
                        idx=res_idx,
                        atoms=atoms,
                        bonds=[],  # No bonds for standard residues
                        orig_idx=None,
                        atom_center=atom_center,
                        atom_disto=atom_disto,
                        is_standard=True,
                        is_present=True,
                    )
                )
            else:
                # Non-standard residue, try to parse as CCD
                residue = parse_ccd_residue(
                    name=res_name,
                    components=components,
                    res_idx=res_idx,
                    gemmi_mol=residue,
                    is_covalent=False,
                )
                if residue:
                    parsed_residues.append(residue)
        
        if not parsed_residues:
            print(f"No residues parsed for chain {chain_id}")
            return None
            
        # Determine chain type
        chain_type = const.chain_type_ids["PROTEIN"]  # Default to protein
        
        return ParsedChain(
            name=chain_id,
            entity=entity_name,
            residues=parsed_residues,
            type=chain_type,
            sequence=sequence_1letter,
        )
    
    except Exception as e:
        print(f"Error parsing polymer chain {chain_id}: {e}")
        import traceback
        traceback.print_exc()
        return None 

def get_one_letter_code(res_name: str) -> str:
    """Get one-letter code for an amino acid.
    
    Parameters
    ----------
    res_name : str
        Three-letter residue name
        
    Returns
    -------
    str
        One-letter code or 'X' if unknown
    """
    # Standard amino acids
    aa_map = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
        'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
        'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
        'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
        'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'MSE': 'M',  # Selenomethionine
    }
    
    return aa_map.get(res_name, 'X')


def is_ligand(residue: gemmi.Residue) -> bool:
    """Check if a residue is a ligand.
    
    Parameters
    ----------
    residue : gemmi.Residue
        The residue to check
        
    Returns
    -------
    bool
        True if the residue is a ligand
    """
    # Non-standard amino acids and small molecules are ligands
    standard_residues = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
        'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
        'THR', 'TRP', 'TYR', 'VAL', 'MSE',
        # Nucleic acids
        'A', 'C', 'G', 'T', 'U', 'DA', 'DC', 'DG', 'DT'
    }
    
    return residue.name not in standard_residues 

def prepare_rna_from_csv(
    csv_path: str,
    seq_csv_path: str,
    output_dir: str = None,
) -> Dict[str, str]:
    """Process Stanford RNA dataset entries and convert them to PDB format.
    
    Parameters
    ----------
    csv_path : str
        Path to the labels CSV file (e.g., train_labels.csv)
    seq_csv_path : str
        Path to the sequences CSV file (e.g., train_sequences.csv)
    output_dir : str, optional
        Directory to save generated PDB files, if None, uses a temporary directory
    
    Returns
    -------
    Dict[str, str]
        Dictionary mapping RNA IDs to their PDB file paths
    """
    import os
    import csv
    import tempfile
    from collections import defaultdict
    import numpy as np
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="rna_pdb_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing RNA dataset from {csv_path}")
    print(f"Output directory: {output_dir}")
    
    # Read sequence information
    rna_sequences = {}
    with open(seq_csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header (target_id,sequence,...)
        for row in reader:
            if len(row) >= 2:
                rna_id = row[0]
                sequence = row[1]
                rna_sequences[rna_id] = sequence
    
    print(f"Read {len(rna_sequences)} RNA sequences")
    
    # Group coordinates by RNA structure
    rna_coordinates = defaultdict(list)
    skipped_entries = 0
    
    # Read coordinates from the labels file
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header: ID,resname,resid,x_1,y_1,z_1
        
        # Check if header exists and get column indices
        id_col = header.index('ID') if 'ID' in header else 0
        resname_col = header.index('resname') if 'resname' in header else 1
        resid_col = header.index('resid') if 'resid' in header else 2
        x_col = header.index('x_1') if 'x_1' in header else 3
        y_col = header.index('y_1') if 'y_1' in header else 4
        z_col = header.index('z_1') if 'z_1' in header else 5
        
        for row in reader:
            if len(row) >= 6:
                # Format from CSV: ID,resname,resid,x_1,y_1,z_1
                # Example: 1SCL_A_3,G,3,5.529,-27.813,5.878
                entry_id = row[id_col]
                nucleotide = row[resname_col]
                
                # Skip entries with missing coordinates
                try:
                    residue_id = int(row[resid_col])
                    
                    # Handle empty coordinate values
                    x_str = row[x_col].strip()
                    y_str = row[y_col].strip()
                    z_str = row[z_col].strip()
                    
                    if x_str and y_str and z_str:
                        x = float(x_str)
                        y = float(y_str)
                        z = float(z_str)
                    else:
                        # Skip entries with missing coordinates
                        skipped_entries += 1
                        continue
                    
                    # Extract RNA ID (e.g., 1SCL_A from 1SCL_A_3)
                    parts = entry_id.rsplit('_', 1)
                    if len(parts) == 2:
                        rna_id = parts[0]
                        rna_coordinates[rna_id].append((residue_id, nucleotide, x, y, z))
                except (ValueError, IndexError):
                    skipped_entries += 1
                    continue
    
    print(f"Processed coordinates for {len(rna_coordinates)} RNA structures")
    print(f"Skipped {skipped_entries} entries with missing or invalid data")
    
    # Generate PDB files for each RNA structure
    pdb_paths = {}
    
    for rna_id, coords in rna_coordinates.items():
        # Skip RNA structures with no valid coordinates
        if not coords:
            continue
            
        # Sort by residue ID
        coords.sort(key=lambda x: x[0])
        
        # Get sequence
        sequence = rna_sequences.get(rna_id, "")
        
        # Create PDB file
        pdb_path = os.path.join(output_dir, f"{rna_id}.pdb")
        
        with open(pdb_path, 'w') as f:
            # Write PDB header
            f.write(f"HEADER    RNA STRUCTURE {rna_id}\n")
            f.write(f"TITLE     {rna_id} RNA STRUCTURE FROM STANFORD RNA DATASET\n")
            f.write(f"REMARK    400 COMPOUND  MOL_ID: 1\n")
            f.write(f"REMARK    400 COMPOUND  CHAIN: {rna_id.split('_')[-1]}\n")
            f.write(f"REMARK    400 COMPOUND  MOLECULE: RNA\n")
            f.write(f"DBREF  {rna_id.split('_')[-1]} RNA    1    {len(sequence)} UNK    UNK   UNK       UNK\n")
            
            # Write sequence as REMARK
            f.write(f"REMARK 999 SEQUENCE: {sequence}\n")
            
            # Define proper nucleotide names
            nucleotide_map = {
                'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'
            }
            
            # Write SEQRES record
            chain_id = rna_id.split("_")[-1]  # Extract chain ID (e.g., 'A' from '1SCL_A')
            f.write(f"SEQRES   1 {len(sequence)} {chain_id} ")
            for i, nt in enumerate(sequence):
                f.write(f"{nucleotide_map.get(nt, 'X')} ")
                if (i + 1) % 13 == 0 and i < len(sequence) - 1:
                    f.write(f"\nSEQRES   {(i+1)//13 + 1} {len(sequence)} {chain_id} ")
            f.write("\n")
            
            # Write atom records
            atom_index = 1
            
            for residue_id, nucleotide, x, y, z in coords:
                # Map nucleotide code to PDB residue name (keep as single character for RNA)
                residue_name = nucleotide_map.get(nucleotide, 'X')
                
                # Create a single atom entry for each nucleotide using the exact coordinates from CSV
                f.write(f"ATOM  {atom_index:5d}  P     {residue_name}   {chain_id}{residue_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n")
                atom_index += 1
            
            # Write TER record
            f.write(f"TER   {atom_index:5d}      {residue_name}   {chain_id}{residue_id:4d}\n")
            
            # Write END record
            f.write("END\n")
        
        pdb_paths[rna_id] = pdb_path
    
    print(f"Generated {len(pdb_paths)} PDB files")
    return pdb_paths

def parse_stanford_rna_structure(
    rna_id: str,
    components: Dict[str, Mol],
    csv_path: str = None, 
    seq_csv_path: str = None,
    temp_dir: str = None,
    use_assembly: bool = True,
) -> ParsedStructure:
    """Parse a Stanford RNA dataset structure by ID.
    
    Parameters
    ----------
    rna_id : str
        RNA structure ID (e.g., '1SCL_A')
    components: Dict[str, Mol]
        The preprocessed PDB components dictionary.
    csv_path: str, optional
        Path to the labels CSV file (defaults to "/ist-nas/users/bunditb/boltz/stanford-rna/train_labels.csv")
    seq_csv_path: str, optional
        Path to the sequences CSV file (defaults to "/ist-nas/users/bunditb/boltz/stanford-rna/train_sequences.csv")
    temp_dir: str, optional
        Directory to store temporary PDB files
    use_assembly: bool
        Whether to use the first assembly.
        
    Returns
    -------
    ParsedStructure
        The parsed RNA structure.
    """
    import os
    import csv
    import numpy as np
    from collections import defaultdict
    
    # Set default paths if not provided
    if csv_path is None:
        csv_path = "/ist-nas/users/bunditb/boltz/stanford-rna/train_labels.csv"
    if seq_csv_path is None:
        seq_csv_path = "/ist-nas/users/bunditb/boltz/stanford-rna/train_sequences.csv"
    
    # Generate PDB files
    pdb_paths = prepare_rna_from_csv(csv_path, seq_csv_path, temp_dir)
    
    # Check if the requested RNA ID exists
    if rna_id not in pdb_paths:
        msg = f"RNA structure {rna_id} not found in the dataset."
        raise ValueError(msg)
    
    try:
        # Try to parse the PDB file using the standard parser
        pdb_path = pdb_paths[rna_id]
        parsed_structure = parse_pdb(pdb_path, components, use_assembly)
        return parsed_structure
    except Exception as e:
        print(f"Standard PDB parsing failed: {e}")
        print("Falling back to direct RNA structure creation...")
        
        # Read sequence and coordinates directly from CSV files
        sequence = ""
        coordinates = []
        
        # Get sequence from sequences file
        with open(seq_csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                if len(row) >= 2 and row[0] == rna_id:
                    sequence = row[1]
                    break
        
        if not sequence:
            msg = f"RNA sequence for {rna_id} not found."
            raise ValueError(msg)
        
        # Get coordinates from labels file
        coords_by_resid = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            id_col = header.index('ID') if 'ID' in header else 0
            resname_col = header.index('resname') if 'resname' in header else 1
            resid_col = header.index('resid') if 'resid' in header else 2
            x_col = header.index('x_1') if 'x_1' in header else 3
            y_col = header.index('y_1') if 'y_1' in header else 4
            z_col = header.index('z_1') if 'z_1' in header else 5
            
            for row in reader:
                if len(row) >= 6:
                    entry_id = row[id_col]
                    parts = entry_id.rsplit('_', 1)
                    if len(parts) == 2 and parts[0] == rna_id:
                        try:
                            residue_id = int(row[resid_col])
                            nucleotide = row[resname_col]
                            
                            # Handle empty coordinate values
                            x_str = row[x_col].strip()
                            y_str = row[y_col].strip()
                            z_str = row[z_col].strip()
                            
                            if x_str and y_str and z_str:
                                x = float(x_str)
                                y = float(y_str)
                                z = float(z_str)
                                coords_by_resid[residue_id] = (nucleotide, (x, y, z))
                        except (ValueError, IndexError):
                            continue
        
        if not coords_by_resid:
            msg = f"No valid coordinates found for {rna_id}."
            raise ValueError(msg)
        
        # Create arrays for all structure components
        chain_id = rna_id.split("_")[-1]  # Extract chain ID (e.g., 'A' from '1SCL_A')
        
        # Sort residues by ID
        sorted_res_ids = sorted(coords_by_resid.keys())
        
        # Create atom data
        atom_data = []
        bond_data = []
        res_data = []
        
        atom_idx = 0
        for res_idx, res_id in enumerate(sorted_res_ids):
            nucleotide, coord = coords_by_resid[res_id]
            
            # Add residue
            res_name = nucleotide
            res_type = const.token_ids.get(nucleotide, 0)
            res_data.append((
                res_name,      # name
                res_type,      # type
                res_id,        # idx
                atom_idx,      # atom_offset
                1,             # atom_count (just P atom)
                atom_idx,      # atom_center (P atom)
                atom_idx,      # atom_disto (P atom)
                True,          # is_standard
                True           # is_present
            ))
            
            # Add atoms - only P atom with exact coordinates from CSV
            x, y, z = coord
            
            # P atom
            atom_data.append((
                convert_atom_name("P"),   # name
                15,                      # element (P)
                0,                       # charge
                (x, y, z),               # coords
                (x, y, z),               # conformer
                True,                    # is_present
                0                        # chirality
            ))
            
            # No bonds since we only have one atom per residue
            atom_idx += 1
        
        # Create the chain data
        chain_data = [(
            chain_id,                   # name
            const.chain_type_ids["RNA"], # type
            0,                          # entity_id
            0,                          # sym_id
            0,                          # asym_id
            0,                          # atom_offset
            len(atom_data),             # atom_count
            0,                          # res_offset
            len(res_data)               # res_count
        )]
        
        # Convert to NumPy arrays
        atoms = np.array(atom_data, dtype=Atom)
        bonds = np.array(bond_data, dtype=Bond)  # Empty since we have no bonds
        residues = np.array(res_data, dtype=Residue)
        chains = np.array(chain_data, dtype=Chain)
        connections = np.array([], dtype=Connection)
        interfaces = np.array([], dtype=Interface)
        mask = np.ones(len(chain_data), dtype=bool)
        
        # Create info
        info = StructureInfo(
            deposited="",
            revised="",
            released="",
            resolution=0.0,
            method="theoretical",
            num_chains=len(chains),
            num_interfaces=0,
        )
        
        # Create structure
        data = Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            connections=connections,
            interfaces=interfaces,
            mask=mask,
        )
        
        return ParsedStructure(data=data, info=info, covalents=[]) 