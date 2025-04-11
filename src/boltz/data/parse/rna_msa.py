import gzip
import os
from pathlib import Path
from typing import Optional, TextIO, List, Tuple, Dict, Any

import numpy as np
import torch

# Define RNA nucleotide constants
RNA_A = 0
RNA_U = 1
RNA_G = 2
RNA_C = 3

# Mapping from nucleotide to index
RNA_NUC_TO_INDEX = {
    'A': RNA_A,
    'U': RNA_U,
    'G': RNA_G,
    'C': RNA_C,
    'T': RNA_U,  # Handle T as U for transcribed sequences
    'N': -1,     # Unknown nucleotide
    '-': -1,     # Gap
    '.': -1,     # Gap
    'X': -1,     # Unknown
    ' ': -1,     # Space
    'a': RNA_A,  # Handle lowercase (insertion in some formats)
    'u': RNA_U,
    'g': RNA_G,
    'c': RNA_C,
    't': RNA_U,
}

# Mapping from index to nucleotide
RNA_INDEX_TO_NUC = {
    RNA_A: 'A',
    RNA_U: 'U',
    RNA_G: 'G',
    RNA_C: 'C'
}

class RNAMSA:
    """Class to represent an RNA Multiple Sequence Alignment.
    
    This class stores the MSA data including the sequences, deletions, 
    and provides methods to convert to various formats.
    """
    
    def __init__(self, 
                residues: np.ndarray, 
                deletions: List[Tuple[int, int]], 
                sequences: List[Tuple[int, int, int, int, int, int]]):
        """
        Initialize the RNA MSA object.
        
        Parameters
        ----------
        residues : np.ndarray
            Array of residue indices.
        deletions : List[Tuple[int, int]]
            List of (residue_index, deletion_count) pairs.
        sequences : List[Tuple[int, int, int, int, int, int]]
            List of (seq_idx, taxonomy_id, res_start, res_end, del_start, del_end) tuples.
        """
        self.residues = residues
        self.deletions = deletions
        self.sequences = sequences
    
    def get_sequence(self, idx: int) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Get the sequence and its deletions at the given index.
        
        Parameters
        ----------
        idx : int
            Index of the sequence to retrieve.
        
        Returns
        -------
        Tuple[np.ndarray, List[Tuple[int, int]]]
            The sequence residues and deletions.
        """
        if idx >= len(self.sequences):
            raise IndexError(f"Sequence index {idx} out of range")
        
        seq_info = self.sequences[idx]
        seq_residues = self.residues[seq_info[2]:seq_info[3]]
        seq_deletions = self.deletions[seq_info[4]:seq_info[5]]
        
        return seq_residues, seq_deletions
    
    def to_one_hot(self, max_seqs: Optional[int] = None, max_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        Convert the MSA to one-hot encoded tensors.
        
        Parameters
        ----------
        max_seqs : int, optional
            Maximum number of sequences to include.
        max_len : int, optional
            Maximum sequence length.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with one-hot encoded MSA data:
            - rna_msa: [num_seqs, seq_len, 4] one-hot encoded sequences
            - has_deletion: [num_seqs, seq_len] boolean mask for positions with deletions
            - deletion_value: [num_seqs, seq_len] count of deletions at each position
        """
        # Determine number of sequences and max length
        num_seqs = min(len(self.sequences), max_seqs) if max_seqs is not None else len(self.sequences)
        
        # Find maximum sequence length
        max_sequence_length = 0
        for i in range(min(num_seqs, len(self.sequences))):
            seq_info = self.sequences[i]
            seq_len = seq_info[3] - seq_info[2]
            max_sequence_length = max(max_sequence_length, seq_len)
        
        # Apply max_len constraint if provided
        if max_len is not None:
            max_sequence_length = min(max_sequence_length, max_len)
        
        # Initialize tensors
        rna_msa = torch.zeros((num_seqs, max_sequence_length, 4))
        has_deletion = torch.zeros((num_seqs, max_sequence_length), dtype=torch.bool)
        deletion_value = torch.zeros((num_seqs, max_sequence_length))
        
        # Fill tensors with data
        for i in range(num_seqs):
            if i >= len(self.sequences):
                break
                
            seq_residues, seq_deletions = self.get_sequence(i)
            seq_len = min(len(seq_residues), max_sequence_length)
            
            # Fill one-hot encoding
            for j in range(seq_len):
                res_idx = seq_residues[j]
                if 0 <= res_idx < 4:  # Valid RNA nucleotide
                    rna_msa[i, j, res_idx] = 1.0
            
            # Fill deletion information
            for res_idx, del_count in seq_deletions:
                if res_idx < max_sequence_length:
                    has_deletion[i, res_idx] = True
                    deletion_value[i, res_idx] = del_count
        
        return {
            'rna_msa': rna_msa,
            'has_deletion': has_deletion,
            'deletion_value': deletion_value
        }
    
    def to_strings(self, max_seqs: Optional[int] = None) -> List[str]:
        """
        Convert the MSA to a list of string sequences.
        
        Parameters
        ----------
        max_seqs : int, optional
            Maximum number of sequences to include.
        
        Returns
        -------
        List[str]
            List of string sequences.
        """
        num_seqs = min(len(self.sequences), max_seqs) if max_seqs is not None else len(self.sequences)
        sequences = []
        
        for i in range(num_seqs):
            seq_residues, seq_deletions = self.get_sequence(i)
            
            # Convert to string with proper handling of deletions
            seq_str = ""
            del_idx = 0
            
            for j, res_idx in enumerate(seq_residues):
                # Add nucleotide
                if 0 <= res_idx < 4:
                    seq_str += RNA_INDEX_TO_NUC[res_idx]
                else:
                    seq_str += "N"  # Unknown nucleotide
                
                # Add deletions
                while del_idx < len(seq_deletions) and seq_deletions[del_idx][0] == j:
                    seq_str += "-" * seq_deletions[del_idx][1]
                    del_idx += 1
            
            sequences.append(seq_str)
        
        return sequences
    
    def __len__(self) -> int:
        """Get the number of sequences in the MSA."""
        return len(self.sequences)


def _parse_fasta_msa(
    lines: TextIO,
    max_seqs: Optional[int] = None,
    handle_lowercase: bool = True
) -> RNAMSA:
    """
    Process an RNA MSA file in FASTA format.
    
    Parameters
    ----------
    lines : TextIO
        The lines of the MSA file.
    max_seqs : int, optional
        The maximum number of sequences to include.
    handle_lowercase : bool
        Whether to handle lowercase letters as insertions (not counted in MSA columns).
    
    Returns
    -------
    RNAMSA
        The RNA MSA object.
    """
    visited = set()
    sequences = []
    deletions = []
    residues = []
    
    seq_idx = 0
    current_seq = ""
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        if line.startswith(">"):
            # Process previous sequence if exists
            if current_seq:
                # Skip if duplicate sequence
                str_seq = current_seq.replace("-", "").upper()
                if str_seq not in visited:
                    visited.add(str_seq)
                    
                    # Process sequence
                    residue = []
                    deletion = []
                    count = 0
                    res_idx = 0
                    
                    for c in current_seq:
                        # Handle lowercase letters as insertions
                        if handle_lowercase and c.islower() and c != '-':
                            count += 1
                            continue
                        
                        # Convert to token
                        token = RNA_NUC_TO_INDEX.get(c, -1)
                        residue.append(token)
                        
                        if count > 0:
                            deletion.append((res_idx, count))
                            count = 0
                        res_idx += 1
                    
                    # Record remaining deletions at the end if any
                    if count > 0:
                        deletion.append((res_idx - 1, count))
                    
                    res_start = len(residues)
                    res_end = res_start + len(residue)
                    
                    del_start = len(deletions)
                    del_end = del_start + len(deletion)
                    
                    # Use -1 as placeholder for taxonomy ID (not used for RNA)
                    sequences.append((seq_idx, -1, res_start, res_end, del_start, del_end))
                    residues.extend(residue)
                    deletions.extend(deletion)
                    
                    seq_idx += 1
                    if max_seqs is not None and seq_idx >= max_seqs:
                        break
            
            # Reset for new sequence
            current_seq = ""
        else:
            current_seq += line
    
    # Process the last sequence
    if current_seq and (max_seqs is None or seq_idx < max_seqs):
        # Skip if duplicate sequence
        str_seq = current_seq.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
            
            # Process sequence
            residue = []
            deletion = []
            count = 0
            res_idx = 0
            
            for c in current_seq:
                # Handle lowercase letters as insertions
                if handle_lowercase and c.islower() and c != '-':
                    count += 1
                    continue
                
                # Convert to token
                token = RNA_NUC_TO_INDEX.get(c, -1)
                residue.append(token)
                
                if count > 0:
                    deletion.append((res_idx, count))
                    count = 0
                res_idx += 1
            
            # Record remaining deletions at the end if any
            if count > 0:
                deletion.append((res_idx - 1, count))
            
            res_start = len(residues)
            res_end = res_start + len(residue)
            
            del_start = len(deletions)
            del_end = del_start + len(deletion)
            
            # Use -1 as placeholder for taxonomy ID (not used for RNA)
            sequences.append((seq_idx, -1, res_start, res_end, del_start, del_end))
            residues.extend(residue)
            deletions.extend(deletion)
    
    # Create RNA MSA object
    msa = RNAMSA(
        residues=np.array(residues, dtype=np.int32),
        deletions=deletions,
        sequences=sequences,
    )
    return msa


def parse_rna_msa(
    path: str,
    max_seqs: Optional[int] = None,
    handle_lowercase: bool = True
) -> RNAMSA:
    """
    Parse an RNA MSA file in FASTA format.
    
    Parameters
    ----------
    path : str
        Path to the MSA file.
    max_seqs : int, optional
        Maximum number of sequences to include.
    handle_lowercase : bool
        Whether to handle lowercase letters as insertions (not counted in MSA columns).
    
    Returns
    -------
    RNAMSA
        The RNA MSA object.
    """
    path_obj = Path(path)
    
    # Handle gzipped files
    if path_obj.suffix == ".gz":
        with gzip.open(str(path_obj), "rt") as f:
            msa = _parse_fasta_msa(f, max_seqs, handle_lowercase)
    else:
        with open(path, "r") as f:
            msa = _parse_fasta_msa(f, max_seqs, handle_lowercase)
    
    return msa


def extract_secondary_structure(
    sequences: List[str],
    use_vienna_rna: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract secondary structure features from RNA sequences.
    
    Parameters
    ----------
    sequences : List[str]
        List of RNA sequences.
    use_vienna_rna : bool
        Whether to use ViennaRNA for structure prediction.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with secondary structure features:
        - stem_prob: [num_seqs, seq_len] probability of being in a stem
        - loop_prob: [num_seqs, seq_len] probability of being in a loop
        - bulge_prob: [num_seqs, seq_len] probability of being in a bulge
    """
    num_seqs = len(sequences)
    if num_seqs == 0:
        return {}
    
    max_len = max(len(seq) for seq in sequences)
    
    # Initialize arrays
    stem_prob = np.zeros((num_seqs, max_len))
    loop_prob = np.zeros((num_seqs, max_len))
    bulge_prob = np.zeros((num_seqs, max_len))
    
    # Process each sequence
    for i, seq in enumerate(sequences):
        # Replace non-standard nucleotides
        clean_seq = ''.join(c if c in 'AUGC' else 'N' for c in seq.upper())
        
        if use_vienna_rna:
            try:
                import RNA
                # Predict secondary structure using ViennaRNA
                fc = RNA.fold_compound(clean_seq)
                structure, _ = fc.mfe()
                
                # Parse the dot-bracket notation
                s_prob, l_prob, b_prob = parse_dot_bracket(structure)
                
                # Copy to output arrays
                seq_len = len(seq)
                stem_prob[i, :seq_len] = s_prob
                loop_prob[i, :seq_len] = l_prob
                bulge_prob[i, :seq_len] = b_prob
                
            except ImportError:
                print("ViennaRNA package not available, using simple heuristic.")
                use_vienna_rna = False
        
        if not use_vienna_rna:
            # Simple heuristic based on base-pairing potential
            seq_len = len(seq)
            for j in range(seq_len):
                # Simple heuristic: G-C rich regions more likely to form stems
                if seq[j].upper() in 'GC':
                    stem_prob[i, j] = 0.7
                    loop_prob[i, j] = 0.2
                    bulge_prob[i, j] = 0.1
                else:
                    stem_prob[i, j] = 0.3
                    loop_prob[i, j] = 0.6
                    bulge_prob[i, j] = 0.1
    
    return {
        'stem_prob': stem_prob,
        'loop_prob': loop_prob,
        'bulge_prob': bulge_prob
    }


def parse_dot_bracket(dot_bracket: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse dot-bracket notation to get stem, loop, and bulge probabilities.
    
    Parameters
    ----------
    dot_bracket : str
        RNA secondary structure in dot-bracket notation
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Probabilities of stem, loop, and bulge for each position
    """
    n = len(dot_bracket)
    stem_prob = np.zeros(n)
    loop_prob = np.zeros(n)
    bulge_prob = np.zeros(n)
    
    # Stack to track opening parentheses
    stack = []
    pair_dict = {}
    
    # First pass: identify paired positions
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if stack:
                j = stack.pop()
                pair_dict[j] = i
                pair_dict[i] = j
    
    # Second pass: assign probabilities
    for i, char in enumerate(dot_bracket):
        if char == '(':
            stem_prob[i] = 1.0
        elif char == ')':
            stem_prob[i] = 1.0
        else:  # '.'
            # Check if it's a loop (enclosed by paired bases) or likely a bulge
            is_loop = False
            
            # Simple heuristic: if flanked by paired bases within a short distance, likely a loop
            left_paired = False
            right_paired = False
            
            # Check up to 4 positions to the left
            for j in range(max(0, i-4), i):
                if j in pair_dict:
                    left_paired = True
                    break
            
            # Check up to 4 positions to the right
            for j in range(i+1, min(i+5, n)):
                if j in pair_dict:
                    right_paired = True
                    break
            
            if left_paired and right_paired:
                # Inside a loop
                loop_prob[i] = 0.8
                bulge_prob[i] = 0.2
            else:
                # More likely a bulge or external loop
                loop_prob[i] = 0.3
                bulge_prob[i] = 0.7
    
    return stem_prob, loop_prob, bulge_prob


def load_rna_msa_data(
    msa_file: str,
    max_seqs: int = 4,
    max_len: Optional[int] = None,
    predict_structure: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Load RNA MSA data from a file.
    
    Parameters
    ----------
    msa_file : str
        Path to the MSA file.
    max_seqs : int
        Maximum number of sequences to include.
    max_len : int, optional
        Maximum sequence length.
    predict_structure : bool
        Whether to predict secondary structure.
    
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary with MSA features:
        - rna_msa: [max_seqs, seq_len, 4] one-hot encoded sequences
        - has_deletion: [max_seqs, seq_len] mask for positions with deletions
        - deletion_value: [max_seqs, seq_len] deletion values
        - msa_mask: [max_seqs, seq_len] mask for valid positions
        - query_mask: [seq_len] mask for query sequence
        - stem_prob: [max_seqs, seq_len] probability of being in a stem
        - loop_prob: [max_seqs, seq_len] probability of being in a loop
        - bulge_prob: [max_seqs, seq_len] probability of being in a bulge
    """
    # Check if MSA file exists
    if not os.path.exists(msa_file):
        print(f"MSA file {msa_file} does not exist")
        return {}
    
    # Parse MSA file
    msa = parse_rna_msa(msa_file, max_seqs=max_seqs)
    
    if len(msa) == 0:
        print(f"No sequences found in MSA file {msa_file}")
        return {}
    
    # Convert to tensors
    msa_tensors = msa.to_one_hot(max_seqs=max_seqs, max_len=max_len)
    
    # Create masks
    rna_msa = msa_tensors['rna_msa']
    num_seqs, seq_len, _ = rna_msa.shape
    
    # MSA mask: 1 where sum of one-hot is > 0 (valid nucleotide)
    msa_mask = (rna_msa.sum(dim=2) > 0).float()
    
    # Query mask: 1 for all positions in the query sequence (first sequence)
    query_mask = msa_mask[0].clone()
    
    # Get string sequences for structure prediction
    if predict_structure:
        string_seqs = msa.to_strings(max_seqs=max_seqs)
        structure_features = extract_secondary_structure(string_seqs)
        
        # Convert to tensors
        stem_prob = torch.tensor(structure_features['stem_prob'][:num_seqs, :seq_len])
        loop_prob = torch.tensor(structure_features['loop_prob'][:num_seqs, :seq_len])
        bulge_prob = torch.tensor(structure_features['bulge_prob'][:num_seqs, :seq_len])
        
        # Add to output
        msa_tensors['stem_prob'] = stem_prob
        msa_tensors['loop_prob'] = loop_prob
        msa_tensors['bulge_prob'] = bulge_prob
    
    # Add masks to output
    msa_tensors['msa_mask'] = msa_mask
    msa_tensors['query_mask'] = query_mask
    
    return msa_tensors


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    One-hot encode an RNA sequence.
    
    Parameters
    ----------
    sequence : str
        RNA sequence as string
        
    Returns
    -------
    np.ndarray
        One-hot encoded RNA sequence of shape [seq_len, 4]
    """
    seq_len = len(sequence)
    one_hot = np.zeros((seq_len, 4))
    
    for i, nuc in enumerate(sequence):
        idx = RNA_NUC_TO_INDEX.get(nuc, -1)
        if idx >= 0:
            one_hot[i, idx] = 1.0
            
    return one_hot 