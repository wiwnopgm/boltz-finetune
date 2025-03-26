#!/usr/bin/env python
"""
Script to test the RNA MSA Module with real data.

This script loads real RNA sequence data from:
1. CSV file with RNA sequence information at /ist-nas/users/bunditb/boltz/stanford-rna/train_sequences.csv
2. MSA files from /ist-nas/users/bunditb/boltz/stanford-rna/MSA

It demonstrates:
1. How to prepare input data for RNA MSA Module from real RNA data
2. How to initialize and run the module
3. The expected output format
"""

import torch
import numpy as np
import pandas as pd
import os
import RNA  # Import ViennaRNA package
from typing import Dict, List, Tuple
from collections import defaultdict

from boltz.data import const
from boltz.model.modules.rna_trunk import (
    RNAInputEmbedder,
    RNAMSAModule,
    RNADistogramModule
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define RNA-specific constants
if not hasattr(const, "rna_num_tokens"):
    setattr(const, "rna_num_tokens", 4)  # A, U, G, C

# RNA nucleotide indices for one-hot encoding
RNA_A = 0
RNA_U = 1
RNA_G = 2
RNA_C = 3

# Mapping from nucleotide to one-hot index
NUC_TO_INDEX = {
    'A': RNA_A,
    'U': RNA_U,
    'G': RNA_G,
    'C': RNA_C,
    # Handle additional cases for sequences that might have non-standard nucleotides
    'N': -1,  # Unknown nucleotide
    '-': -1,  # Gap
    '.': -1,  # Gap
}

# Mapping from one-hot index to nucleotide
INDEX_TO_NUC = {
    RNA_A: 'A',
    RNA_U: 'U',
    RNA_G: 'G',
    RNA_C: 'C'
}

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
        idx = NUC_TO_INDEX.get(nuc, -1)
        if idx >= 0:
            one_hot[i, idx] = 1.0
            
    return one_hot

def parse_dot_bracket(dot_bracket: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse dot-bracket notation to get stem, loop, and bulge probabilities.
    
    Parameters
    ----------
    dot_bracket : str
        RNA secondary structure in dot-bracket notation
        
    Returns
    -------
    tuple
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

def read_msa_file(filepath: str) -> List[str]:
    """
    Read an MSA file in FASTA format.
    
    Parameters
    ----------
    filepath : str
        Path to the MSA file
        
    Returns
    -------
    List[str]
        List of sequences from the MSA file
    """
    sequences = []
    current_seq = ""
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                    current_seq = ""
                else:
                    current_seq += line
                    
            if current_seq:
                sequences.append(current_seq)
    except Exception as e:
        print(f"Error reading MSA file {filepath}: {e}")
        return []
        
    return sequences

def load_real_rna_data(csv_path: str, msa_dir: str, batch_size: int = 2, max_seq_len: int = 100, max_num_seqs: int = 4) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Load real RNA data from CSV file and MSA directory.
    
    Parameters
    ----------
    csv_path : str
        Path to the CSV file with RNA data
    msa_dir : str
        Path to the directory with MSA files
    batch_size : int
        Number of sequences to include in the batch
    max_seq_len : int
        Maximum sequence length to use (will truncate longer sequences)
    max_num_seqs : int
        Maximum number of sequences to use from each MSA file
        
    Returns
    -------
    Tuple[Dict[str, torch.Tensor], torch.Tensor]
        Dictionary of input features and initial pairwise embeddings
    """
    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} RNA sequences from {csv_path}")
    
    # Get unique RNA chains
    pdb_chains = df['target_id'].tolist()
    print(f"Found {len(pdb_chains)} unique RNA chains")
    
    # Select a subset of chains for the batch
    batch_chains = pdb_chains[:batch_size]
    print(f"Using PDB chains: {batch_chains}")
    
    # Initialize lists to store data for each chain
    sequences = []
    msa_data = []
    
    # Process each chain in the batch
    for pdb_chain in batch_chains:
        # Get sequence data for this chain
        chain_data = df[df['target_id'] == pdb_chain]
        
        # Extract sequence
        seq = chain_data['sequence'].values[0]
        seq = seq[:max_seq_len]  # Truncate to max_seq_len
        sequences.append(seq)
        
        print(f"Chain {pdb_chain} sequence: {seq}")
        
        # Get MSA file for this chain
        msa_file = os.path.join(msa_dir, f"{pdb_chain}.MSA.fasta")
        print(f"Looking for MSA file: {msa_file}")
        
        # Read MSA file if it exists
        if os.path.exists(msa_file):
            msa_sequences = read_msa_file(msa_file)
            print(f"Found MSA file with {len(msa_sequences)} sequences")
            # Truncate to max_seq_len and max_num_seqs
            msa_sequences = [s[:max_seq_len] for s in msa_sequences[:max_num_seqs]]
            # If MSA has fewer sequences than max_num_seqs, repeat the query sequence
            while len(msa_sequences) < max_num_seqs:
                msa_sequences.append(seq[:max_seq_len])
        else:
            print(f"MSA file not found: {msa_file}, using replicated query sequence")
            # Use the query sequence repeated max_num_seqs times
            msa_sequences = [seq[:max_seq_len]] * max_num_seqs
        
        msa_data.append(msa_sequences)
    
    # Find the maximum sequence length in the batch
    max_len = max([len(s) for s in sequences])
    print(f"Maximum sequence length in batch: {max_len}")
    
    # Create one-hot encoded nucleotide sequences
    # Shape: [batch_size, seq_len, 4] for A, U, G, C
    nuc_type = torch.zeros((batch_size, max_len, const.rna_num_tokens), device=device)
    
    # Create MSA data - RNA multiple sequence alignment
    # Shape: [batch_size, num_seqs, seq_len, 4]
    rna_msa = torch.zeros((batch_size, max_num_seqs, max_len, const.rna_num_tokens), device=device)
    
    # Secondary structure features
    # Shape: [batch_size, num_seqs, seq_len]
    stem_prob = torch.zeros((batch_size, max_num_seqs, max_len), device=device)
    loop_prob = torch.zeros((batch_size, max_num_seqs, max_len), device=device)
    bulge_prob = torch.zeros((batch_size, max_num_seqs, max_len), device=device)
    
    # MSA paired positions
    # Shape: [batch_size, num_seqs, seq_len]
    msa_paired = torch.zeros((batch_size, max_num_seqs, max_len), device=device)
    
    # Process each sequence in the batch
    for b, (seq, msa_seqs) in enumerate(zip(sequences, msa_data)):
        # One-hot encode the main sequence
        one_hot = one_hot_encode_sequence(seq)
        nuc_type[b, :len(seq), :] = torch.tensor(one_hot, device=device)
        
        # One-hot encode MSA sequences
        for s, msa_seq in enumerate(msa_seqs):
            msa_one_hot = one_hot_encode_sequence(msa_seq)
            rna_msa[b, s, :len(msa_seq), :] = torch.tensor(msa_one_hot, device=device)
            
            # Use ViennaRNA to predict structure for each MSA sequence
            fc = RNA.fold_compound(msa_seq)
            structure, _ = fc.mfe()
            
            print(f"Batch {b}, Seq {s}: {msa_seq}")
            print(f"Structure: {structure}")
            
            # Parse dot-bracket notation to get probabilities
            stem_p, loop_p, bulge_p = parse_dot_bracket(structure)
            
            # Set structure probabilities
            stem_prob[b, s, :len(msa_seq)] = torch.tensor(stem_p, device=device)
            loop_prob[b, s, :len(msa_seq)] = torch.tensor(loop_p, device=device)
            bulge_prob[b, s, :len(msa_seq)] = torch.tensor(bulge_p, device=device)
            
            # Set paired positions based on structure
            for i, char in enumerate(structure):
                if i >= max_len:
                    break
                if char in '()':
                    msa_paired[b, s, i] = 1.0
    
    # Create deletion info (for this example, we'll use zeros, as we don't have real deletion info)
    has_deletion = torch.zeros((batch_size, max_num_seqs, max_len), device=device)
    deletion_value = torch.zeros((batch_size, max_num_seqs, max_len), device=device)
    
    # Calculate mean deletion value (all zeros in this case)
    deletion_mean = torch.mean(deletion_value, dim=1)
    
    # Create arbitrary profile info (could be refined in a real implementation)
    profile_dim = 20
    profile = torch.zeros((batch_size, max_len, profile_dim), device=device)
    
    # Create masks
    msa_mask = torch.ones((batch_size, max_num_seqs, max_len), device=device)
    token_pad_mask = torch.ones((batch_size, max_len), device=device)
    
    # Create secondary structure features for input embedding
    sec_structure = torch.cat([
        stem_prob.mean(dim=1).unsqueeze(-1),
        loop_prob.mean(dim=1).unsqueeze(-1),
        bulge_prob.mean(dim=1).unsqueeze(-1)
    ], dim=-1)
    
    # Create placeholder for pocket feature
    pocket_feature = torch.zeros((batch_size, max_len, 1), device=device)
    
    # Create initial pairwise embeddings
    token_z = 128  # Example dimension
    z = torch.randn((batch_size, max_len, max_len, token_z), device=device) * 0.01
    
    # Package all features
    feats = {
        # Single sequence features
        "nuc_type": nuc_type,
        "profile": profile,
        "deletion_mean": deletion_mean,
        
        # MSA features
        "rna_msa": rna_msa,
        "has_deletion": has_deletion,
        "deletion_value": deletion_value,
        "msa_mask": msa_mask,
        
        # Pairing information
        "msa_paired": msa_paired,
        
        # Secondary structure
        "stem_prob": stem_prob,
        "loop_prob": loop_prob,
        "bulge_prob": bulge_prob,
        "sec_structure": sec_structure,
        
        # Masking
        "token_pad_mask": token_pad_mask,
        
        # Additional features
        "pocket_feature": pocket_feature,
    }
    
    return feats, z


def run_example():
    """Run an example of the RNA MSA Module with real data."""
    # Paths to data
    csv_path = "/ist-nas/users/bunditb/boltz/stanford-rna/train_sequences.csv"
    msa_dir = "/ist-nas/users/bunditb/boltz/stanford-rna/MSA"
    
    # Parameters
    batch_size = 2
    max_seq_len = 50  # Use smaller sequences for faster demonstration
    max_num_seqs = 4
    
    # Model parameters
    atom_s = 32
    atom_z = 32
    token_s = 32
    token_z = 128
    
    # Atoms per window parameters
    atoms_per_window_queries = 8
    atoms_per_window_keys = 8
    
    # Atom encoder parameters
    atom_feature_dim = 16
    atom_encoder_depth = 2
    atom_encoder_heads = 4
    
    # MSA parameters
    msa_s = 32
    s_input_dim = token_s + const.rna_num_tokens + 20 + 1 + 3 + 1  # Assuming a profile dim of 20
    msa_blocks = 2
    msa_dropout = 0.15
    z_dropout = 0.15
    
    # Load real RNA data
    print("Loading real RNA data...")
    feats, z = load_real_rna_data(csv_path, msa_dir, batch_size, max_seq_len, max_num_seqs)
    
    # Print detailed information about input features
    print("\n===== INPUT FEATURES =====")
    for key, value in feats.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            # Print some sample values for key tensors
            if key in ['nuc_type', 'stem_prob', 'loop_prob', 'bulge_prob']:
                print(f"  Sample values (first batch, first few positions): {value[0, :5]}")
    
    print(f"\nInitial pairwise embeddings (z): shape={z.shape}, dtype={z.dtype}")
    print(f"  Z stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")
    
    # Get device
    device = z.device
    print(f"Using device: {device}")
    
    # Initialize RNAInputEmbedder
    print("\n===== RNA INPUT EMBEDDER =====")
    input_embedder = RNAInputEmbedder(
        atom_s=atom_s,
        atom_z=atom_z,
        token_s=token_s,
        token_z=token_z,
        atoms_per_window_queries=atoms_per_window_queries,
        atoms_per_window_keys=atoms_per_window_keys,
        atom_feature_dim=atom_feature_dim,
        atom_encoder_depth=atom_encoder_depth,
        atom_encoder_heads=atom_encoder_heads,
        no_atom_encoder=True,  # Simplify for this example
        use_secondary_structure=True,
    ).to(device)
    print(f"RNAInputEmbedder parameters: {sum(p.numel() for p in input_embedder.parameters())}")
    
    # Forward pass through RNAInputEmbedder
    print("Running forward pass through RNAInputEmbedder...")
    s, base_pairing = input_embedder(feats)
    
    print(f"Embedded tokens (s): shape={s.shape}, dtype={s.dtype}")
    print(f"  S stats: min={s.min().item():.4f}, max={s.max().item():.4f}, mean={s.mean().item():.4f}, std={s.std().item():.4f}")
    print(f"Base-pairing constraints: shape={base_pairing.shape}, dtype={base_pairing.dtype}")
    print(f"  Base-pairing stats: min={base_pairing.min().item():.4f}, max={base_pairing.max().item():.4f}, mean={base_pairing.mean().item():.4f}")
    
    # Visualize base-pairing for the first sequence
    if base_pairing.shape[1] <= 50:  # Only print if not too large
        print("\nBase-pairing matrix for first sequence (first 10x10):")
        for i in range(min(10, base_pairing.shape[1])):
            row_str = ' '.join(f'{val:.1f}' for val in base_pairing[0, i, :min(10, base_pairing.shape[2])].cpu().numpy())
            print(f"  {row_str}")
    
    # Initialize RNAMSAModule
    print("\n===== RNA MSA MODULE =====")
    msa_module = RNAMSAModule(
        msa_s=msa_s,
        token_z=token_z,
        s_input_dim=s_input_dim,
        msa_blocks=msa_blocks,
        msa_dropout=msa_dropout,
        z_dropout=z_dropout,
        use_paired_feature=True,
        use_secondary_structure=True,
        use_watson_crick_constraints=True,
    ).to(device)
    print(f"RNAMSAModule parameters: {sum(p.numel() for p in msa_module.parameters())}")
    
    # Forward pass through RNAMSAModule
    print("Running forward pass through RNAMSAModule...")
    z_out = msa_module(z, s, feats, base_pairing)
    print(f"Output pairwise embeddings (z_out): shape={z_out.shape}, dtype={z_out.dtype}")
    print(f"  Z_out stats: min={z_out.min().item():.4f}, max={z_out.max().item():.4f}, mean={z_out.mean().item():.4f}, std={z_out.std().item():.4f}")
    
    # Visualize the changes in the pairwise representation
    print("\nPairwise representation changes:")
    print(f"  Initial z l2 norm (mean): {torch.norm(z, dim=-1).mean().item():.4f}")
    print(f"  Final z_out l2 norm (mean): {torch.norm(z_out, dim=-1).mean().item():.4f}")
    
    # Compute the difference between input and output pairwise embeddings
    z_diff = z_out - z
    print(f"  Difference l2 norm (mean): {torch.norm(z_diff, dim=-1).mean().item():.4f}")
    
    # Construct structure info for distogram prediction
    structure_info = {
        'stem_prob': feats['stem_prob'].mean(dim=1),
        'loop_prob': feats['loop_prob'].mean(dim=1),
        'bulge_prob': feats['bulge_prob'].mean(dim=1),
        'base_pairing': base_pairing
    }
    
    print("\n===== STRUCTURAL INFORMATION FOR DISTOGRAM =====")
    for key, value in structure_info.items():
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        print(f"  {key} stats: min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
    
    # Initialize RNADistogramModule
    print("\n===== RNA DISTOGRAM MODULE =====")
    distogram_module = RNADistogramModule(
        token_z=token_z,
        num_bins=36,  # Example number of distance bins
    ).to(device)
    print(f"RNADistogramModule parameters: {sum(p.numel() for p in distogram_module.parameters())}")
    
    # Forward pass through RNADistogramModule
    print("Running forward pass through RNADistogramModule...")
    distogram = distogram_module(z_out, structure_info)
    print(f"Output distogram: shape={distogram.shape}, dtype={distogram.dtype}")
    print(f"  Distogram stats: min={distogram.min().item():.4f}, max={distogram.max().item():.4f}, mean={distogram.mean().item():.4f}")
    
    # Visualize the first few distance bins distribution
    distogram_softmax = torch.softmax(distogram, dim=-1)
    bin_means = distogram_softmax.mean(dim=[0, 1, 2])
    print("\nDistance bin distribution (average probability across all positions):")
    bins_to_show = min(10, bin_means.shape[0])
    bin_str = ' '.join(f'bin{i}:{bin_means[i].item():.3f}' for i in range(bins_to_show))
    print(f"  {bin_str}...")
    
    # Find the most likely distance bin for each position pair
    most_likely_bins = torch.argmax(distogram, dim=-1)
    bin_counts = torch.bincount(most_likely_bins.flatten(), minlength=distogram.shape[-1])
    bin_percentages = 100 * bin_counts.float() / bin_counts.sum()
    
    print("\nMost likely distance bin distribution (percentage of position pairs):")
    bins_to_show = min(10, bin_percentages.shape[0])
    bin_str = ' '.join(f'bin{i}:{bin_percentages[i].item():.1f}%' for i in range(bins_to_show))
    print(f"  {bin_str}...")
    
    print("\n===== EXAMPLE SUMMARY =====")
    print("Input format requirements:")
    print("  - nuc_type: One-hot encoded nucleotides [batch, seq_len, 4]")
    print("  - rna_msa: Multiple sequence alignment [batch, num_seqs, seq_len, 4]")
    print("  - has_deletion: Deletion indicator [batch, num_seqs, seq_len]")
    print("  - deletion_value: Deletion values [batch, num_seqs, seq_len]")
    print("  - msa_paired: Base-pairing indicator [batch, num_seqs, seq_len]")
    print("  - Secondary structure features (stem_prob, loop_prob, bulge_prob): [batch, num_seqs, seq_len]")
    print("  - Various masks for padding and attention")
    print("\nOutput formats:")
    print("  - Pairwise embeddings (z_out): [batch, seq_len, seq_len, token_z]")
    print("  - Distogram: [batch, seq_len, seq_len, num_bins]")
    print("\nRNA-specific constraints incorporated:")
    print("  - Base-pairing from ViennaRNA structure prediction")
    print("  - Secondary structure elements (stems, loops, bulges)")
    print("  - Watson-Crick and wobble base-pairing rules")
    
    # Also print info about the sequences we processed
    print("\nSequences processed:")
    df = pd.read_csv(csv_path)
    selected_chains = df['target_id'].tolist()[:batch_size]
    selected_sequences = df['sequence'].tolist()[:batch_size]
    for i, (chain, seq) in enumerate(zip(selected_chains, selected_sequences)):
        print(f"  {i+1}. {chain}: {seq[:30]}{'...' if len(seq) > 30 else ''}")
        # Find the sequence in our processed data
        if i < len(feats['stem_prob']):
            avg_stem = feats['stem_prob'][i].mean(dim=0).mean().item()
            avg_loop = feats['loop_prob'][i].mean(dim=0).mean().item()
            avg_bulge = feats['bulge_prob'][i].mean(dim=0).mean().item()
            print(f"     Secondary structure stats: avg stem prob={avg_stem:.2f}, avg loop prob={avg_loop:.2f}, avg bulge prob={avg_bulge:.2f}")
    
    return {
        'input_features': feats,
        'input_pairwise': z,
        'embedded_tokens': s,
        'base_pairing': base_pairing,
        'output_pairwise': z_out,
        'distogram': distogram,
        'structure_info': structure_info
    }


if __name__ == "__main__":
    results = run_example()
    print(results)
    print("\nSuccessfully ran RNA MSA module with real data!") 