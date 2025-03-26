#!/usr/bin/env python
"""
Example script to test the RNA MSA Module.

This script demonstrates:
1. How to prepare input data for RNA MSA Module
2. How to initialize and run the module
3. The expected output format

The RNA MSA Module expects specific input formats related to RNA sequences
and their multiple sequence alignments, with particular attention to
RNA-specific features like base-pairing and secondary structure.
"""

import torch
import numpy as np
from typing import Dict
import RNA  # Import ViennaRNA package

from boltz.data import const
from boltz.model.modules.rna_trunk import (
    RNAInputEmbedder,
    RNAMSAModule,
    RNADistogramModule
)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define RNA-specific constants if not already defined
if not hasattr(const, "rna_num_tokens"):
    setattr(const, "rna_num_tokens", 4)  # A, U, G, C

# RNA nucleotide indices for one-hot encoding
RNA_A = 0
RNA_U = 1
RNA_G = 2
RNA_C = 3

# Mapping from one-hot index to nucleotide
INDEX_TO_NUC = {
    RNA_A: 'A',
    RNA_U: 'U',
    RNA_G: 'G',
    RNA_C: 'C'
}

def one_hot_to_sequence(one_hot):
    """
    Convert one-hot encoded RNA sequence to string sequence.
    
    Parameters
    ----------
    one_hot : numpy.ndarray
        One-hot encoded RNA sequence
        
    Returns
    -------
    str
        RNA sequence as string
    """
    indices = np.argmax(one_hot, axis=-1)
    return ''.join([INDEX_TO_NUC[idx] for idx in indices])

def parse_dot_bracket(dot_bracket):
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

def create_sample_rna_data(batch_size=2, seq_len=16, num_seqs=4):
    """
    Create sample RNA data for testing the RNA MSA Module.
    
    Parameters
    ----------
    batch_size : int
        Batch size
    seq_len : int
        Sequence length
    num_seqs : int
        Number of sequences in MSA
        
    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary of input features
    """
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create one-hot encoded nucleotide sequences
    # Shape: [batch_size, seq_len, 4] for A, U, G, C
    nuc_type = torch.zeros((batch_size, seq_len, const.rna_num_tokens), device=device)
    
    # Randomly assign nucleotides (one-hot encoding)
    for b in range(batch_size):
        for i in range(seq_len):
            nucleotide = np.random.randint(0, const.rna_num_tokens)
            nuc_type[b, i, nucleotide] = 1.0
    
    # Create MSA data - RNA multiple sequence alignment
    # Shape: [batch_size, num_seqs, seq_len, 4]
    rna_msa = torch.zeros((batch_size, num_seqs, seq_len, const.rna_num_tokens), device=device)
    
    # Fill with random nucleotides, with some resemblance to the original sequence
    # First sequence in MSA is the original sequence
    rna_msa[:, 0, :, :] = nuc_type
    
    # Create mutated versions for other sequences in MSA
    for b in range(batch_size):
        for s in range(1, num_seqs):
            for i in range(seq_len):
                # 80% chance to keep the same nucleotide, 20% chance to mutate
                if np.random.random() < 0.8:
                    rna_msa[b, s, i, :] = nuc_type[b, i, :]
                else:
                    mutated = np.random.randint(0, const.rna_num_tokens)
                    rna_msa[b, s, i, mutated] = 1.0
    
    # Create evolutionary profile
    # Shape: [batch_size, seq_len, profile_dim]
    profile_dim = 20  # Arbitrary dimension for this example
    profile = torch.rand((batch_size, seq_len, profile_dim), device=device)
    
    # Create deletion information
    # Shape: [batch_size, num_seqs, seq_len]
    has_deletion = torch.zeros((batch_size, num_seqs, seq_len), device=device)
    deletion_value = torch.zeros((batch_size, num_seqs, seq_len), device=device)
    
    # Random deletions
    for b in range(batch_size):
        for s in range(num_seqs):
            for i in range(seq_len):
                if np.random.random() < 0.05:  # 5% chance of deletion
                    has_deletion[b, s, i] = 1.0
                    deletion_value[b, s, i] = np.random.random()
    
    # Calculate mean deletion value
    # Shape: [batch_size, seq_len]
    deletion_mean = torch.mean(deletion_value, dim=1)
    
    # Create masks
    # MSA mask: [batch_size, num_seqs, seq_len]
    msa_mask = torch.ones((batch_size, num_seqs, seq_len), device=device)
    
    # Token padding mask: [batch_size, seq_len]
    token_pad_mask = torch.ones((batch_size, seq_len), device=device)
    
    # RNA-specific: paired positions in MSA
    # Shape: [batch_size, num_seqs, seq_len]
    msa_paired = torch.zeros((batch_size, num_seqs, seq_len), device=device)
    
    # Secondary structure features
    # Shape: [batch_size, num_seqs, seq_len]
    stem_prob = torch.zeros((batch_size, num_seqs, seq_len), device=device)
    loop_prob = torch.zeros((batch_size, num_seqs, seq_len), device=device)
    bulge_prob = torch.zeros((batch_size, num_seqs, seq_len), device=device)
    
    # Generate realistic RNA secondary structure using ViennaRNA
    for b in range(batch_size):
        for s in range(num_seqs):
            # Convert one-hot to sequence
            one_hot_cpu = rna_msa[b, s].cpu().numpy()
            seq = one_hot_to_sequence(one_hot_cpu)
            
            # Use ViennaRNA to predict structure
            fc = RNA.fold_compound(seq)
            structure, _ = fc.mfe()
            
            # Also get base pair probabilities
            fc.pf()
            bpp = fc.bpp()
            
            print(f"Batch {b}, Seq {s}: {seq}")
            print(f"Structure: {structure}")
            
            # Parse dot-bracket notation to get probabilities
            stem_p, loop_p, bulge_p = parse_dot_bracket(structure)
            
            # Set paired positions based on structure
            for i in range(seq_len):
                # Mark paired positions from structure
                if structure[i] in '()':
                    msa_paired[b, s, i] = 1.0
                
                # Set structure probabilities
                stem_prob[b, s, i] = torch.tensor(stem_p[i], device=device)
                loop_prob[b, s, i] = torch.tensor(loop_p[i], device=device)
                bulge_prob[b, s, i] = torch.tensor(bulge_p[i], device=device)
    
    # Additional features for input embedding
    sec_structure = torch.cat([
        stem_prob.mean(dim=1).unsqueeze(-1),
        loop_prob.mean(dim=1).unsqueeze(-1),
        bulge_prob.mean(dim=1).unsqueeze(-1)
    ], dim=-1)
    
    # Placeholder for additional pocket feature
    pocket_feature = torch.zeros((batch_size, seq_len, 1), device=device)
    
    # Create pairwise embeddings
    # Shape: [batch_size, seq_len, seq_len, token_z]
    token_z = 128  # Example dimension
    z = torch.randn((batch_size, seq_len, seq_len, token_z), device=device) * 0.01
    
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
    """Run an example of the RNA MSA Module."""
    # Parameters
    batch_size = 2
    seq_len = 16
    num_seqs = 4
    
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
    
    # Create sample data
    print("Creating sample RNA data...")
    feats, z = create_sample_rna_data(batch_size, seq_len, num_seqs)
    
    # Get device
    device = z.device
    print(f"Using device: {device}")
    
    # Initialize RNAInputEmbedder
    print("Initializing RNAInputEmbedder...")
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
    
    # Initialize RNAMSAModule
    print("Initializing RNAMSAModule...")
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
    
    # Initialize RNADistogramModule
    print("Initializing RNADistogramModule...")
    distogram_module = RNADistogramModule(
        token_z=token_z,
        num_bins=36,  # Example number of distance bins
    ).to(device)
    
    # Forward pass through RNAInputEmbedder
    print("\nRunning forward pass through RNAInputEmbedder...")
    s, base_pairing = input_embedder(feats)
    
    print(f"Embedded tokens shape: {s.shape}")
    print(f"Base-pairing constraints shape: {base_pairing.shape}")
    
    # Forward pass through RNAMSAModule
    print("\nRunning forward pass through RNAMSAModule...")
    z_out = msa_module(z, s, feats, base_pairing)
    print(f"Output pairwise embeddings shape: {z_out.shape}")
    
    # Construct structure info for distogram prediction with comprehensive RNA features
    structure_info = {
        'stem_prob': feats['stem_prob'].mean(dim=1),
        'loop_prob': feats['loop_prob'].mean(dim=1),
        'bulge_prob': feats['bulge_prob'].mean(dim=1),
        'base_pairing': base_pairing  # Add base-pairing constraints
    }
    
    print("\nPreparing structural information for distogram prediction...")
    print(f"stem_prob shape: {structure_info['stem_prob'].shape}")
    print(f"loop_prob shape: {structure_info['loop_prob'].shape}")
    print(f"bulge_prob shape: {structure_info['bulge_prob'].shape}")
    print(f"base_pairing shape: {structure_info['base_pairing'].shape}")
    
    # Forward pass through RNADistogramModule
    print("\nRunning forward pass through RNADistogramModule...")
    distogram = distogram_module(z_out, structure_info)
    print(f"Output distogram shape: {distogram.shape}")
    
    print("\nExample summary:")
    print("-------------------------")
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
    print("\nSuccessfully ran RNA MSA module example!") 