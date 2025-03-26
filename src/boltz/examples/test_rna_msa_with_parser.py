#!/usr/bin/env python
"""
Script to test the RNA MSA Module with real data, using the RNA MSA parser.

This script loads real RNA sequence data from:
1. CSV file with RNA sequence information at /ist-nas/users/bunditb/boltz/stanford-rna/train_sequences.csv
2. MSA files from /ist-nas/users/bunditb/boltz/stanford-rna/MSA

It demonstrates:
1. How to use the RNA MSA parser to prepare input data
2. How to initialize and run the RNA MSA Module
3. The expected output format
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add the src directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from boltz.data import const
from boltz.model.modules.rna_trunk import (
    RNAInputEmbedder,
    RNAMSAModule,
    RNADistogramModule
)
from boltz.data.parse.rna_msa import load_rna_msa_data

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define RNA-specific constants
if not hasattr(const, "rna_num_tokens"):
    setattr(const, "rna_num_tokens", 4)  # A, U, G, C

def print_tensor_stats(name, tensor):
    """Print statistics about a tensor."""
    if isinstance(tensor, torch.Tensor):
        print(f"{name}: shape={tensor.shape}, dtype={tensor.dtype}")
        # Handle boolean tensors by converting to float
        if tensor.dtype == torch.bool:
            float_tensor = tensor.float()
            print(f"  Stats: min={float_tensor.min().item():.4f}, max={float_tensor.max().item():.4f}, "
                  f"mean={float_tensor.mean().item():.4f}, std={float_tensor.std().item():.4f}")
        else:
            print(f"  Stats: min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, "
                  f"mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}")
    else:
        print(f"{name}: {tensor}")

def load_real_rna_data(csv_path: str, msa_dir: str, batch_size: int = 2, max_seq_len: int = 100, max_num_seqs: int = 4) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """
    Load real RNA data from CSV file and MSA directory using the RNA MSA parser.
    
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
    msa_features_list = []
    
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
        
        # Use the RNA MSA parser to load the MSA file
        if os.path.exists(msa_file):
            # Load the MSA data
            msa_features = load_rna_msa_data(
                msa_file=msa_file, 
                max_seqs=max_num_seqs, 
                max_len=max_seq_len,
                predict_structure=True
            )
            print(f"Loaded MSA file with {msa_features['rna_msa'].shape[0]} sequences")
        else:
            raise ValueError(f"MSA file not found: {msa_file}")
            
        msa_features_list.append(msa_features)
    
    # Find the maximum sequence length in the batch after processing
    max_len = max([msa_feat['rna_msa'].shape[1] for msa_feat in msa_features_list])
    print(f"Maximum sequence length after processing: {max_len}")
    
    # Combine features from all MSAs into batch tensors
    batch_features = {}
    for key in ['rna_msa', 'has_deletion', 'deletion_value', 'stem_prob', 'loop_prob', 'bulge_prob', 'msa_mask']:
        # Skip if any MSA features don't have this key
        if any(key not in msa_feat for msa_feat in msa_features_list):
            continue
            
        # Get the shapes from the first MSA
        shape1 = msa_features_list[0][key].shape
        
        # Create a batch tensor with padding to max_len
        if len(shape1) == 2:  # [num_seqs, seq_len]
            batch_tensor = torch.zeros((batch_size, shape1[0], max_len), device=device)
        elif len(shape1) == 3:  # [num_seqs, seq_len, feat_dim]
            batch_tensor = torch.zeros((batch_size, shape1[0], max_len, shape1[2]), device=device)
        else:
            print(f"Unexpected shape for {key}: {shape1}")
            continue
            
        # Fill the batch tensor
        for i, msa_feat in enumerate(msa_features_list):
            if key not in msa_feat:
                continue
                
            tensor = msa_feat[key]
            if len(tensor.shape) == 2:  # [num_seqs, seq_len]
                batch_tensor[i, :, :tensor.shape[1]] = tensor
            elif len(tensor.shape) == 3:  # [num_seqs, seq_len, feat_dim]
                batch_tensor[i, :, :tensor.shape[1], :] = tensor
                
        batch_features[key] = batch_tensor
        
    # Handle query_mask separately (it has a different shape)
    if 'query_mask' in msa_features_list[0]:
        batch_query_mask = torch.zeros((batch_size, max_len), device=device)
        for i, msa_feat in enumerate(msa_features_list):
            if 'query_mask' in msa_feat:
                query_mask = msa_feat['query_mask']
                batch_query_mask[i, :query_mask.shape[0]] = query_mask
        batch_features['query_mask'] = batch_query_mask
    
    # Create one-hot encoded nucleotide types
    # This can be derived directly from the first sequence in each MSA
    nuc_type = torch.zeros((batch_size, max_len, const.rna_num_tokens), device=device)
    for i, msa_feat in enumerate(msa_features_list):
        if 'rna_msa' in msa_feat:
            seq_len = msa_feat['rna_msa'].shape[1]
            nuc_type[i, :seq_len, :] = msa_feat['rna_msa'][0, :seq_len, :]  # First sequence
    batch_features['nuc_type'] = nuc_type
    
    # Create pairwise representation (randomly initialized instead of zeros)
    token_z = 128  # Ensure this matches token_z in run_example (128)
    # Use a small standard deviation to prevent initial large values
    z = torch.randn((batch_size, max_len, max_len, token_z), device=device) * 0.02
    print(f"Created randomly initialized pairwise embeddings z with shape {z.shape}")
    print(f"  Random init stats: min={z.min().item():.4f}, max={z.max().item():.4f}, mean={z.mean().item():.4f}, std={z.std().item():.4f}")

    # Calculate actual profile based on MSA data (nucleotide frequencies) instead of using dummy values
    if 'rna_msa' in batch_features:
        # Shape of rna_msa: [batch_size, num_seqs, max_len, 4]
        msa_data = batch_features['rna_msa']
        
        # Count valid sequences using msa_mask
        if 'msa_mask' in batch_features:
            mask = batch_features['msa_mask'].unsqueeze(-1)  # [batch_size, num_seqs, max_len, 1]
            # Apply mask to count only valid positions
            masked_msa = msa_data * mask
            # Sum over dimension 1 (num_seqs) to get counts for each position across all MSA sequences
            seq_counts = masked_msa.sum(dim=1)  # [batch_size, max_len, 4]
            seq_total = seq_counts.sum(dim=-1, keepdim=True) + 1e-8
        else:
            # Sum over dimension 1 (num_seqs) to get counts for each position across all MSA sequences
            seq_counts = msa_data.sum(dim=1)  # [batch_size, max_len, 4]
            seq_total = seq_counts.sum(dim=-1, keepdim=True) + 1e-8
        
        # Calculate nucleotide frequencies for each position within each batch item
        # This is done per batch item, NOT across batches
        nucleotide_freqs = seq_counts / seq_total  # [batch_size, max_len, 4]
        
        # Create profile with nucleotide frequencies in first 4 positions
        profile = torch.zeros((batch_size, max_len, 20), device=device)
        profile[:, :, :4] = nucleotide_freqs
        
        # Calculate additional informative features for the remaining dimensions:
        # All calculations are done per batch item, using the MSA information
        
        # 1. Information content/conservation (higher values mean more conserved positions)
        # Shannon entropy of nucleotide distribution (scaled to be between 0-1)
        entropy = -torch.sum(nucleotide_freqs * torch.log2(nucleotide_freqs + 1e-10), dim=-1) / 2.0
        conservation = 1.0 - entropy.unsqueeze(-1)  # Higher values = more conserved
        profile[:, :, 4] = conservation.squeeze(-1)
        
        # 2. GC content percentage at each position
        gc_content = (nucleotide_freqs[:, :, 2] + nucleotide_freqs[:, :, 3])  # G + C
        profile[:, :, 5] = gc_content
        
        # 3. Purine/Pyrimidine ratio (A+G vs C+U)
        purine = (nucleotide_freqs[:, :, 0] + nucleotide_freqs[:, :, 2])  # A + G
        pyrimidine = (nucleotide_freqs[:, :, 1] + nucleotide_freqs[:, :, 3])  # U + C
        purine_ratio = purine / (pyrimidine + 1e-8)
        profile[:, :, 6] = torch.clamp(purine_ratio, 0, 5)  # Clamp to avoid extreme values
        
        batch_features['profile'] = profile
        
        print(f"Created profile from MSA data - shape: {profile.shape}")
        print(f"  Profile stats: min={profile.min().item():.4f}, max={profile.max().item():.4f}, mean={profile.mean().item():.4f}")
    else:
        # Fallback using nuc_type information if no MSA data available
        profile = torch.zeros((batch_size, max_len, 20), device=device)
        # Fill the first 4 positions with the one-hot encoding
        profile[:, :, :4] = batch_features['nuc_type']
        # Fill the remaining positions with small random values
        profile[:, :, 4:] = torch.randn((batch_size, max_len, 16), device=device) * 0.01
        batch_features['profile'] = profile
        print("Created profile using one-hot encoded nucleotides with random initialization for remaining features")

    # Add deletion_mean with small random values instead of zeros
    batch_features['deletion_mean'] = torch.rand((batch_size, max_len), device=device) * 0.005  # Very small values

    # Add sec_structure - use the structure probabilities we already have
    if 'stem_prob' in batch_features and 'loop_prob' in batch_features and 'bulge_prob' in batch_features:
        # Average the structural features across MSA sequences (dim=1) for each batch item
        stem_prob = batch_features['stem_prob'].mean(dim=1)  # [batch_size, seq_len]
        loop_prob = batch_features['loop_prob'].mean(dim=1)  # [batch_size, seq_len]
        bulge_prob = batch_features['bulge_prob'].mean(dim=1)  # [batch_size, seq_len]
        
        # Stack the averaged structural features for each batch item
        sec_structure = torch.stack([stem_prob, loop_prob, bulge_prob], dim=2)  # [batch_size, seq_len, 3]
        batch_features['sec_structure'] = sec_structure
        
        print(f"Created secondary structure tensor from MSA data - shape: {sec_structure.shape}")
    else:
        # Create tensor with random small values instead of zeros
        # We use softmax to ensure probabilities sum to 1 across the 3 structure types
        random_probs = torch.rand((batch_size, max_len, 3), device=device)
        sec_structure = torch.softmax(random_probs, dim=-1)  # Ensures sum to 1 for each position
        batch_features['sec_structure'] = sec_structure
        print("Created randomly initialized secondary structure probabilities")

    # Add token_pad_mask - we can derive this from query_mask or create it from scratch
    if 'query_mask' in batch_features:
        # Use query_mask as token_pad_mask
        batch_features['token_pad_mask'] = batch_features['query_mask']
    else:
        # Create a mask where all tokens are valid (non-padded)
        batch_features['token_pad_mask'] = torch.ones((batch_size, max_len), device=device)

    # Add other necessary features for the MSA module
    # has_deletion and deletion_value should already be in batch_features
    
    return batch_features, z

def run_example():
    """Run an example of the RNA MSA Module with real data using the RNA MSA parser."""
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
    token_z = 128  # Changed back to 128 to match the expected dimension in pair_averaging
    
    # Atoms per window parameters
    atoms_per_window_queries = 8
    atoms_per_window_keys = 8
    
    # Atom encoder parameters
    atom_feature_dim = 16
    atom_encoder_depth = 2
    atom_encoder_heads = 4
    
    # MSA parameters
    msa_s = 32
    s_input_dim = 60  # This is the actual output dimension from RNAInputEmbedder in the main script
    msa_blocks = 2
    msa_dropout = 0.15
    z_dropout = 0.15
    
    # Load real RNA data using the RNA MSA parser
    print("Loading real RNA data with RNA MSA parser...")
    feats, z = load_real_rna_data(csv_path, msa_dir, batch_size, max_seq_len, max_num_seqs)
    
    # After loading the data and before printing the general input features, add this section
    # Detailed analysis of structure probabilities for the first sequence
    print("\n===== DETAILED STRUCTURE ANALYSIS FOR FIRST SEQUENCE =====")
    if 'stem_prob' in feats and 'loop_prob' in feats and 'bulge_prob' in feats:
        # Get the sequence from our previously loaded data
        csv_data = pd.read_csv(csv_path)
        first_pdb_id = csv_data['target_id'].iloc[0]
        first_seq = csv_data['sequence'].iloc[0][:max_seq_len]
        print(f"RNA Sequence: {first_seq}")
        print(f"PDB ID: {first_pdb_id}")
        print(f"MSA File: {msa_dir}/{first_pdb_id}.MSA.fasta")
        
        # Get structure probabilities for first batch item, first sequence
        first_stem_prob = feats['stem_prob'][0, 0, :len(first_seq)].cpu().numpy()
        first_loop_prob = feats['loop_prob'][0, 0, :len(first_seq)].cpu().numpy()
        first_bulge_prob = feats['bulge_prob'][0, 0, :len(first_seq)].cpu().numpy()
        
        # Create a table with nucleotides and their structure probabilities
        print("\nPositional Structure Probabilities:")
        print("Pos  Nucleotide  Stem Prob  Loop Prob  Bulge Prob  Most Likely")
        print("-" * 65)
        
        for i, nuc in enumerate(first_seq):
            stem_p = first_stem_prob[i]
            loop_p = first_loop_prob[i]
            bulge_p = first_bulge_prob[i]
            
            # Determine most likely structure
            probs = [stem_p, loop_p, bulge_p]
            most_likely_idx = np.argmax(probs)
            structure_type = ["Stem", "Loop", "Bulge"][most_likely_idx]
            
            print(f"{i+1:3d}  {nuc:10s}  {stem_p:.4f}     {loop_p:.4f}     {bulge_p:.4f}     {structure_type}")
        
        # Try to get the original dot-bracket structure if available
        try:
            import RNA
            fc = RNA.fold_compound(first_seq)
            structure, mfe = fc.mfe()
            print(f"\nViennaRNA Dot-Bracket Structure (MFE: {mfe:.2f} kcal/mol):")
            print(first_seq)
            print(structure)
        except Exception as e:
            print(f"\nCouldn't get ViennaRNA structure: {e}")
            
            # Reconstruct approximate dot-bracket notation from probabilities
            dot_bracket = ""
            for i in range(len(first_seq)):
                if first_stem_prob[i] > 0.5:  # Threshold for being considered part of a stem
                    # Ideally, we would distinguish between opening and closing brackets
                    # but for simplicity, we'll use '(' for stem positions
                    dot_bracket += "("
                elif first_loop_prob[i] > first_bulge_prob[i]:
                    dot_bracket += "."
                else:
                    dot_bracket += "."
            
            print(f"\nApproximate Dot-Bracket Structure (reconstructed from probabilities):")
            print(first_seq)
            print(dot_bracket)
        
        # Calculate statistics for this sequence
        print(f"\nStructure Statistics for First Sequence:")
        print(f"  - Stem positions: {np.sum(first_stem_prob > 0.5)}/{len(first_seq)} ({np.mean(first_stem_prob)*100:.1f}%)")
        print(f"  - Loop-dominant positions: {np.sum(first_loop_prob > first_bulge_prob)}/{len(first_seq)} ({np.mean(first_loop_prob)*100:.1f}%)")
        print(f"  - Bulge-dominant positions: {np.sum(first_bulge_prob > first_loop_prob)}/{len(first_seq)} ({np.mean(first_bulge_prob)*100:.1f}%)")

    # Continue with the original input features printing
    print("\n===== INPUT FEATURES =====")
    for key, value in feats.items():
        if isinstance(value, torch.Tensor):
            print_tensor_stats(key, value)
            # Add note for stem_prob
            if key == 'stem_prob':
                print("  Note: stem_prob is used as source for base-pairing information")

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
    
    # Forward pass through RNAInputEmbedder
    print("Running forward pass through RNAInputEmbedder...")
    print(f"RNAInputEmbedder parameters: {sum(p.numel() for p in input_embedder.parameters())}")
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
    # Create thresholded stem_prob for paired positions (previously msa_paired)
    print("Using thresholded stem_prob (>0.5) as indicator for paired positions")
    # Ensure feats dictionary has the required features for the RNA MSA module
    feats['msa_paired'] = (feats['stem_prob'] > 0.5).float()  # Dynamically add paired indicator on-the-fly
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
        print_tensor_stats(key, value)
    
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
    print("  - Secondary structure features (stem_prob, loop_prob, bulge_prob): [batch, num_seqs, seq_len]")
    print("    Note: stem_prob > 0.5 is used as base-pairing indicator")
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
    print("\nSuccessfully ran RNA MSA module with real data using RNA MSA parser!") 