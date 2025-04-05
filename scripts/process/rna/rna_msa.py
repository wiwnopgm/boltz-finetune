#!/usr/bin/env python
"""
NumPy-based RNA MSA processor without PyTorch dependencies.

This script processes RNA MSA files in FASTA format and extracts
secondary structure information using ViennaRNA. The processed
data is saved as compressed numpy files for efficient loading
during model training.

This version doesn't depend on PyTorch to avoid module conflicts.
"""

import argparse
import os
import sys
import time
import gzip
import re
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    print("Warning: pandas package not found. CSV file support will be disabled.")
    HAS_PANDAS = False

# Try to import RNA (ViennaRNA)
try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    print("Warning: ViennaRNA package not found. Secondary structure prediction will be disabled.")
    HAS_VIENNA = False

# RNA nucleotide indices for one-hot encoding
RNA_A, RNA_U, RNA_G, RNA_C = 0, 1, 2, 3

# Mapping from nucleotide to one-hot index
RNA_NUC_TO_INDEX = {
    'A': RNA_A, 'U': RNA_U, 'G': RNA_G, 'C': RNA_C, 
    'T': RNA_U,  # Convert T to U
    'N': -1,     # Unknown nucleotide
    '-': -1,     # Gap
    '.': -1,     # Gap
    'X': -1,     # Unknown
    ' ': -1,     # Space
    # Lowercase versions
    'a': RNA_A, 'u': RNA_U, 'g': RNA_G, 'c': RNA_C, 't': RNA_U
}

# Mapping from index to nucleotide
RNA_INDEX_TO_NUC = {
    RNA_A: 'A', RNA_U: 'U', RNA_G: 'G', RNA_C: 'C'
}


class RNAMSA:
    """Class to represent an RNA Multiple Sequence Alignment."""
    
    def __init__(self, sequences: List[str]):
        """Initialize the RNA MSA object with a list of sequences."""
        self.sequences = sequences
    
    def to_one_hot(self, max_seqs: Optional[int] = None, max_len: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Convert the MSA to one-hot encoded numpy arrays with derived features."""
        # Determine dimensions
        num_seqs = min(len(self.sequences), max_seqs) if max_seqs is not None else len(self.sequences)
        max_sequence_length = max(len(seq) for seq in self.sequences[:num_seqs])
        if max_len is not None:
            max_sequence_length = min(max_sequence_length, max_len)
        
        # Initialize arrays
        rna_msa = np.zeros((num_seqs, max_sequence_length, 4))
        has_deletion = np.zeros((num_seqs, max_sequence_length), dtype=bool)
        deletion_value = np.zeros((num_seqs, max_sequence_length))
        msa_mask = np.zeros((num_seqs, max_sequence_length))
        
        # Fill arrays with data
        for i in range(num_seqs):
            seq = self.sequences[i]
            seq_len = min(len(seq), max_sequence_length)
            
            # Set mask for valid positions
            msa_mask[i, :seq_len] = 1.0
            
            # Fill one-hot encoding
            for j in range(seq_len):
                nucleotide = seq[j]
                idx = RNA_NUC_TO_INDEX.get(nucleotide, -1)
                if idx >= 0:
                    rna_msa[i, j, idx] = 1.0
                
                # Record deletions
                if nucleotide == '-' or nucleotide == '.':
                    has_deletion[i, j] = True
                    deletion_value[i, j] = 1.0
        
        # Create query mask (for first sequence)
        query_mask = np.zeros(max_sequence_length)
        query_len = min(len(self.sequences[0]), max_sequence_length)
        query_mask[:query_len] = 1.0
        
        # Calculate nucleotide frequencies across MSA sequences
        # Apply mask to count only valid positions
        masked_msa = rna_msa * msa_mask[:, :, np.newaxis]
        # Sum over sequences dimension to get counts
        seq_counts = masked_msa.sum(axis=0)  # [max_len, 4]
        seq_total = seq_counts.sum(axis=1, keepdims=True) + 1e-8
        nucleotide_freqs = seq_counts / seq_total  # [max_len, 4]
        
        # Calculate additional informative features
        # Information content/conservation (higher values mean more conserved positions)
        entropy = -np.sum(nucleotide_freqs * np.log2(nucleotide_freqs + 1e-10), axis=1) / 2.0
        conservation = 1.0 - entropy
        
        # GC content percentage at each position
        gc_content = nucleotide_freqs[:, 2] + nucleotide_freqs[:, 3]  # G + C
        
        # Purine/Pyrimidine ratio (A+G vs C+U)
        purine = nucleotide_freqs[:, 0] + nucleotide_freqs[:, 2]  # A + G
        pyrimidine = nucleotide_freqs[:, 1] + nucleotide_freqs[:, 3]  # U + C
        purine_ratio = purine / (pyrimidine + 1e-8)
        purine_ratio = np.clip(purine_ratio, 0, 5)  # Clamp to avoid extreme values
        
        # Create a nucleotide profile with frequencies and derived features
        profile = np.zeros((max_sequence_length, 7))
        profile[:, :4] = nucleotide_freqs
        profile[:, 4] = conservation
        profile[:, 5] = gc_content
        profile[:, 6] = purine_ratio
        
        # Calculate deletion mean across sequences
        deletion_mean = deletion_value.mean(axis=0)
        
        return {
            'rna_msa': rna_msa,
            'has_deletion': has_deletion,
            'deletion_value': deletion_value,
            'msa_mask': msa_mask,
            'query_mask': query_mask,
            'profile': profile,
            'deletion_mean': deletion_mean,
            'nucleotide_freqs': nucleotide_freqs
        }


def parse_dot_bracket(dot_bracket: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse dot-bracket notation to get stem, loop, and bulge probabilities."""
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
        elif char == ')' and stack:
            j = stack.pop()
            pair_dict[j] = i
            pair_dict[i] = j
    
    # Second pass: assign probabilities
    for i, char in enumerate(dot_bracket):
        if char in '()':
            stem_prob[i] = 1.0
        else:  # '.'
            # Check if it's a loop or likely a bulge
            left_paired = any(j in pair_dict for j in range(max(0, i-4), i))
            right_paired = any(j in pair_dict for j in range(i+1, min(i+5, n)))
            
            if left_paired and right_paired:
                # Inside a loop
                loop_prob[i] = 0.8
                bulge_prob[i] = 0.2
            else:
                # More likely a bulge or external loop
                loop_prob[i] = 0.3
                bulge_prob[i] = 0.7
    
    return stem_prob, loop_prob, bulge_prob


def extract_secondary_structure(sequences: List[str]) -> Dict[str, np.ndarray]:
    """Predict secondary structure for RNA sequences."""
    if not HAS_VIENNA:
        print("Warning: ViennaRNA package not found. Using fallback structure prediction.")
    
    num_seqs = len(sequences)
    seq_len = max(len(seq) for seq in sequences)
    
    # Initialize arrays
    stem_prob = np.zeros((num_seqs, seq_len))
    loop_prob = np.zeros((num_seqs, seq_len))
    bulge_prob = np.zeros((num_seqs, seq_len))
    
    # Get nucleotide distribution for fallback cases
    # This is more data-driven than using fixed constants
    valid_nucs = 0
    a_count, u_count, g_count, c_count = 0, 0, 0, 0
    
    for seq in sequences:
        for nuc in seq:
            if nuc.upper() in 'AUGC':
                valid_nucs += 1
                if nuc.upper() == 'A':
                    a_count += 1
                elif nuc.upper() == 'U' or nuc.upper() == 'T':
                    u_count += 1
                elif nuc.upper() == 'G':
                    g_count += 1
                elif nuc.upper() == 'C':
                    c_count += 1
    
    # Calculate base probability of being in a stem based on nucleotide distribution
    # G-C rich regions have higher probability of forming stems
    gc_content = (g_count + c_count) / max(1, valid_nucs)
    fallback_stem_prob = min(0.6, max(0.3, gc_content * 0.8))
    fallback_loop_prob = (1.0 - fallback_stem_prob) * 0.6
    fallback_bulge_prob = 1.0 - fallback_stem_prob - fallback_loop_prob
    
    # If ViennaRNA is not available, use simplified prediction based on sequence patterns
    if not HAS_VIENNA:
        for i, seq in enumerate(sequences):
            seq = seq.replace('T', 'U').replace('t', 'u')  # Convert T to U
            seq_len = len(seq)
            
            for j in range(seq_len):
                # Skip gaps
                if seq[j] in '-.':
                    continue
                    
                # Use sequence patterns for simple approximation
                # Look at 5-base windows for potential complementary sequences
                window_size = 5
                half_window = window_size // 2
                
                # Find potential base pairs in a simplified way
                local_gc = 0
                for k in range(max(0, j-half_window), min(seq_len, j+half_window+1)):
                    if k != j and seq[k].upper() in 'GC':
                        local_gc += 1
                
                # Calculate local stem probability based on GC content and other heuristics
                local_gc_ratio = local_gc / min(window_size, seq_len)
                stem_p = min(0.7, max(0.2, fallback_stem_prob + local_gc_ratio * 0.3))
                
                # Assign probabilities
                stem_prob[i, j] = stem_p
                remaining = 1.0 - stem_p
                loop_prob[i, j] = remaining * 0.6
                bulge_prob[i, j] = remaining * 0.4
    else:
        # Predict secondary structure for each sequence using ViennaRNA
        for i, seq in enumerate(sequences):
            seq = seq.replace('T', 'U').replace('t', 'u')  # Convert T to U
            
            # Skip sequences with non-standard nucleotides
            if not all(c.upper() in 'AUGC-.' for c in seq):
                # Use data-driven probabilities instead of hardcoded values
                seq_len = len(seq)
                stem_prob[i, :seq_len] = fallback_stem_prob
                loop_prob[i, :seq_len] = fallback_loop_prob
                bulge_prob[i, :seq_len] = fallback_bulge_prob
                continue
            
            # Remove gaps for structure prediction
            seq_no_gaps = seq.replace('-', '').replace('.', '')
            if not seq_no_gaps:
                continue
            
            try:
                # Use ViennaRNA to predict structure
                fc = RNA.fold_compound(seq_no_gaps)
                structure, _ = fc.mfe()
                
                # Also get base pair probabilities if possible
                try:
                    fc.pf()  # Partition function calculation
                    bpp = fc.bpp()  # Base pair probabilities
                    has_bpp = True
                except:
                    has_bpp = False
                
                # Parse dot-bracket notation
                s_prob, l_prob, b_prob = parse_dot_bracket(structure)
                
                # Enhance with base pair probabilities if available
                if has_bpp:
                    # Base pair probabilities are in the form of a matrix
                    # We can use this to refine our stem probabilities
                    struct_len = len(seq_no_gaps)
                    for j in range(struct_len):
                        # If any strong base pairing exists for this position, increase stem probability
                        paired_prob = sum(bpp[j+1][k+1] for k in range(struct_len) if k != j)
                        if paired_prob > 0:
                            s_prob[j] = max(s_prob[j], paired_prob)
                            # Adjust loop and bulge probabilities accordingly
                            total = s_prob[j] + l_prob[j] + b_prob[j]
                            if total > 0:
                                l_prob[j] *= (1 - s_prob[j]) / total
                                b_prob[j] *= (1 - s_prob[j]) / total
                
                # Map back to original sequence with gaps
                struct_idx = 0
                for j, nuc in enumerate(seq):
                    if nuc not in '-.' and struct_idx < len(s_prob):
                        stem_prob[i, j] = s_prob[struct_idx]
                        loop_prob[i, j] = l_prob[struct_idx]
                        bulge_prob[i, j] = b_prob[struct_idx]
                        struct_idx += 1
                    elif j > 0 and j < len(seq) - 1:
                        # For gaps, use average of neighbors
                        stem_prob[i, j] = (stem_prob[i, j-1] + stem_prob[i, j+1]) / 2
                        loop_prob[i, j] = (loop_prob[i, j-1] + loop_prob[i, j+1]) / 2
                        bulge_prob[i, j] = (bulge_prob[i, j-1] + bulge_prob[i, j+1]) / 2
            except Exception as e:
                print(f"Error predicting structure for sequence {i}: {e}")
                # Use data-driven probabilities if prediction fails
                seq_len = len(seq)
                stem_prob[i, :seq_len] = fallback_stem_prob
                loop_prob[i, :seq_len] = fallback_loop_prob
                bulge_prob[i, :seq_len] = fallback_bulge_prob
    
    return {
        'stem_prob': stem_prob,
        'loop_prob': loop_prob,
        'bulge_prob': bulge_prob
    }


def read_fasta(fasta_file: str, max_seqs: Optional[int] = None) -> List[str]:
    """Read sequences from a FASTA file."""
    sequences = []
    current_seq = ""
    
    # Check if file is gzipped
    is_gzipped = fasta_file.endswith('.gz')
    
    try:
        with gzip.open(fasta_file, 'rt') if is_gzipped else open(fasta_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_seq:
                        sequences.append(current_seq)
                        if max_seqs and len(sequences) >= max_seqs:
                            break
                    current_seq = ""
                else:
                    current_seq += line
            
            if current_seq and (not max_seqs or len(sequences) < max_seqs):
                sequences.append(current_seq)
    except Exception as e:
        print(f"Error reading file {fasta_file}: {e}")
    
    return sequences


def parse_rna_msa(path: str, max_seqs: Optional[int] = None, handle_lowercase: bool = True) -> RNAMSA:
    """Parse RNA MSA file."""
    # Read sequences from the MSA file
    sequences = read_fasta(path, max_seqs)
    
    # Process sequences
    processed_sequences = []
    
    for seq in sequences:
        # Convert T to U
        seq = seq.replace('T', 'U').replace('t', 'u')
        
        # Handle lowercase letters
        if handle_lowercase:
            seq = re.sub(r'[a-z]', '', seq)
        else:
            seq = seq.upper()
        
        processed_sequences.append(seq)
    
    return RNAMSA(processed_sequences)


def process_rna_msa(
    path: Path,
    outdir: Path,
    max_seqs: int,
    predict_structure: bool = True,
    handle_lowercase: bool = True,
    stem_prob_threshold: float = 0.5,  # Threshold for stem probabilities to create msa_paired
) -> None:
    """Process a single RNA MSA file."""
    # Create output filename
    out_path = outdir / f"{path.stem}.npz"
    
    # Skip if already processed
    if out_path.exists():
        return
    
    try:
        print(f"Processing {path.name}...")
        
        # Parse the RNA MSA file
        rna_msa = parse_rna_msa(
            path=str(path),
            max_seqs=max_seqs,
            handle_lowercase=handle_lowercase
        )
        
        # Convert to one-hot encoding with derived features
        features = rna_msa.to_one_hot(max_seqs=max_seqs)
        
        # Get additional features if requested
        if predict_structure:
            # Check if ViennaRNA is available but still proceed with fallback if not
            if not HAS_VIENNA:
                print(f"Note: Using simplified structure prediction for {path.name}")
                
            # Extract secondary structure features
            structure_features = extract_secondary_structure(rna_msa.sequences[:max_seqs])
            
            # Add structure features to the features dictionary
            features.update(structure_features)
            
            # Create msa_paired feature from stem_prob
            if 'stem_prob' in structure_features:
                stem_prob = structure_features['stem_prob']
                features['msa_paired'] = (stem_prob > stem_prob_threshold).astype(np.float32)
        
        # Create sec_structure by averaging the stem, loop, and bulge probabilities across MSA sequences
        if 'stem_prob' in features and 'loop_prob' in features and 'bulge_prob' in features:
            # Average the structural features across MSA sequences
            stem_prob_avg = features['stem_prob'].mean(axis=0)  # [seq_len]
            loop_prob_avg = features['loop_prob'].mean(axis=0)  # [seq_len]
            bulge_prob_avg = features['bulge_prob'].mean(axis=0)  # [seq_len]
            
            # Stack them to create the sec_structure feature
            sec_structure = np.stack([stem_prob_avg, loop_prob_avg, bulge_prob_avg], axis=1)  # [seq_len, 3]
            features['sec_structure'] = sec_structure
        
        # Save as compressed numpy file
        np.savez_compressed(out_path, **features)
        
        print(f"Successfully processed {path.name} -> {out_path.name}")
        
    except Exception as e:
        print(f"Error processing {path}: {e}")


def find_msa_files(msadir: Path, csv_path: Path = None) -> list:
    """Find all RNA MSA files in the directory or based on CSV file."""
    if csv_path is not None and csv_path.exists():
        # Use CSV file if available and pandas is installed
        if HAS_PANDAS:
            try:
                # Read CSV file to get list of RNA IDs
                df = pd.read_csv(csv_path)
                if 'target_id' in df.columns:
                    # Get list of PDB chain IDs from CSV
                    pdb_chains = df['target_id'].tolist()
                    # Find corresponding MSA files
                    msa_files = []
                    for pdb_chain in pdb_chains:
                        msa_file = msadir / f"{pdb_chain}.MSA.fasta"
                        if msa_file.exists():
                            msa_files.append(msa_file)
                    if msa_files:
                        return msa_files
                    else:
                        print(f"Warning: No MSA files found for IDs in {csv_path}. Falling back to directory scan.")
            except Exception as e:
                print(f"Error reading CSV file {csv_path}: {e}")
                print("Falling back to scanning directory for MSA files.")
        else:
            print(f"Warning: pandas not installed. Cannot read CSV file {csv_path}.")
            print("Falling back to scanning directory for MSA files.")
    
    # Fallback: Find all FASTA files in the directory
    fasta_files = list(msadir.glob("*.MSA.fasta")) + list(msadir.glob("*.fasta"))
    if not fasta_files:
        print(f"Warning: No MSA files found in directory {msadir}")
    return fasta_files


def process_with_multiprocessing(files, outdir, max_seqs, predict_structure, handle_lowercase, stem_prob_threshold, num_processes):
    """Process files using multiprocessing Pool."""
    process_func = partial(
        process_rna_msa,
        outdir=outdir,
        max_seqs=max_seqs,
        predict_structure=predict_structure,
        handle_lowercase=handle_lowercase,
        stem_prob_threshold=stem_prob_threshold
    )
    
    with Pool(processes=num_processes) as pool:
        _ = list(tqdm(pool.imap(process_func, files), total=len(files)))


def process(args) -> None:
    """Run the RNA MSA processing task."""
    # Create output directory
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # Find RNA MSA files
    data = find_msa_files(args.msadir, args.csv_path)
    print(f"Found {len(data)} RNA MSA files.")
    
    if not data:
        print("No RNA MSA files found. Please check the input directory or CSV file.")
        return
    
    # Process a subset if specified
    if args.num_files > 0:
        data = data[:args.num_files]
        print(f"Processing {len(data)} files as requested.")
    
    # Check if we can run in parallel
    num_processes = max(1, min(args.num_processes, cpu_count(), len(data)))
    parallel = num_processes > 1 and len(data) > 1
    
    start_time = time.time()
    
    # Process files
    if parallel:
        print(f"Processing {len(data)} files with {num_processes} parallel processes...")
        process_with_multiprocessing(
            files=data,
            outdir=args.outdir,
            max_seqs=args.max_seqs,
            predict_structure=args.predict_structure,
            handle_lowercase=args.handle_lowercase,
            stem_prob_threshold=args.stem_prob_threshold,
            num_processes=num_processes
        )
    else:
        print(f"Processing {len(data)} files sequentially...")
        for path in tqdm(data):
            process_rna_msa(
                path=path,
                outdir=args.outdir,
                max_seqs=args.max_seqs,
                predict_structure=args.predict_structure,
                handle_lowercase=args.handle_lowercase,
                stem_prob_threshold=args.stem_prob_threshold,
            )
    
    elapsed_time = time.time() - start_time
    print(f"Successfully processed {len(data)} RNA MSA files in {elapsed_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process RNA MSA files using NumPy only.")
    parser.add_argument("--msadir", type=Path, required=True, help="Directory with RNA MSA files")
    parser.add_argument("--outdir", type=Path, default="data/processed_rna_msa", help="Output directory")
    parser.add_argument("--csv-path", type=Path, help="Path to CSV with RNA sequence information")
    parser.add_argument("--max-seqs", type=int, default=256, help="Max sequences per MSA")
    parser.add_argument("--num-processes", type=int, default=cpu_count() // 2, help="Number of parallel processes")
    parser.add_argument("--predict-structure", action="store_true", help="Predict secondary structure")
    parser.add_argument("--handle-lowercase", action="store_true", help="Treat lowercase as insertions")
    parser.add_argument("--num-files", type=int, default=0, help="Number of files to process (0 for all)")
    parser.add_argument("--stem-prob-threshold", type=float, default=0.5, help="Threshold for stem probabilities to create msa_paired")
    
    args = parser.parse_args()
    process(args) 