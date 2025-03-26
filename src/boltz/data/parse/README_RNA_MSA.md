# RNA MSA Parser

This module provides tools for parsing and processing RNA Multiple Sequence Alignment (MSA) files. It's designed to handle RNA-specific features including secondary structure prediction, deletions, and one-hot encoding.

## Key Features

- Parse RNA MSA files in FASTA format
- Proper handling of deletions and insertions
- Secondary structure prediction using ViennaRNA (if available)
- Conversion to one-hot encoding for model input
- Statistics generation for RNA sequences

## Main Components

### RNAMSA Class

The `RNAMSA` class represents an RNA multiple sequence alignment and provides methods to:
- Extract individual sequences and their deletion information
- Convert to one-hot encoded format
- Convert back to string representation
- Calculate statistics about the alignment

### Functions

- `parse_rna_msa`: Parses RNA MSA files and returns an RNAMSA object
- `load_rna_msa_data`: Loads an RNA MSA file and returns a dictionary of tensors
- `extract_secondary_structure`: Predicts secondary structure features
- `parse_dot_bracket`: Parses dot-bracket notation to get structure probabilities

## Usage Examples

### Basic Usage

```python
from boltz.data.parse.rna_msa import parse_rna_msa, load_rna_msa_data

# Parse an MSA file
msa = parse_rna_msa("/path/to/rna.MSA.fasta", max_seqs=4)

# Get information about the MSA
num_sequences = len(msa)
seq_residues, seq_deletions = msa.get_sequence(0)  # Get first sequence

# Get one-hot encoded representation
one_hot_data = msa.to_one_hot(max_seqs=4, max_len=100)

# Load MSA with secondary structure prediction
msa_data = load_rna_msa_data("/path/to/rna.MSA.fasta", max_seqs=4, predict_structure=True)
```

### Feature Dictionary Format

The `load_rna_msa_data` function returns a dictionary with the following keys:

- `rna_msa`: One-hot encoded RNA sequences with shape [num_seqs, seq_len, 4]
- `has_deletion`: Boolean mask for positions with deletions [num_seqs, seq_len]
- `deletion_value`: Count of deletions at each position [num_seqs, seq_len]
- `msa_mask`: Mask for valid positions [num_seqs, seq_len]
- `query_mask`: Mask for query sequence [seq_len]
- `stem_prob`: Probability of each position being in a stem [num_seqs, seq_len]
- `loop_prob`: Probability of each position being in a loop [num_seqs, seq_len]
- `bulge_prob`: Probability of each position being in a bulge [num_seqs, seq_len]

## RNA Nucleotide Representation

RNA nucleotides are encoded as follows:
- A (Adenine): 0
- U (Uracil): 1
- G (Guanine): 2
- C (Cytosine): 3

## Secondary Structure Features

The parser provides three types of secondary structure probabilities:
1. **Stem probability**: Likelihood of a position being part of a helical stem
2. **Loop probability**: Likelihood of a position being part of a loop
3. **Bulge probability**: Likelihood of a position being part of a bulge or external region

## Example Script

See `boltz/src/boltz/examples/parse_rna_msa_example.py` for a complete example of how to use the RNA MSA parser with detailed statistics and visualization.

## Requirements

- numpy
- torch
- Optional: ViennaRNA package for accurate secondary structure prediction

## Notes on MSA Format

The parser expects MSA files in FASTA format where:
- Each sequence is preceded by a header line starting with '>'
- Deletions are represented by '-'
- Lowercase letters can be treated as insertions (controlled by the `handle_lowercase` parameter)
- The first sequence is assumed to be the query sequence

## Handling Special Cases

- T (thymine) is automatically converted to U (uracil)
- Lowercase letters can be treated as insertions using the `handle_lowercase` parameter
- Unknown nucleotides (N, X, etc.) are handled gracefully 

## RNA MSA Module Enhancements

The RNA MSA Module introduces several RNA-specific features to improve structure prediction accuracy. These enhancements leverage domain knowledge about RNA biology and provide significant advantages over generic protein-based MSA approaches.

### Key Enhancements

1. **RNA Secondary Structure Integration**
   - Three distinct probabilities (`stem_prob`, `loop_prob`, `bulge_prob`) capture the complex secondary structure of RNA
   - ViennaRNA integration for accurate structure prediction based on thermodynamic principles
   - Secondary structure guides spatial relationships in 3D structure prediction

2. **Base-Pairing Constraints**
   - Watson-Crick pairs (A-U, G-C) recognized with 0.5 weight (1.0 / temperature)
   - Wobble pairs (G-U) supported with 0.4 weight (0.8 / temperature)
   - Temperature scaling (default 2.0) provides flexible constraints rather than rigid rules
   - Base-pairing information directly influences attention mechanisms in the MSA module

3. **RNA-Specific Feature Processing**
   - Nucleotide frequencies calculated across MSA sequences (not batches) for better evolutionary signal
   - Profile features include conservation measures, GC content, and purine/pyrimidine ratios
   - Stem probabilities used to create dynamic pairing indicators (`stem_prob > 0.5`)

4. **Distogram Prediction Improvements**
   - Secondary structure integrated into distance prediction
   - Base-pairing constraints guide expected distances between nucleotides
   - Structure-aware attention mechanisms in the MSA module

### Domain Knowledge Integration

1. **RNA Structure Hierarchy**
   - **Primary**: Nucleotide sequence (A, U, G, C)
   - **Secondary**: Base-pairing patterns forming stems, loops, and bulges
   - **Tertiary**: 3D folding influenced by secondary structure elements

2. **RNA Secondary Structure Elements**
   - **Stems**: Helical regions formed by contiguous base pairs, providing structural stability
   - **Loops**: Unpaired regions connecting helical segments (hairpin, internal, multi-branch loops)
   - **Bulges**: Unpaired nucleotides within otherwise regular stems, creating bends in the helix

3. **Base-Pairing Principles**
   - **Watson-Crick pairs**: A-U and G-C, strongest and most common
   - **Wobble pairs**: G-U, weaker but biologically significant
   - **Non-canonical pairs**: Less common but important for specialized structures

4. **Structure Prediction Approach**
   - Dot-bracket notation parsed to extract structure probabilities
   - Minimum free energy (MFE) calculation from ViennaRNA
   - Structure classification based on context and pairing patterns

### Benefits to Model Training

1. **Improved Structure Prediction**
   - More accurate distance distributions between nucleotides
   - Better recognition of structural motifs in RNA
   - Reduced false positives in predicted interactions

2. **Enhanced Feature Representation**
   - Richer input embeddings with biologically relevant information
   - More informative attention patterns reflecting RNA structure
   - Nucleotide interactions guided by established biochemical principles

3. **Efficiency Improvements**
   - Reduced redundancy by using `stem_prob` directly for base-pairing
   - Better initialization with structure-aware tensors
   - More effective gradient flow through structure-guided attention

4. **Generalization to Diverse RNA Families**
   - Structure-based features transfer well across different RNA types
   - Evolutionary patterns captured through MSA processing
   - Secondary structure constraints applicable across the RNA universe

These enhancements significantly improve the RNA MSA Module's ability to model RNA structure by incorporating critical domain knowledge about RNA biology, resulting in more accurate predictions and better interpretability of the model's outputs. 