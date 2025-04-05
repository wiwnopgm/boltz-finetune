# Structure Processing Module

This module provides tools for parsing and processing protein and RNA structures from various file formats.

For detailed data processing instructions, please see our [data processing guide](../../docs/training.md).

## Overview

The module includes parsers for multiple file formats:

- `mmcif.py`: Parses MMCIF formatted files
- `pdb.py`: Parses PDB formatted files and CSV files containing 3D coordinates
- `rcsb.py`: Main entry point that determines the appropriate parser based on file extension

## Protein Structure Parsing

### Parser Selection

- **For MMCIF/CIF/PDB formatted input**: Use `rcsb.py` as the main entry point
- **For CSV formatted input containing 3D coordinates**: Convert to PDB structure format first

### Extracted Information

Both parsers extract the following information:
- Atom coordinates, elements, and properties
- Bonds between atoms
- Residue information
- Chain information
- Interface detection between chains
- Metadata (resolution, deposition date, etc.)

### Usage

The `rcsb.py` script automatically detects file formats based on extensions:
- `.pdb` for PDB files
- `.cif*` for MMCIF files

```bash
# Process a directory of structures
python -m boltz.scripts.process.rcsb --datadir /path/to/mmcif_or_pdb_files --outdir /path/to/output
```

## RNA Structure Parsing

The `pdb.py` module includes specialized functionality for parsing RNA structures from various sources.

### Standard PDB Files
- The `parse_pdb()` function can handle RNA structures in standard PDB format
- RNA chains are identified by their polymer type (Rna)
- Standard RNA residues (A, C, G, U) are processed automatically
- Non-standard RNA residues require entries in the components dictionary

### CSV Dataset
- The `convert_csv_to_pdb()` function specifically handles RNA structures from the Stanford RNA dataset
- It reads sequence and coordinate information from CSV files
- Coordinates are converted to PDB format for processing
- Supports both standard and non-standard RNA residues

### RNA-Specific Processing
- RNA structures are identified by their polymer type
- One-letter codes are used for sequence representation (A, C, G, U)
- Special handling for RNA-specific atoms and bonds
- Support for RNA modifications and non-standard bases

### Coordinate Handling
- RNA coordinates are typically provided as phosphate positions
- The `prepare_rna_from_csv()` function converts these to PDB format
- Missing coordinates are handled gracefully with appropriate error messages

### Error Handling
- Invalid RNA structures raise ValueError with descriptive messages
- Missing components are reported and skipped
- Coordinate parsing errors are caught and reported

### Usage Example

To parse an RNA structure from the Stanford RNA dataset:

```python
from boltz.scripts.process.pdb import convert_csv_to_pdb

# Parse an RNA structure
rna_structure = convert_csv_to_pdb(
    rna_id="1SCL_A",
    components=components_dict,
    csv_path="/path/to/train_labels.csv",
    seq_csv_path="/path/to/train_sequences.csv"
)
```