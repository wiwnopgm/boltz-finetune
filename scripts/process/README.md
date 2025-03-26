Please see our [data processing instructions](../../docs/training.md).

## PDB and MMCIF File Parsing

This module includes parsers for both PDB and MMCIF file formats:

- `mmcif.py`: Parses MMCIF formatted files
- `pdb.py`: Parses PDB formatted files
- `rcsb.py`: Main entry point that determines the appropriate parser based on file extension

Both parsers extract the following information:
- Atom coordinates, elements, and properties
- Bonds between atoms
- Residue information
- Chain information
- Interface detection between chains
- Metadata (resolution, deposition date, etc.)

### Usage

To process structures:

```bash
python -m boltz.scripts.process.rcsb --datadir /path/to/pdb_files --outdir /path/to/output
```

The script will automatically detect whether files are in PDB or MMCIF format based on their extensions (.pdb for PDB files, .cif* for MMCIF files).

To test the PDB parser specifically:

```bash
python -m boltz.scripts.process.test_pdb_parser --pdb-file /path/to/example.pdb --output-dir ./output
```