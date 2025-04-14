# Boltz Fine-tuning

This repository focuses on extending the capabilities of Boltz-1, the state-of-the-art open-source model for biomolecular structure prediction. For the original Boltz-1 model and its capabilities, please refer to the [original repository](https://github.com/jwohlwend/boltz).

## Installation

To install the extended capabilities, run:

```bash
git clone https://github.com/wiwnopgm/boltz-finetune.git
cd boltz-finetune
pip install -e .
```
> Note: We strongly recommend installing in a fresh Python environment to avoid dependency conflicts.

## Command Line Interface

This extension adds convenient command-line interface (CLI) commands for training, fine-tuning, and inference:

```bash
# Train a model
boltz train --data_dir /path/to/structures --msa_dir /path/to/msas --output_dir ./training_output

# Fine-tune a model (LoRA by default)
boltz finetune --data_dir /path/to/structures --msa_dir /path/to/msas --model_path /path/to/model.ckpt

# Run predictions
boltz predict /path/to/input.fasta --out_dir ./results
```

For complete documentation of all CLI options, run:
```bash
boltz train --help
boltz finetune --help
boltz predict --help
```

See the detailed CLI documentation in [docs/cli.md](docs/cli.md) for more information.

## Extended Capabilities

This extension to Boltz-1 include enhanced training architectures and specialized modules for RNA structure prediction. The following sections detail the key features and usage instructions.

### Fine-tuning Pipeline

Working with 3D molecular structures is challenging, as training data preparation for PDB structures and their Multiple Sequence Alignments (MSA) consists of multiple stages. To streamline this pre-processing step, we have created a unified pipeline that simplifies the process to just specifying paths to your PDB and MSA raw data.

#### Dataset Preparation

1. Download and start the Chemical Component Dictionary (CCD) database:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
redis-server --dbfilename ccd.rdb --port 7777
```

2. Download and start the Taxonomy database:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/taxonomy.rdb
redis-server --dbfilename taxonomy.rdb --port 7778
```

3. Prepare your input files:
   - PDB or mmCIF/CIF files containing 3D complex structures
   - MSA files: pre-computed alignments can be generated using `run_mmseqs2`

#### Data Processing

Use our unified processing script by specifying the paths to all necessary inputs:

```bash
python scripts/process/run_pipeline.py \
  --data_dir /path/to/pdb_or_mmcif_files \
  --msa_dir  /path/to/a3m_files \
  --output_dir /path/to/output
```

### RNA-Specific Capabilities

We have enhanced the model with specialized RNA processing capabilities:

- Custom MSA module with RNA-specific feature extraction
- Advanced processing of RNA structural features and tertiary interactions

### Inference

The model supports multiple inference modes:

```python
# Standard inference
boltz predict input_path --use_msa_server

# Inference using LoRA fine-tuned model
boltz predict input_path --use_msa_server --lora_weights path/to/weights

# RNA structure prediction
boltz predict input_path --use_msa_server --rna_mode
```

### Hyperparameter Optimization

For systematic hyperparameter optimization, we support parallel training using SLURM job arrays. Use our template script `scripts/train/slurm_scripts/parallel_run_finetune_template.sbatch`:

```bash
# Configure parameter combinations
export PARAM1_VALUES="0.3 0.5 0.7"  # e.g., train_binder_pocket_conditioned_prop
export PARAM2_VALUES="1 2 4"        # e.g., batch_size
export PARAM3_VALUES="0.001 0.0018 0.002"  # e.g., learning_rate

# Set parameter paths in config
export PARAM1_PATH="data.train_binder_pocket_conditioned_prop"
export PARAM2_PATH="data.batch_size"
export PARAM3_PATH="model.training_args.max_lr"

# Launch parallel jobs in slurm cluster
sbatch scripts/train/slurm_scripts/parallel_run_finetune_template.sbatch
```

Each job creates a separate output directory with parameter values in the name for easy result comparison.

### Performance Optimizations

The pipeline includes several optimizations for enhanced training performance:
- Triton Kernel Optimization (Work in Progress)
- Additional optimizations coming soon

### Analysis Tools

*Coming soon: Comprehensive documentation for prediction analysis tools and visualization scripts*

## Real-World Applications

This fine-tuning pipeline has demonstrated its effectiveness in real-world applications:
- Achieved 5th place out of 80+ submissions in the [anti-viral ligand pose challenge](https://polarishub.io/competitions/asap-discovery/antiviral-drug-discovery-2025#competiton-results)

## Citations

If you use this work, please cite both the original Boltz-1 paper and our fine-tuning extensions:

```bibtex
@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}

@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}

@article{boltz_finetuning,
  title={Extended Boltz: RNA-Specialized Structure Prediction and Ligand Pose Optimization},
  author={[Your Name]},
  journal={[Journal/Preprint]},
  year={2024}
}
```
