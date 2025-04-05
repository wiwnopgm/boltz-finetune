<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz_title.png" width="400"/>

## Boltz Fine-tuning

This repository extends the capabilities of Boltz-1, the state-of-the-art open-source model for biomolecular structure prediction. For the original Boltz-1 model and its capabilities, please refer to the [original repository](https://github.com/jwohlwend/boltz).

## Extended Capabilities

Our fine-tuning pipeline for protein-ligand complex prediction provides:
- High performance [5th place](https://polarishub.io/competitions/asap-discovery/antiviral-drug-discovery-2025#competiton-results) in anti-viral ligand pose prediction challenge
- Robust molecule processing with automatic fixing of problematic ligand structures
- Integration with advanced docking tools for enhanced binding site prediction
- Optimized modeling of protein-ligand interactions with improved accuracy

### RNA-Specialized Boltz
We've enhanced Boltz with dedicated RNA structure prediction capabilities through a specialized RNA MSA Module. This extension offers:
- Improved RNA structure prediction through optimized RNA-specific MSA generation
- Enhanced handling of RNA-specific structural features and tertiary interactions
- Superior performance on RNA-protein and RNA-ligand complex predictions
- Specialized processing pipeline for RNA structures in various formats

## Installation
Install the extended capabilities with:

```
git clone https://github.com/wiwnopgm/boltz-finetune.git
cd boltz-finetune
pip install -e .
```
> Note: we recommend installing in a fresh python environment

## Inference

For RNA-specialized prediction:

```
boltz predict input_path --use_msa_server --rna_mode
```

For optimized protein-ligand complex prediction with the fine-tuned model:

```
boltz predict input_path --use_msa_server --protein_ligand
```

## Training

### Full fine-tuning
Currently, the repository supports full fine-tuning of the model. Parameter-efficient fine-tuning methods like LoRA are currently work in progress (WIP).

We provide dedicated training pipelines for specialized tasks:

1. **RNA Structure Prediction**
   - Utilizes our RNA MSA Module for optimized RNA feature extraction
   - Specialized data processing for RNA structures using `rna_process.py`
   - Custom training configurations optimized for RNA folding

2. **Protein-Ligand Complex Fine-tuning**
   - End-to-end fine-tuning framework for protein-ligand interaction prediction
   - Robust pre-processing with automatic molecule error fixing
   - Integration with state-of-the-art docking tools
   - Solution ranked 5th in anti-viral ligand pose prediction challenge

For the fine-tuning instructions in details, see the explanations in the `boltz-finetune/docs/finetune.md`.

## Pipeline Setup and Usage

### Database Setup
Before running the pipeline, you need to set up the required databases:

1. Download and start the CCD database:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
redis-server --dbfilename ccd.rdb --port 7777
```

2. Download and start the Taxonomy database:
```bash
wget https://boltz1.s3.us-east-2.amazonaws.com/taxonomy.rdb
redis-server --dbfilename taxonomy.rdb --port 7778
```

### Running the Pipeline
Once both database servers are running, you can execute the pipeline:

```bash
cd boltz
python scripts/process/run_pipeline.py \
  --data_dir /path/to/pdb_or_mmcif_files \
  --msa_dir  /path/to/a3m_files \
  --output_dir /path/to/output
```

### Modular Configuration
- Use the reference `config.yaml` to specify:
  - Sampling parameters
  - Modeling parameters
  - Domain specialization mode (protein-ligand, protein-protein, or nucleotides)

For systematic hyperparameter optimization, you can run multiple training jobs in parallel using SLURM job arrays. Use the template script `scripts/train/slurm_scripts/parallel_run_finetune_template.sbatch` to explore different parameter combinations:

```bash
# Configure your parameter combinations
export PARAM1_VALUES="0.3 0.5 0.7"  # e.g., train_binder_pocket_conditioned_prop
export PARAM2_VALUES="1 2 4"        # e.g., batch_size
export PARAM3_VALUES="0.001 0.0018 0.002"  # e.g., learning_rate

# Set parameter paths in config
export PARAM1_PATH="data.train_binder_pocket_conditioned_prop"
export PARAM2_PATH="data.batch_size"
export PARAM3_PATH="model.training_args.max_lr"

# Run parallel jobs
sbatch scripts/train/slurm_scripts/parallel_run_finetune_template.sbatch
```

Each job will create a separate output directory with the parameter values in the name, allowing you to compare results across different configurations.

### Optimized Training
The pipeline includes optimizations for faster training performance.
- Triton's Kernel Optimization (WIP)

### Prediction Analysis
*Coming soon: Detailed documentation for prediction analysis scripts*

## Citations

Please cite both the original Boltz-1 paper and our extensions:

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