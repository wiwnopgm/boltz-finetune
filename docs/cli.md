# Boltz CLI Documentation

The Boltz package provides a command-line interface (CLI) with several commands for running different operations on the Boltz-1 model. This document provides an overview of the available commands and their options.

## Installation

Ensure the Boltz package is properly installed:

```bash
pip install -e .
```

Once installed, the `boltz` command will be available in your terminal.

## Prerequisites

Before using the Boltz CLI, ensure that:

1. The CCD Redis server is running on the specified port (default: 7777)
2. The taxonomy Redis server is running on the specified port (default: 7778)

You can check if these servers are running by attempting to connect to them:

```bash
redis-cli -h localhost -p 7777 ping
redis-cli -h localhost -p 7778 ping
```

If these commands return "PONG", the servers are running correctly.

## Available Commands

### 1. `boltz process_data`

Process input data for Boltz model training or fine-tuning.

```bash
boltz process_data /path/to/structures --msa_dir /path/to/msas --out_dir ./processed_data
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--msa_dir` | Directory containing MSA files | (Required) |
| `--out_dir` | Directory for processed outputs | `./boltz_processed` |
| `--redis_host` | Redis host for CCD and taxonomy servers | localhost |
| `--ccd_port` | Port for CCD Redis server | 7777 |
| `--taxonomy_port` | Port for taxonomy Redis server | 7778 |
| `--num_processes` | Number of processes to use for data processing | 4 |
| `--max_seqs` | Maximum number of sequences to process in MSA | 1000 |

For full details, run:
```bash
boltz process_data --help
```

### 2. `boltz predict`

Run predictions with the Boltz-1 model on input sequences.

```bash
boltz predict /path/to/input.fasta --out_dir ./results
```

For full options, run:
```bash
boltz predict --help
```

### 3. `boltz train`

Train a Boltz-1 model using the provided data.

```bash
boltz train --data_dir /path/to/processed_data --output_dir ./training_output
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Directory containing processed data | (Required) |
| `--output_dir` | Directory for model checkpoints and logs | `./boltz_checkpoints` |
| `--max_epochs` | Maximum number of epochs for training | 100 |
| `--batch_size` | Batch size for training | 32 |
| `--learning_rate` | Learning rate for training | 1e-4 |
| `--method` | Training method: 'lora' for LoRA training or 'full' for full model training | full |
| `--rank` | Rank for LoRA training (only used if method='lora') | 8 |
| `--alpha` | Alpha parameter for LoRA training (only used if method='lora') | 16.0 |

For full details, run:
```bash
boltz train --help
```

### 4. `boltz finetune`

Fine-tune a pre-trained Boltz-1 model using the provided data.

```bash
boltz finetune --data_dir /path/to/processed_data --pretrained_path /path/to/model.ckpt --output_dir ./finetuning_output
```

#### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--data_dir` | Directory containing processed data | (Required) |
| `--pretrained_path` | Path to the pretrained model to fine-tune | (Required) |
| `--output_dir` | Directory for model checkpoints and logs | `./boltz_finetuned` |
| `--method` | Fine-tuning method: LoRA or full model fine-tuning | lora |
| `--rank` | Rank for LoRA fine-tuning (only used if method is 'lora') | 8 |
| `--alpha` | Alpha value for LoRA fine-tuning (only used if method is 'lora') | 16.0 |
| `--batch_size` | Batch size for fine-tuning | 16 |
| `--learning_rate` | Learning rate for fine-tuning | 1e-5 |
| `--max_epochs` | Number of epochs for fine-tuning | 50 |

For full details, run:
```bash
boltz finetune --help
```

## Examples

Example shell scripts for running these commands can be found in the `examples/` directory:

- `examples/process_example.sh`: Example for data processing
- `examples/train_example.sh`: Example for training
- `examples/finetune_example.sh`: Example for fine-tuning

## Data Requirements

### Structure Data

The `--data_dir` should contain protein structure files in PDB or CIF format.

### MSA Data

The `--msa_dir` should contain multiple sequence alignment (MSA) files. These can be in A3M or CSV format.

## Workflow

The typical workflow for using Boltz is:

1. Process your data using the `process_data` command
2. Train a new model using the `train` command, or fine-tune an existing model using the `finetune` command
3. Run predictions using the `predict` command

## Customizing Training with a Configuration File

You can provide a custom YAML configuration file with the `--config_file` option. This allows for more fine-grained control over the training process.

Example configuration:

```yaml
trainer:
  accelerator: gpu
  devices: 1
  precision: 32
  gradient_clip_val: 10.0
  max_epochs: 10
  accumulate_grad_batches: 128

wandb:
  name: boltz_training_run
  project: boltz-training
  entity: your-entity

output: /path/to/output
pretrained: /path/to/pretrained/model.pth

data:
  datasets:
    - _target_: boltz.data.module.training.DatasetConfig
      target_dir: /path/to/processed/targets
      msa_dir: /path/to/msas
      prob: 1.0
      sampler:
        _target_: boltz.data.sample.cluster.ClusterSampler
      cropper:
        _target_: boltz.data.crop.boltz.BoltzCropper
        min_neighborhood: 0
        max_neighborhood: 40
``` 