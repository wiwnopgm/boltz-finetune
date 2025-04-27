# Boltz-JAX

This is a JAX/Flax implementation of the Boltz protein structure prediction model.

## Overview

Boltz-JAX reimplements the PyTorch-based Boltz model using JAX and Flax for improved performance and hardware acceleration. The implementation maintains the same core architecture and capabilities as the original Boltz model while leveraging JAX's automatic differentiation, compilation, and parallel computing capabilities.

## Features

- Complete JAX/Flax reimplementation of Boltz model
- Support for the same diffusion-based protein structure prediction
- MSA processing and integration
- Confidence prediction (pLDDT and PAE)
- XLA compilation for accelerated performance

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/boltz-finetune.git
cd boltz-finetune

# Install requirements
pip install -e .
```

## Usage

The API is designed to be similar to the original Boltz implementation:

```bash
# Predict protein structure from a FASTA file
python -m boltz_jax.main predict input.fasta --out_dir ./output
```

## Implementation Details

The JAX implementation includes the following components:

- **Model Architecture**: Maintains the same transformer-based architecture with token and pair representations.
- **Diffusion Process**: Uses the same diffusion-based structure generation process.
- **Performance Optimizations**: Leverages JAX's JIT compilation and vectorization for improved performance.

## Key Components

- `model/model.py`: Main model architecture
- `model/modules/trunk.py`: Core network components (InputEmbedder, MSAModule, PairformerModule)
- `model/modules/diffusion.py`: Diffusion-based structure prediction
- `model/modules/confidence.py`: Confidence prediction
- `model/modules/encoders.py`: Position and feature encoders

## Requirements

- JAX
- Flax
- NumPy
- Click

## Differences from PyTorch Version

- Uses Flax's `nn.Module` instead of PyTorch Lightning's `LightningModule`
- Differently structured training and evaluation loops
- XLA compilation for improved performance
- Array manipulation more compatible with JAX's functional style

## License

This project is licensed under the same license as the original Boltz repository. 