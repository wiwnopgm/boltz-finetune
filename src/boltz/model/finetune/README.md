# LoRA Fine-tuning for Diffusion Module

This directory contains implementations for fine-tuning the diffusion module using LoRA (Low-Rank Adaptation).

## Overview

LoRA is a parameter-efficient fine-tuning method that adds small trainable rank decomposition matrices to existing weights while keeping the original weights frozen. This approach significantly reduces the number of trainable parameters and memory requirements during fine-tuning.

## Implementation Details

The implementation consists of the following components:

1. `LoRALinear`: A LoRA-adapted linear layer that adds low-rank matrices to the original linear layer.
2. `LoRAAttentionPairBias`: A LoRA-adapted attention mechanism that adds low-rank matrices to the query, key, and value projections.
3. `LoRADiffusionModule`: A LoRA-adapted diffusion module that applies LoRA to key components of the diffusion module.

## Usage

### Basic Usage

```python
from boltz.model.modules.lora import LoRADiffusionModule
from boltz.model.modules.diffusion import DiffusionModule

# Load pretrained model
pretrained_model = DiffusionModule(...)

# Create LoRA-adapted model
lora_model = LoRADiffusionModule(
    # Same parameters as DiffusionModule
    ...,
    # LoRA-specific parameters
    lora_rank=8,
    lora_alpha=16,
    lora_dropout=0.0,
)

# Copy weights from pretrained model
lora_model.load_state_dict(pretrained_model.state_dict(), strict=False)

# Train only LoRA parameters
optimizer = optim.AdamW(
    [p for n, p in lora_model.named_parameters() if "lora" in n],
    lr=1e-4,
)

# Save LoRA weights
lora_model.save_lora_weights("lora_weights.pt")

# Load LoRA weights
lora_model.load_lora_weights("lora_weights.pt")
```

### Using the Fine-tuning Script

The `lora_finetune.py` script provides a complete workflow for fine-tuning the diffusion module using LoRA:

```bash
python -m boltz.model.finetune.lora_finetune \
    --token_s 384 \
    --token_z 128 \
    --atom_s 384 \
    --atom_z 128 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.0 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 10 \
    --pretrained_model_path /path/to/pretrained/model.pt \
    --save_lora_path lora_weights.pt
```

## LoRA Parameters

- `lora_rank`: The rank of the low-rank matrices. Higher rank allows for more expressiveness but increases the number of parameters.
- `lora_alpha`: The scaling factor for the LoRA matrices. Typically set to `2 * lora_rank`.
- `lora_dropout`: The dropout rate for the LoRA matrices.

## Advantages of LoRA Fine-tuning

1. **Parameter Efficiency**: LoRA adds only a small number of trainable parameters compared to full fine-tuning.
2. **Memory Efficiency**: Since most of the model weights are frozen, the memory requirements during training are significantly reduced.
3. **Modularity**: LoRA weights can be saved and loaded separately, allowing for easy switching between different fine-tuned versions.
4. **Performance**: LoRA fine-tuning often achieves comparable or better performance compared to full fine-tuning.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Parameter-Efficient Fine-Tuning for Large Language Models](https://arxiv.org/abs/2203.15556) 