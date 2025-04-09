import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union, Callable

from boltz.model.modules.lora import (
    LoRALinear,
    apply_lora_to_model,
    enable_lora,
    merge_lora_weights,
)


class FineTuningConfig:
    """
    Configuration for fine-tuning methods.
    """
    def __init__(
        self,
        method: str = "lora",
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        offload_to_cpu: bool = False,
    ):
        """
        Initialize the fine-tuning configuration.
        
        Parameters
        ----------
        method : str, optional
            The fine-tuning method to use, by default "lora".
        rank : int, optional
            The rank of the low-rank matrices for LoRA, by default 8.
        alpha : float, optional
            The scaling factor for the LoRA weights, by default 16.
        dropout : float, optional
            The dropout probability, by default 0.0.
        target_modules : Optional[List[str]], optional
            The names of the modules to adapt, by default None (all linear layers).
        exclude_modules : Optional[List[str]], optional
            The names of the modules to exclude, by default None.
        offload_to_cpu : bool, optional
            Whether to offload model to CPU during fine-tuning, by default False.
        """
        self.method = method
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["Linear"]
        self.exclude_modules = exclude_modules or []
        self.offload_to_cpu = offload_to_cpu


def prepare_model_for_fine_tuning(
    model: nn.Module,
    config: FineTuningConfig,
) -> Dict[str, Any]:
    """
    Prepare a model for fine-tuning using the specified method.
    
    Parameters
    ----------
    model : nn.Module
        The model to prepare for fine-tuning.
    config : FineTuningConfig
        The fine-tuning configuration.
        
    Returns
    -------
    Dict[str, Any]
        A dictionary containing information about the fine-tuning setup.
    """
    info = {
        "method": config.method,
        "offload_to_cpu": config.offload_to_cpu,
    }
    
    if config.method == "lora":
        # Apply LoRA to the model
        lora_layers = apply_lora_to_model(
            model=model,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
            target_modules=config.target_modules,
            exclude_modules=config.exclude_modules,
        )
        
        info["lora_layers"] = lora_layers
        info["rank"] = config.rank
        info["alpha"] = config.alpha
        info["dropout"] = config.dropout
        info["target_modules"] = config.target_modules
        info["exclude_modules"] = config.exclude_modules
    
    elif config.method == "full":
        # No special preparation needed for full fine-tuning
        pass
    
    else:
        raise ValueError(f"Unknown fine-tuning method: {config.method}")
    
    return info


def enable_fine_tuning(
    model: nn.Module,
    enabled: bool = True,
    offload_to_cpu: bool = False,
) -> None:
    """
    Enable or disable fine-tuning for a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to enable or disable fine-tuning for.
    enabled : bool, optional
        Whether to enable fine-tuning, by default True.
    offload_to_cpu : bool, optional
        Whether to offload model to CPU during fine-tuning, by default False.
    """
    # Enable or disable LoRA if it's being used
    enable_lora(model, enabled)
    
    # Offload to CPU if requested
    if offload_to_cpu and enabled:
        model.cpu()
    elif not enabled and offload_to_cpu:
        # Move back to the original device if we have it stored
        if hasattr(model, "_original_device"):
            model.to(model._original_device)
            delattr(model, "_original_device")
    elif offload_to_cpu and enabled:
        # Store the original device
        if not hasattr(model, "_original_device"):
            model._original_device = next(model.parameters()).device
        model.cpu()


def merge_fine_tuning_weights(
    model: nn.Module,
    method: str = "lora",
) -> nn.Module:
    """
    Merge fine-tuning weights with base weights for a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to merge fine-tuning weights for.
    method : str, optional
        The fine-tuning method to use, by default "lora".
        
    Returns
    -------
    nn.Module
        The model with merged weights.
    """
    if method == "lora":
        return merge_lora_weights(model)
    elif method == "full":
        # No merging needed for full fine-tuning
        return model
    else:
        raise ValueError(f"Unknown fine-tuning method: {method}")


def create_fine_tuning_optimizer(
    model: nn.Module,
    method: str = "lora",
    lr: float = 1e-3,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """
    Create an optimizer for fine-tuning a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to create an optimizer for.
    method : str, optional
        The fine-tuning method to use, by default "lora".
    lr : float, optional
        The learning rate, by default 1e-3.
    weight_decay : float, optional
        The weight decay, by default 0.0.
        
    Returns
    -------
    torch.optim.Optimizer
        The optimizer for fine-tuning.
    """
    if method == "lora":
        # Only optimize LoRA parameters
        lora_params = []
        for module in model.modules():
            if isinstance(module, LoRALinear):
                lora_params.extend(list(module.lora.parameters()))
        
        return torch.optim.AdamW(
            lora_params,
            lr=lr,
            weight_decay=weight_decay,
        )
    
    elif method == "full":
        # Optimize all parameters
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    
    else:
        raise ValueError(f"Unknown fine-tuning method: {method}")


def create_fine_tuning_scheduler(
    optimizer: torch.optim.Optimizer,
    method: str = "cosine",
    num_warmup_steps: int = 0,
    num_training_steps: int = 1000,
) -> Any:
    """
    Create a learning rate scheduler for fine-tuning.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer to create a scheduler for.
    method : str, optional
        The scheduler method to use, by default "cosine".
    num_warmup_steps : int, optional
        The number of warmup steps, by default 0.
    num_training_steps : int, optional
        The total number of training steps, by default 1000.
        
    Returns
    -------
    Any
        The learning rate scheduler.
    """
    try:
        from transformers import get_scheduler
    except ImportError:
        raise ImportError(
            "The transformers library is required for creating a fine-tuning scheduler. "
            "Please install it with `pip install transformers`."
        )
    
    return get_scheduler(
        name=method,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    ) 