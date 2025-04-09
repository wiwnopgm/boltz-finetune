from __future__ import annotations

from math import sqrt
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any

from boltz.model.modules.diffusion import DiffusionModule
from boltz.model.layers.attention import AttentionPairBias


class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer that can be applied to any linear layer.
    
    This implementation follows the paper "LoRA: Low-Rank Adaptation of Large Language Models"
    by Hu et al. (2021).
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        bias: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the LoRA layer.
        
        Parameters
        ----------
        in_features : int
            The number of input features.
        out_features : int
            The number of output features.
        rank : int, optional
            The rank of the low-rank matrices, by default 8.
        alpha : float, optional
            The scaling factor for the LoRA weights, by default 16.
        dropout : float, optional
            The dropout probability, by default 0.0.
        bias : bool, optional
            Whether to include a bias term, by default False.
        device : Optional[torch.device], optional
            The device to use, by default None.
        dtype : Optional[torch.dtype], optional
            The data type to use, by default None.
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Initialize the low-rank matrices
        self.lora_A = nn.Parameter(
            torch.zeros((rank, in_features), device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((out_features, rank), device=device, dtype=dtype)
        )
        
        # Initialize weights using Kaiming initialization
        nn.init.kaiming_uniform_(self.lora_A, a=torch.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Optional dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Optional bias
        self.bias = nn.Parameter(
            torch.zeros(out_features, device=device, dtype=dtype)
        ) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRA layer.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            
        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        # Apply the low-rank transformation
        lora_output = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T
        lora_output = lora_output * self.scaling
        
        # Add bias if it exists
        if self.bias is not None:
            lora_output = lora_output + self.bias
            
        return lora_output
    
    def merge_weights(self, base_weight: torch.Tensor, base_bias: Optional[torch.Tensor] = None) -> tuple:
        """
        Merge the LoRA weights with the base weights.
        
        Parameters
        ----------
        base_weight : torch.Tensor
            The base weight matrix.
        base_bias : Optional[torch.Tensor], optional
            The base bias vector, by default None.
            
        Returns
        -------
        tuple
            The merged weight matrix and bias vector.
        """
        # Compute the merged weight
        merged_weight = base_weight + self.scaling * (self.lora_B @ self.lora_A)
        
        # Compute the merged bias
        merged_bias = base_bias + self.bias if self.bias is not None else base_bias
        
        return merged_weight, merged_bias


class LoRALinear(nn.Module):
    """
    A linear layer with LoRA adaptation.
    
    This module wraps a standard linear layer and adds LoRA adaptation.
    """
    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Initialize the LoRA linear layer.
        
        Parameters
        ----------
        linear_layer : nn.Linear
            The base linear layer to adapt.
        rank : int, optional
            The rank of the low-rank matrices, by default 8.
        alpha : float, optional
            The scaling factor for the LoRA weights, by default 16.
        dropout : float, optional
            The dropout probability, by default 0.0.
        device : Optional[torch.device], optional
            The device to use, by default None.
        dtype : Optional[torch.dtype], optional
            The data type to use, by default None.
        """
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            in_features=linear_layer.in_features,
            out_features=linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear_layer.bias is not None,
            device=device,
            dtype=dtype,
        )
        self.enabled = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LoRA linear layer.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor.
            
        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        # Apply the base linear layer
        base_output = self.linear(x)
        
        # Apply the LoRA adaptation if enabled
        if self.enabled:
            lora_output = self.lora(x)
            return base_output + lora_output
        
        return base_output
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge the LoRA weights with the base weights.
        
        Returns
        -------
        nn.Linear
            A new linear layer with merged weights.
        """
        # Get the merged weights and bias
        merged_weight, merged_bias = self.lora.merge_weights(
            self.linear.weight, self.linear.bias
        )
        
        # Create a new linear layer with the merged weights
        merged_linear = nn.Linear(
            self.linear.in_features,
            self.linear.out_features,
            bias=merged_bias is not None,
            device=merged_weight.device,
            dtype=merged_weight.dtype,
        )
        
        # Set the weights and bias
        merged_linear.weight.data = merged_weight
        if merged_bias is not None:
            merged_linear.bias.data = merged_bias
        
        return merged_linear


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    exclude_modules: Optional[List[str]] = None,
) -> Dict[str, LoRALinear]:
    """
    Apply LoRA adaptation to a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to adapt.
    rank : int, optional
        The rank of the low-rank matrices, by default 8.
    alpha : float, optional
        The scaling factor for the LoRA weights, by default 16.
    dropout : float, optional
        The dropout probability, by default 0.0.
    target_modules : Optional[List[str]], optional
        The names of the modules to adapt, by default None (all linear layers).
    exclude_modules : Optional[List[str]], optional
        The names of the modules to exclude, by default None.
        
    Returns
    -------
    Dict[str, LoRALinear]
        A dictionary mapping module names to LoRA linear layers.
    """
    if target_modules is None:
        target_modules = ["Linear"]
    
    if exclude_modules is None:
        exclude_modules = []
    
    lora_layers = {}
    
    # Iterate through all named modules
    for name, module in model.named_modules():
        # Skip if the module is in the exclude list
        if any(exclude in name for exclude in exclude_modules):
            continue
        
        # Check if the module is a target module
        if any(target in module.__class__.__name__ for target in target_modules):
            # Replace the module with a LoRA linear layer
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            else:
                parent = model
            
            # Create the LoRA linear layer
            lora_linear = LoRALinear(
                linear_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            
            # Replace the module
            setattr(parent, child_name, lora_linear)
            
            # Add to the dictionary
            lora_layers[name] = lora_linear
    
    return lora_layers


def enable_lora(model: nn.Module, enabled: bool = True) -> None:
    """
    Enable or disable LoRA adaptation for a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to enable or disable LoRA for.
    enabled : bool, optional
        Whether to enable LoRA, by default True.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.enabled = enabled


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights with base weights for a model.
    
    Parameters
    ----------
    model : nn.Module
        The model to merge LoRA weights for.
        
    Returns
    -------
    nn.Module
        The model with merged weights.
    """
    # Create a copy of the model
    merged_model = type(model)(model.__init_args__) if hasattr(model, "__init_args__") else model
    
    # Iterate through all named modules
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Get the parent module
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            
            if parent_name:
                parent = merged_model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            else:
                parent = merged_model
            
            # Merge the weights
            merged_linear = module.merge_weights()
            
            # Replace the module
            setattr(parent, child_name, merged_linear)
    
    return merged_model


class LoRAAttentionPairBias(nn.Module):
    """LoRA-adapted attention pair bias layer"""
    
    def __init__(
        self,
        attention_module,
        rank=8,
        alpha=16,
        dropout=0.0,
    ):
        super().__init__()
        
        self.attention = attention_module
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze the original weights
        for param in self.attention.parameters():
            param.requires_grad = False
            
        # Create LoRA matrices for query, key, value projections
        c_s = self.attention.c_s
        num_heads = self.attention.num_heads
        head_dim = c_s // num_heads
        
        # Create LoRA matrices for each projection
        self.lora_q_A = nn.Parameter(torch.zeros(c_s, rank))
        self.lora_q_B = nn.Parameter(torch.zeros(rank, c_s))
        self.lora_k_A = nn.Parameter(torch.zeros(c_s, rank))
        self.lora_k_B = nn.Parameter(torch.zeros(rank, c_s))
        self.lora_v_A = nn.Parameter(torch.zeros(c_s, rank))
        self.lora_v_B = nn.Parameter(torch.zeros(rank, c_s))
        
        # Initialize LoRA weights
        for lora_A in [self.lora_q_A, self.lora_k_A, self.lora_v_A]:
            nn.init.kaiming_uniform_(lora_A, a=sqrt(5))
        for lora_B in [self.lora_q_B, self.lora_k_B, self.lora_v_B]:
            nn.init.zeros_(lora_B)
            
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, s, z, mask, multiplicity=1, to_keys=None, model_cache=None):
        # Apply LoRA to query, key, value projections
        B = s.shape[0]
        
        # Layer norms (if any)
        if self.attention.initial_norm:
            s = self.attention.norm_s(s)
            
        # Handle to_keys if provided
        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s
            
        # Original projections
        q = self.attention.proj_q(s)
        k = self.attention.proj_k(k_in)
        v = self.attention.proj_v(k_in)
        
        # LoRA paths
        q_lora = (self.dropout(s) @ self.lora_q_A @ self.lora_q_B) * self.scaling
        k_lora = (self.dropout(k_in) @ self.lora_k_A @ self.lora_k_B) * self.scaling
        v_lora = (self.dropout(k_in) @ self.lora_v_A @ self.lora_v_B) * self.scaling
        
        # Combine original and LoRA projections
        q = q + q_lora
        k = k + k_lora
        v = v + v_lora
        
        # Reshape for multi-head attention
        q = q.view(B, -1, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.attention.num_heads, self.attention.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.attention.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, -self.attention.inf)
            
        # Apply softmax
        attn = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, -1, self.attention.c_s)
        
        # Apply output projection
        out = self.attention.proj_o(out)
        
        return out


class LoRADiffusionModule(DiffusionModule):
    """LoRA-adapted diffusion module"""
    
    def __init__(
        self,
        *args,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Apply LoRA to key components
        self._apply_lora()
        
    def _apply_lora(self):
        """Apply LoRA to key components of the diffusion module"""
        
        # Apply LoRA to the s_to_a_linear layer
        if hasattr(self, 's_to_a_linear') and len(self.s_to_a_linear) > 1:
            self.s_to_a_linear[1] = LoRALinear(
                self.s_to_a_linear[1],
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
            
        # Apply LoRA to the token transformer
        if hasattr(self, 'token_transformer'):
            # Apply LoRA to each transformer layer
            for i, layer in enumerate(self.token_transformer.layers):
                if hasattr(layer, 'attn'):
                    layer.attn = LoRAAttentionPairBias(
                        layer.attn,
                        rank=self.lora_rank,
                        alpha=self.lora_alpha,
                        dropout=self.lora_dropout,
                    )
                    
        # Apply LoRA to the atom attention encoder
        if hasattr(self, 'atom_attention_encoder'):
            # First collect all attention modules that need to be replaced
            attention_modules = {}
            for name, module in self.atom_attention_encoder.named_modules():
                if isinstance(module, AttentionPairBias):
                    attention_modules[name] = module
            
            # Then replace them
            for name, module in attention_modules.items():
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                if parent_name:
                    parent = self.atom_attention_encoder.get_submodule(parent_name)
                else:
                    parent = self.atom_attention_encoder
                setattr(parent, module_name, LoRAAttentionPairBias(
                    module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                ))
            
        # Apply LoRA to the atom attention decoder
        if hasattr(self, 'atom_attention_decoder'):
            # First collect all attention modules that need to be replaced
            attention_modules = {}
            for name, module in self.atom_attention_decoder.named_modules():
                if isinstance(module, AttentionPairBias):
                    attention_modules[name] = module
            
            # Then replace them
            for name, module in attention_modules.items():
                parent_name = '.'.join(name.split('.')[:-1])
                module_name = name.split('.')[-1]
                if parent_name:
                    parent = self.atom_attention_decoder.get_submodule(parent_name)
                else:
                    parent = self.atom_attention_decoder
                setattr(parent, module_name, LoRAAttentionPairBias(
                    module,
                    rank=self.lora_rank,
                    alpha=self.lora_alpha,
                    dropout=self.lora_dropout,
                ))
            
    def save_lora_weights(self, path):
        """Save only the LoRA weights"""
        lora_state_dict = {}
        
        # Collect LoRA weights from all adapted modules
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRAAttentionPairBias)):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A
                lora_state_dict[f"{name}.lora_B"] = module.lora_B
                if hasattr(module, 'bias') and module.bias is not None:
                    lora_state_dict[f"{name}.bias"] = module.bias
                    
        torch.save(lora_state_dict, path)
        
    def load_lora_weights(self, path):
        """Load only the LoRA weights"""
        lora_state_dict = torch.load(path)
        
        # Load LoRA weights into adapted modules
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRAAttentionPairBias)):
                if f"{name}.lora_A" in lora_state_dict:
                    module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state_dict:
                    module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
                if hasattr(module, 'bias') and module.bias is not None and f"{name}.bias" in lora_state_dict:
                    module.bias.data = lora_state_dict[f"{name}.bias"] 