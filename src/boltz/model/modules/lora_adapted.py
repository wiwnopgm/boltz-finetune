from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module
from typing import Optional, Dict, Any, List, Union

from boltz.model.modules.confidence import ConfidenceModule
from boltz.model.modules.diffusion import DiffusionModule
from boltz.model.modules.lora import (
    LoRALinear,
    LoRAAttentionPairBias,
    apply_lora_to_model,
    enable_lora,
    merge_lora_weights,
)


class LoRAConfidenceModule(ConfidenceModule):
    """
    LoRA-adapted version of the ConfidenceModule.
    
    This module applies LoRA adaptation to the ConfidenceModule,
    allowing for parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        *args,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the LoRA-adapted confidence module.
        
        Parameters
        ----------
        *args
            Arguments to pass to the parent ConfidenceModule.
        lora_rank : int, optional
            The rank of the low-rank matrices for LoRA, by default 8.
        lora_alpha : float, optional
            The scaling factor for the LoRA weights, by default 16.
        lora_dropout : float, optional
            The dropout probability, by default 0.0.
        target_modules : Optional[List[str]], optional
            The names of the modules to adapt, by default None (all linear layers).
        exclude_modules : Optional[List[str]], optional
            The names of the modules to exclude, by default None.
        **kwargs
            Keyword arguments to pass to the parent ConfidenceModule.
        """
        super().__init__(*args, **kwargs)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Apply LoRA to the model
        self.lora_layers = apply_lora_to_model(
            model=self,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
        )
    
    def enable_lora(self, enabled: bool = True) -> None:
        """
        Enable or disable LoRA adaptation.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to enable LoRA, by default True.
        """
        enable_lora(self, enabled)
    
    def merge_lora_weights(self) -> ConfidenceModule:
        """
        Merge LoRA weights with base weights.
        
        Returns
        -------
        ConfidenceModule
            The model with merged weights.
        """
        return merge_lora_weights(self)
    
    def save_lora_weights(self, path: str) -> None:
        """
        Save only the LoRA weights.
        
        Parameters
        ----------
        path : str
            The path to save the weights to.
        """
        lora_state_dict = {}
        
        # Collect LoRA weights from all adapted modules
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data
                lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data
                if module.lora.bias is not None:
                    lora_state_dict[f"{name}.bias"] = module.lora.bias.data
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str) -> None:
        """
        Load only the LoRA weights.
        
        Parameters
        ----------
        path : str
            The path to load the weights from.
        """
        lora_state_dict = torch.load(path)
        
        # Load LoRA weights into adapted modules
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                if f"{name}.lora_A" in lora_state_dict:
                    module.lora.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                if f"{name}.lora_B" in lora_state_dict:
                    module.lora.lora_B.data = lora_state_dict[f"{name}.lora_B"]
                if module.lora.bias is not None and f"{name}.bias" in lora_state_dict:
                    module.lora.bias.data = lora_state_dict[f"{name}.bias"]


class LoRADiffusionModule(DiffusionModule):
    """
    LoRA-adapted version of the DiffusionModule.
    
    This module applies LoRA adaptation to the DiffusionModule,
    allowing for parameter-efficient fine-tuning.
    """
    
    def __init__(
        self,
        *args,
        lora_rank: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.0,
        target_modules: Optional[List[str]] = None,
        exclude_modules: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize the LoRA-adapted diffusion module.
        
        Parameters
        ----------
        *args
            Arguments to pass to the parent DiffusionModule.
        lora_rank : int, optional
            The rank of the low-rank matrices for LoRA, by default 8.
        lora_alpha : float, optional
            The scaling factor for the LoRA weights, by default 16.
        lora_dropout : float, optional
            The dropout probability, by default 0.0.
        target_modules : Optional[List[str]], optional
            The names of the modules to adapt, by default None (all linear layers).
        exclude_modules : Optional[List[str]], optional
            The names of the modules to exclude, by default None.
        **kwargs
            Keyword arguments to pass to the parent DiffusionModule.
        """
        super().__init__(*args, **kwargs)
        
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Apply LoRA to the model
        self.lora_layers = apply_lora_to_model(
            model=self,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=target_modules,
            exclude_modules=exclude_modules,
        )
    
    def enable_lora(self, enabled: bool = True) -> None:
        """
        Enable or disable LoRA adaptation.
        
        Parameters
        ----------
        enabled : bool, optional
            Whether to enable LoRA, by default True.
        """
        enable_lora(self, enabled)
    
    def merge_lora_weights(self) -> DiffusionModule:
        """
        Merge LoRA weights with base weights.
        
        Returns
        -------
        DiffusionModule
            The model with merged weights.
        """
        return merge_lora_weights(self)
    
    def save_lora_weights(self, path: str) -> None:
        """
        Save only the LoRA weights.
        
        Parameters
        ----------
        path : str
            The path to save the weights to.
        """
        lora_state_dict = {}
        
        # Collect LoRA weights from all adapted modules
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRAAttentionPairBias)):
                if isinstance(module, LoRALinear):
                    lora_state_dict[f"{name}.lora_A"] = module.lora.lora_A.data
                    lora_state_dict[f"{name}.lora_B"] = module.lora.lora_B.data
                    if module.lora.bias is not None:
                        lora_state_dict[f"{name}.bias"] = module.lora.bias.data
                else:  # LoRAAttentionPairBias
                    lora_state_dict[f"{name}.lora_q_A"] = module.lora_q_A.data
                    lora_state_dict[f"{name}.lora_q_B"] = module.lora_q_B.data
                    lora_state_dict[f"{name}.lora_k_A"] = module.lora_k_A.data
                    lora_state_dict[f"{name}.lora_k_B"] = module.lora_k_B.data
                    lora_state_dict[f"{name}.lora_v_A"] = module.lora_v_A.data
                    lora_state_dict[f"{name}.lora_v_B"] = module.lora_v_B.data
        
        torch.save(lora_state_dict, path)
    
    def load_lora_weights(self, path: str) -> None:
        """
        Load only the LoRA weights.
        
        Parameters
        ----------
        path : str
            The path to load the weights from.
        """
        lora_state_dict = torch.load(path)
        
        # Load LoRA weights into adapted modules
        for name, module in self.named_modules():
            if isinstance(module, (LoRALinear, LoRAAttentionPairBias)):
                if isinstance(module, LoRALinear):
                    if f"{name}.lora_A" in lora_state_dict:
                        module.lora.lora_A.data = lora_state_dict[f"{name}.lora_A"]
                    if f"{name}.lora_B" in lora_state_dict:
                        module.lora.lora_B.data = lora_state_dict[f"{name}.lora_B"]
                    if module.lora.bias is not None and f"{name}.bias" in lora_state_dict:
                        module.lora.bias.data = lora_state_dict[f"{name}.bias"]
                else:  # LoRAAttentionPairBias
                    if f"{name}.lora_q_A" in lora_state_dict:
                        module.lora_q_A.data = lora_state_dict[f"{name}.lora_q_A"]
                    if f"{name}.lora_q_B" in lora_state_dict:
                        module.lora_q_B.data = lora_state_dict[f"{name}.lora_q_B"]
                    if f"{name}.lora_k_A" in lora_state_dict:
                        module.lora_k_A.data = lora_state_dict[f"{name}.lora_k_A"]
                    if f"{name}.lora_k_B" in lora_state_dict:
                        module.lora_k_B.data = lora_state_dict[f"{name}.lora_k_B"]
                    if f"{name}.lora_v_A" in lora_state_dict:
                        module.lora_v_A.data = lora_state_dict[f"{name}.lora_v_A"]
                    if f"{name}.lora_v_B" in lora_state_dict:
                        module.lora_v_B.data = lora_state_dict[f"{name}.lora_v_B"] 