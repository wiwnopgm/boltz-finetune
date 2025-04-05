from __future__ import annotations

from math import sqrt
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from boltz.model.modules.diffusion import DiffusionModule
from boltz.model.layers.attention import AttentionPairBias


class LoRALinear(nn.Module):
    """LoRA-adapted linear layer"""
    
    def __init__(
        self,
        linear_layer,
        rank=8,
        alpha=16,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        
        self.linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Freeze the original weights
        for param in self.linear.parameters():
            param.requires_grad = False
            
        # Create LoRA matrices
        in_features = self.linear.in_features
        out_features = self.linear.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Add bias if needed
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        
    def forward(self, x):
        # Original linear layer
        base_output = self.linear(x)
        
        # LoRA path
        lora_output = (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        
        # Add bias if present
        if self.bias is not None:
            lora_output = lora_output + self.bias
            
        return base_output + lora_output


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