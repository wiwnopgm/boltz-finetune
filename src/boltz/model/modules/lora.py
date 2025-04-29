

from __future__ import annotations

from math import sqrt
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import math

# LoRA implementation
class LoRALayer:
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class LoRALinear(nn.Linear, LoRALayer):
    
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        # 
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                self.merged = True       

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)            
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class LoRAEmbedding(nn.Embedding, LoRALayer):
    # LoRA implemented in an embedding layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)
            # use normal distribution for the weights in embedding layers

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                self.merged = True
        
    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class LoRAAttentionPairBias(nn.Module, LoRALayer):
    """LoRA-adapted attention pair bias layer"""
    
    def __init__(
        self,
        attention_module,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, 
                           lora_dropout=lora_dropout, merge_weights=merge_weights)
        
        # Store original attention module
        self.attention = attention_module
        
        # Copy necessary attributes from original module
        self.c_s = self.attention.c_s
        self.num_heads = self.attention.num_heads
        self.head_dim = self.c_s // self.num_heads
        self.inf = getattr(self.attention, 'inf', 1e9)
        
        # Freeze all original parameters
        for param in self.attention.parameters():
            param.requires_grad = False
        
        # Create a new trainable adapter (instead of using the frozen one)
        self.adapter_g = nn.Sequential(
            nn.Linear(self.c_s, self.c_s),
            nn.SiLU(),
            nn.Linear(self.c_s, self.c_s),
        )
        
        # Create LoRA wrapped versions of proj_g and proj_o
        # Keep original architecture but add LoRA fine-tuning capability
        if hasattr(self.attention, 'proj_g'):
            orig_proj_g = self.attention.proj_g
            self.proj_g = LoRALinear(
                in_features=orig_proj_g.in_features,
                out_features=orig_proj_g.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
                bias=orig_proj_g.bias is not None
            )
            # Copy the original weights
            with torch.no_grad():
                self.proj_g.weight.copy_(orig_proj_g.weight)
                if orig_proj_g.bias is not None:
                    self.proj_g.bias.copy_(orig_proj_g.bias)
        
        if hasattr(self.attention, 'proj_o'):
            orig_proj_o = self.attention.proj_o
            self.proj_o = LoRALinear(
                in_features=orig_proj_o.in_features,
                out_features=orig_proj_o.out_features,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
                bias=orig_proj_o.bias is not None
            )
            # Copy the original weights
            with torch.no_grad():
                self.proj_o.weight.copy_(orig_proj_o.weight)
                if orig_proj_o.bias is not None:
                    self.proj_o.bias.copy_(orig_proj_o.bias)
        
        # Create LoRA parameters for low-rank adaptation
        if r > 0:
            # Create LoRA matrices for query, key, value projections
            self.lora_q_A = nn.Parameter(torch.zeros(r, self.c_s))
            self.lora_q_B = nn.Parameter(torch.zeros(self.c_s, r))
            self.lora_k_A = nn.Parameter(torch.zeros(r, self.c_s))
            self.lora_k_B = nn.Parameter(torch.zeros(self.c_s, r))
            self.lora_v_A = nn.Parameter(torch.zeros(r, self.c_s))
            self.lora_v_B = nn.Parameter(torch.zeros(self.c_s, r))
            self.scaling = self.lora_alpha / self.r
            
            # Initialize LoRA parameters
            self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA weights for attention adapter"""
        if hasattr(self, 'lora_q_A'):
            # Initialize A matrices with Kaiming uniform
            for lora_A in [self.lora_q_A, self.lora_k_A, self.lora_v_A]:
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            # Initialize B matrices with zeros
            for lora_B in [self.lora_q_B, self.lora_k_B, self.lora_v_B]:
                nn.init.zeros_(lora_B)

    def train(self, mode: bool = True):
        """Handle train/eval mode switching with weight merging"""
        nn.Module.train(self, mode)
        # Always keep original attention in eval mode
        self.attention.eval()
        
        if mode:
            # Training mode: ensure weights are unmerged
            if self.merge_weights and self.merged:
                self._unmerge_weights()
                self.merged = False
        else:
            # Eval mode: merge weights if configured
            if self.merge_weights and not self.merged:
                self._merge_weights()
                self.merged = True
    
    def _merge_weights(self):
        """Merge LoRA weights into original weights for efficient inference"""
        # Add LoRA weights to the original weights
        self.attention.proj_q.weight.data += (self.lora_q_B @ self.lora_q_A) * self.scaling
        self.attention.proj_k.weight.data += (self.lora_k_B @ self.lora_k_A) * self.scaling
        self.attention.proj_v.weight.data += (self.lora_v_B @ self.lora_v_A) * self.scaling
    
    def _unmerge_weights(self):
        """Remove LoRA weights from original weights"""
        # Subtract the LoRA weights
        self.attention.proj_q.weight.data -= (self.lora_q_B @ self.lora_q_A) * self.scaling
        self.attention.proj_k.weight.data -= (self.lora_k_B @ self.lora_k_A) * self.scaling
        self.attention.proj_v.weight.data -= (self.lora_v_B @ self.lora_v_A) * self.scaling

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
    ) -> Tensor:
        """Forward pass with LoRA adapter for attention"""
        # If in eval mode with merged weights, use original attention directly
        if not self.training and self.merged:
            return self.attention(s, z, mask, multiplicity, to_keys, model_cache)
        
        # Get batch size
        B = s.shape[0]
        
        # Handle input normalization and key input
        if hasattr(self.attention, 'initial_norm') and self.attention.initial_norm:
            s = self.attention.norm_s(s)
            
        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s
        
        # Get projections from original module (without gradients)
        with torch.no_grad():
            q = self.attention.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
            k = self.attention.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
            v = self.attention.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)
        
        # Add LoRA contributions if rank > 0
        if self.r > 0:
            # Apply dropout for training
            s_drop = self.lora_dropout(s)
            k_in_drop = self.lora_dropout(k_in)
            
            # Compute LoRA contributions
            q_lora = (s_drop @ self.lora_q_A.t() @ self.lora_q_B.t()).view(B, -1, self.num_heads, self.head_dim) * self.scaling
            k_lora = (k_in_drop @ self.lora_k_A.t() @ self.lora_k_B.t()).view(B, -1, self.num_heads, self.head_dim) * self.scaling
            v_lora = (k_in_drop @ self.lora_v_A.t() @ self.lora_v_B.t()).view(B, -1, self.num_heads, self.head_dim) * self.scaling
            
            # Add LoRA contributions to base projections
            q = q + q_lora
            k = k + k_lora
            v = v + v_lora
        
        # Process Z tensor (pairwise bias)
        with torch.no_grad():
            if model_cache is None or "z" not in model_cache:
                z = self.attention.proj_z(z)
                if model_cache is not None:
                    model_cache["z"] = z
            else:
                z = model_cache["z"]
            
            # Repeat for multiplicity
            z = z.repeat_interleave(multiplicity, 0)
        
        # Process adapter and get gating factor
        s_adapter = self.adapter_g(s)
        g = self.proj_g(s_adapter).sigmoid()
        
        # Compute attention (in float32 for stability)
        with torch.autocast("cuda", enabled=False):
            # Compute attention scores
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5) + z.float()
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)
            
            # Apply attention to values
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        
        # Reshape output
        o = o.reshape(B, -1, self.c_s)
        
        # Apply gating and output projection with LoRA
        o = g * o
        o = self.proj_o(o)
        
        return o