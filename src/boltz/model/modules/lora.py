from __future__ import annotations

from math import sqrt
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

# from boltz.model.modules.diffusion import DiffusionModule

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
        self.num_heads = self.attention.num_heads
        self.head_dim = c_s // self.num_heads
        
        # Create LoRA matrices for each projection
        self.lora_q_A = nn.Parameter(torch.zeros(c_s, rank))
        self.lora_q_B = nn.Parameter(torch.zeros(rank, c_s))
        self.lora_k_A = nn.Parameter(torch.zeros(c_s, rank))
        self.lora_k_B = nn.Parameter(torch.zeros(rank, c_s))
        self.lora_v_A = nn.Parameter(torch.zeros(c_s, rank))
        self.lora_v_B = nn.Parameter(torch.zeros(rank, c_s))

        self.adapter_g = nn.Sequential(
            nn.Linear(c_s, c_s),
            nn.SiLU(),
            nn.Linear(c_s, c_s),
        )

        # Initialize LoRA weights
        for lora_A in [self.lora_q_A, self.lora_k_A, self.lora_v_A]:
            nn.init.kaiming_uniform_(lora_A, a=sqrt(5))
        for lora_B in [self.lora_q_B, self.lora_k_B, self.lora_v_B]:
            nn.init.zeros_(lora_B)
            
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        s: Tensor,
        z: Tensor,
        mask: Tensor,
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The mask tensor (B, N)
        multiplicity : int, optional
            The diffusion batch size, by default 1

        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = s.shape[0]

        # Layer norms
        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s
        
        # Compute projections
        q = self.attention.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.attention.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.attention.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim) # (B, N)

         # LoRA paths
        q_lora = (self.dropout(s) @ self.lora_q_A @ self.lora_q_B) * self.scaling
        k_lora = (self.dropout(k_in) @ self.lora_k_A @ self.lora_k_B) * self.scaling
        v_lora = (self.dropout(k_in) @ self.lora_v_A @ self.lora_v_B) * self.scaling
        
        # Combine original and LoRA projections
        q = q + q_lora
        k = k + k_lora
        v = v + v_lora

        # Fine-tuning using the adapter
        
        # Caching z projection during diffusion roll-out
        if model_cache is None or "z" not in model_cache:
            z = self.attention.proj_z(z)

            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        # (B * multiplicity, N, N, D)
        z = z.repeat_interleave(multiplicity, 0)

        s_adapter = self.adapter_g(s)
        g = self.attention.proj_g(s_adapter).sigmoid()

        # Autocast on transformer
        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            # similar to the scaled dot product attention

            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            # (B, num_head, N, N)
            attn = attn / (self.head_dim**0.5) + z.float() # (B, H,  N, N)
            attn = attn + (1 - mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)

            # Compute output
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        
        # (B, N, D)
        o = o.reshape(B, -1, self.c_s)
        o = self.proj_o(g * o)

        return o


# class LoRADiffusionModule(DiffusionModule):
#     """LoRA-adapted diffusion module"""
    
#     def __init__(
#         self,
#         *args,
#         lora_rank=8,
#         lora_alpha=16,
#         lora_dropout=0.0,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
        
#         self.lora_rank = lora_rank
#         self.lora_alpha = lora_alpha
#         self.lora_dropout = lora_dropout
        
#         # Apply LoRA to key components
#         self._apply_lora()
        
#     def _apply_lora(self):
#         """Apply LoRA to key components of the diffusion module"""
        
#         # Apply LoRA to the s_to_a_linear layer
#         if hasattr(self, 's_to_a_linear') and len(self.s_to_a_linear) > 1:
#             self.s_to_a_linear[1] = LoRALinear(
#                 self.s_to_a_linear[1],
#                 rank=self.lora_rank,
#                 alpha=self.lora_alpha,
#                 dropout=self.lora_dropout,
#             )
            
#         # Apply LoRA to the token transformer
#         if hasattr(self, 'token_transformer'):
#             # Apply LoRA to each transformer layer
#             for i, layer in enumerate(self.token_transformer.layers):
#                 if hasattr(layer, 'attn'):
#                     layer.attn = LoRAAttentionPairBias(
#                         layer.attn,
#                         rank=self.lora_rank,
#                         alpha=self.lora_alpha,
#                         dropout=self.lora_dropout,
#                     )
                    
#         # Apply LoRA to the atom attention encoder
#         if hasattr(self, 'atom_attention_encoder'):
#             # First collect all attention modules that need to be replaced
#             attention_modules = {}
#             for name, module in self.atom_attention_encoder.named_modules():
#                 if isinstance(module, AttentionPairBias):
#                     attention_modules[name] = module
            
#             # Then replace them
#             for name, module in attention_modules.items():
#                 parent_name = '.'.join(name.split('.')[:-1])
#                 module_name = name.split('.')[-1]
#                 if parent_name:
#                     parent = self.atom_attention_encoder.get_submodule(parent_name)
#                 else:
#                     parent = self.atom_attention_encoder
#                 setattr(parent, module_name, LoRAAttentionPairBias(
#                     module,
#                     rank=self.lora_rank,
#                     alpha=self.lora_alpha,
#                     dropout=self.lora_dropout,
#                 ))
            
#         # Apply LoRA to the atom attention decoder
#         if hasattr(self, 'atom_attention_decoder'):
#             # First collect all attention modules that need to be replaced
#             attention_modules = {}
#             for name, module in self.atom_attention_decoder.named_modules():
#                 if isinstance(module, AttentionPairBias):
#                     attention_modules[name] = module
            
#             # Then replace them
#             for name, module in attention_modules.items():
#                 parent_name = '.'.join(name.split('.')[:-1])
#                 module_name = name.split('.')[-1]
#                 if parent_name:
#                     parent = self.atom_attention_decoder.get_submodule(parent_name)
#                 else:
#                     parent = self.atom_attention_decoder
#                 setattr(parent, module_name, LoRAAttentionPairBias(
#                     module,
#                     rank=self.lora_rank,
#                     alpha=self.lora_alpha,
#                     dropout=self.lora_dropout,
#                 ))
            
#     def save_lora_weights(self, path):
#         """Save only the LoRA weights"""
#         lora_state_dict = {}
        
#         # Collect LoRA weights from all adapted modules
#         for name, module in self.named_modules():
#             if isinstance(module, (LoRALinear, LoRAAttentionPairBias)):
#                 lora_state_dict[f"{name}.lora_A"] = module.lora_A
#                 lora_state_dict[f"{name}.lora_B"] = module.lora_B
#                 if hasattr(module, 'bias') and module.bias is not None:
#                     lora_state_dict[f"{name}.bias"] = module.bias
                    
#         torch.save(lora_state_dict, path)
        
#     def load_lora_weights(self, path):
#         """Load only the LoRA weights"""
#         lora_state_dict = torch.load(path)
        
#         # Load LoRA weights into adapted modules
#         for name, module in self.named_modules():
#             if isinstance(module, (LoRALinear, LoRAAttentionPairBias)):
#                 if f"{name}.lora_A" in lora_state_dict:
#                     module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
#                 if f"{name}.lora_B" in lora_state_dict:
#                     module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
#                 if hasattr(module, 'bias') and module.bias is not None and f"{name}.bias" in lora_state_dict:
#                     module.bias.data = lora_state_dict[f"{name}.bias"] 