"""JAX implementation of the core trunk modules for Boltz."""

from typing import Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from boltz_jax.model.modules.encoders import AtomFeatureEncoder


class InputEmbedder(nn.Module):
    """Input embedder for token and pair representations.
    
    Processes input tokens and atom features into token-level (s) and 
    pair-level (z) representations.
    """
    
    atom_s: int
    atom_z: int
    token_s: int
    token_z: int
    num_layers: int = 3
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    atom_feature_dim: int = 128
    dropout_rate: float = 0.1
    no_atom_encoder: bool = False
    
    @nn.compact
    def __call__(
        self,
        s: jnp.ndarray,
        z: jnp.ndarray,
        feats: Dict[str, jnp.ndarray],
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process inputs to token and pair representations.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            feats: Dictionary of features
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Updated token and pair representations
        """
        # Process atom features if available and enabled
        if not self.no_atom_encoder and "atom_pos" in feats:
            atom_encoder = AtomFeatureEncoder(feature_dim=self.atom_feature_dim)
            atom_features = atom_encoder(
                atom_pos=feats["atom_pos"],
                atom_mask=feats["atom_mask"],
                atom_types=feats["atom_types"],
                train=train,
            )
            
            # Project atom features to token space
            atom_to_token = nn.Dense(features=self.token_s)(atom_features)
            
            # Add atom features to token representations using residue indices
            residue_indices = feats["residue_index"]
            
            # Simplified atom-to-token aggregation
            # In a more complete implementation, this would use attention or more 
            # sophisticated aggregation methods, and handle the atom_to_pair aggregation
            batch_size = s.shape[0]
            seq_len = s.shape[1]
            
            # Create mask for valid atom-residue associations
            atom_residue_mask = (residue_indices >= 0) & (residue_indices < seq_len)
            
            # Sum atom features per residue
            s_update = jnp.zeros_like(s)
            for b in range(batch_size):
                for i in range(seq_len):
                    # Find atoms for this residue
                    atom_mask = (residue_indices[b] == i) & atom_residue_mask[b]
                    if jnp.any(atom_mask):
                        # Sum atom features for this residue
                        s_update = s_update.at[b, i].add(
                            jnp.sum(atom_to_token[b] * atom_mask[:, None], axis=0)
                        )
            
            # Add to token representations
            s = s + s_update
        
        # Process with transformer-like layers
        for i in range(self.num_layers):
            # Token self-attention
            s = nn.LayerNorm()(s)
            s_attn = nn.MultiHeadAttention(
                num_heads=8,
                qkv_features=self.token_s,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=s,
                inputs_kv=s,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            s = s + s_attn
            
            # Token MLP
            s = nn.LayerNorm()(s)
            s_mlp = nn.Sequential([
                nn.Dense(features=self.token_s * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate if train else 0.0),
                nn.Dense(features=self.token_s),
            ])(
                s, 
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            s = s + s_mlp
            
            # Pair self-attention
            z = nn.LayerNorm()(z)
            # Handle 2D attention differently - we'll use 1D attention as approximation
            # A true axial attention would be more appropriate here
            batch_size, seq_len, _, _ = z.shape
            
            # Flatten last two dimensions
            z_flat = z.reshape(batch_size, seq_len * seq_len, self.token_z)
            
            # Apply 1D attention
            z_attn_flat = nn.MultiHeadAttention(
                num_heads=8,
                qkv_features=self.token_z,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=z_flat,
                inputs_kv=z_flat,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            
            # Reshape back
            z_attn = z_attn_flat.reshape(batch_size, seq_len, seq_len, self.token_z)
            z = z + z_attn
            
            # Pair MLP
            z = nn.LayerNorm()(z)
            # Apply MLP to each pair independently
            z_mlp = nn.Sequential([
                nn.Dense(features=self.token_z * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate if train else 0.0),
                nn.Dense(features=self.token_z),
            ])(
                z, 
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            z = z + z_mlp
            
            # Token-pair information exchange (outer product)
            s_norm = nn.LayerNorm()(s)
            z_norm = nn.LayerNorm()(z)
            
            # Project tokens
            s_proj = nn.Dense(features=self.token_z)(s_norm)
            
            # Outer product-like operation
            s_i = s_proj[:, :, None, :]  # [B, L, 1, d_z]
            s_j = s_proj[:, None, :, :]  # [B, 1, L, d_z]
            outer = s_i * s_j
            
            # Add to pair representations
            z = z + outer
            
            # Sum pair representations to update tokens
            z_sum = jnp.sum(z_norm, axis=2)  # [B, L, d_z]
            z_proj = nn.Dense(features=self.token_s)(z_sum)
            s = s + z_proj
        
        return s, z


class MSAModule(nn.Module):
    """Multiple Sequence Alignment (MSA) processing module.
    
    Processes MSA inputs to refine token and pair representations.
    """
    
    token_z: int
    s_input_dim: int
    num_layers: int = 4
    dropout_rate: float = 0.1
    num_heads: int = 8
    
    @nn.compact
    def __call__(
        self, 
        s: jnp.ndarray, 
        z: jnp.ndarray, 
        msa_tokens: jnp.ndarray,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process MSA inputs.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            msa_tokens: MSA tokens [B, N_seq, L, C]
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Updated token and pair representations
        """
        batch_size, num_seqs, seq_len, _ = msa_tokens.shape
        
        # Project MSA tokens to same dimension as s
        msa_features = nn.Dense(features=s.shape[-1])(msa_tokens)
        
        # Extract the first sequence (query sequence)
        query_seq = msa_features[:, 0]
        
        # Add to token representation
        s = s + query_seq
        
        # Process MSA with attention layers
        for i in range(self.num_layers):
            # Row attention (within sequences)
            msa_features_norm = nn.LayerNorm()(msa_features)
            
            # Process each sequence independently
            msa_flat = msa_features_norm.reshape(-1, seq_len, s.shape[-1])
            row_attn_flat = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=s.shape[-1],
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=msa_flat,
                inputs_kv=msa_flat,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            row_attn = row_attn_flat.reshape(batch_size, num_seqs, seq_len, s.shape[-1])
            msa_features = msa_features + row_attn
            
            # Column attention (across sequences for each position)
            msa_features_norm = nn.LayerNorm()(msa_features)
            
            # Transpose to make column attention easier
            msa_features_t = jnp.swapaxes(msa_features_norm, 1, 2)  # [B, L, N_seq, C]
            msa_flat = msa_features_t.reshape(-1, num_seqs, s.shape[-1])
            col_attn_flat = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=s.shape[-1],
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=msa_flat,
                inputs_kv=msa_flat,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            col_attn = col_attn_flat.reshape(batch_size, seq_len, num_seqs, s.shape[-1])
            col_attn = jnp.swapaxes(col_attn, 1, 2)  # [B, N_seq, L, C]
            msa_features = msa_features + col_attn
            
            # MLP
            msa_features_norm = nn.LayerNorm()(msa_features)
            msa_mlp = nn.Sequential([
                nn.Dense(features=s.shape[-1] * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate if train else 0.0),
                nn.Dense(features=s.shape[-1]),
            ])(
                msa_features_norm, 
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            msa_features = msa_features + msa_mlp
        
        # Extract updated query sequence
        updated_query = msa_features[:, 0]
        
        # Add to token representation
        s = s + updated_query
        
        # Generate pair features from MSA
        msa_features_norm = nn.LayerNorm()(msa_features)
        
        # Simple coevolution features (outer product of MSA features)
        # In practice, this would use more sophisticated methods
        # We create a simplified version here
        query_seq = msa_features_norm[:, 0]  # [B, L, C]
        
        # Project to pair dimension
        query_proj = nn.Dense(features=self.token_z)(query_seq)
        
        # Outer product
        query_i = query_proj[:, :, None, :]  # [B, L, 1, d_z]
        query_j = query_proj[:, None, :, :]  # [B, 1, L, d_z]
        pair_update = query_i * query_j
        
        # Add to pair representation
        z = z + pair_update
        
        return s, z


class PairformerModule(nn.Module):
    """Pairformer module for processing pair representations.
    
    Refines token and pair representations with transformer-like layers.
    """
    
    token_s: int
    token_z: int
    num_layers: int = 12
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self, 
        s: jnp.ndarray, 
        z: jnp.ndarray,
        feats: Dict[str, jnp.ndarray],
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process token and pair representations.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            feats: Dictionary of features
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Updated token and pair representations
        """
        batch_size, seq_len, _ = s.shape
        
        # Apply transformer-like layers
        for i in range(self.num_layers):
            # Mask for attention (if needed)
            attention_mask = None
            if "attention_mask" in feats:
                attention_mask = feats["attention_mask"]
            
            # Triangle attention for pairs
            z = nn.LayerNorm()(z)
            
            # Triangle attention: rows as queries, columns as keys/values
            # We'll implement a simplified version that uses 1D attention
            z_flat = z.reshape(batch_size * seq_len, seq_len, self.token_z)
            z_attn_row = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.token_z,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=z_flat,
                inputs_kv=z_flat,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            z_attn_row = z_attn_row.reshape(batch_size, seq_len, seq_len, self.token_z)
            z = z + z_attn_row
            
            # Triangle attention: columns as queries, rows as keys/values
            z = nn.LayerNorm()(z)
            z_trans = jnp.swapaxes(z, 1, 2)  # [B, L, L, d_z] -> [B, L, L, d_z]
            z_flat = z_trans.reshape(batch_size * seq_len, seq_len, self.token_z)
            z_attn_col = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.token_z,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=z_flat,
                inputs_kv=z_flat,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            z_attn_col = z_attn_col.reshape(batch_size, seq_len, seq_len, self.token_z)
            z_attn_col = jnp.swapaxes(z_attn_col, 1, 2)
            z = z + z_attn_col
            
            # Pair MLP
            z = nn.LayerNorm()(z)
            z_mlp = nn.Sequential([
                nn.Dense(features=self.token_z * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate if train else 0.0),
                nn.Dense(features=self.token_z),
            ])(
                z, 
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            z = z + z_mlp
            
            # Token self-attention
            s = nn.LayerNorm()(s)
            s_attn = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.token_s,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=s,
                inputs_kv=s,
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            s = s + s_attn
            
            # Token MLP
            s = nn.LayerNorm()(s)
            s_mlp = nn.Sequential([
                nn.Dense(features=self.token_s * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate if train else 0.0),
                nn.Dense(features=self.token_s),
            ])(
                s, 
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            s = s + s_mlp
            
            # Information exchange between token and pair representations
            s = nn.LayerNorm()(s)
            z = nn.LayerNorm()(z)
            
            # Pair to token attention
            # Sum over columns to get pair context
            z_ctx = jnp.sum(z, axis=2)  # [B, L, d_z]
            z_ctx_proj = nn.Dense(features=self.token_s)(z_ctx)
            s = s + z_ctx_proj
            
            # Token to pair updates
            s_i = nn.Dense(features=self.token_z)(s)
            s_j = nn.Dense(features=self.token_z)(s)
            
            # Outer product
            s_i = s_i[:, :, None, :]  # [B, L, 1, d_z]
            s_j = s_j[:, None, :, :]  # [B, 1, L, d_z]
            pair_update = s_i * s_j
            z = z + pair_update
        
        return s, z


class DistogramModule(nn.Module):
    """Distogram prediction module.
    
    Predicts distance distributions between residue pairs.
    """
    
    token_z: int
    num_bins: int
    
    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """Predict distogram from pair representations.
        
        Args:
            z: Pair representations [B, L, L, d_z]
            
        Returns:
            Distogram logits [B, L, L, num_bins]
        """
        # Project to intermediate representation
        z_dist = nn.Dense(features=self.token_z)(z)
        z_dist = nn.relu(z_dist)
        
        # Project to bins
        distogram_logits = nn.Dense(features=self.num_bins)(z_dist)
        
        return distogram_logits 