"""JAX implementation of various encoder modules for Boltz."""

from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class RelativePositionEncoder(nn.Module):
    """Relative position encoder for pair representations.

    This module computes relative positional encodings for pair representations
    based on the relative positions of residues in the sequence.
    """

    token_z: int

    def __call__(self, seq_len: int) -> jnp.ndarray:
        """Compute relative positional encodings.

        Args:
            seq_len: Sequence length

        Returns:
            Relative positional encodings of shape [1, L, L, d_pair]
        """
        # Create positional indices
        positions = jnp.arange(seq_len)
        rel_pos = positions[:, None] - positions[None, :]
        
        # Compute relative position features (similar to Transformer sinusoidal encodings)
        max_rel_pos = 32
        rel_pos = jnp.clip(rel_pos, -max_rel_pos, max_rel_pos)
        
        # Normalize to [0, 1]
        rel_pos = (rel_pos + max_rel_pos) / (2 * max_rel_pos)
        
        # Create frequency bands
        num_bands = self.token_z // 2
        band_freqs = jnp.arange(1, num_bands + 1).astype(jnp.float32)
        
        # Compute sinusoidal encodings
        rel_pos_expanded = rel_pos[:, :, None]
        band_freqs = band_freqs[None, None, :]
        
        # Compute sines and cosines
        sins = jnp.sin(2 * jnp.pi * rel_pos_expanded * band_freqs)
        coss = jnp.cos(2 * jnp.pi * rel_pos_expanded * band_freqs)
        
        # Interleave sines and cosines
        pos_encodings = jnp.zeros((seq_len, seq_len, self.token_z), dtype=jnp.float32)
        pos_encodings = pos_encodings.at[:, :, 0::2].set(sins)
        pos_encodings = pos_encodings.at[:, :, 1::2].set(coss)
        
        # Add batch dimension
        pos_encodings = pos_encodings[None, :, :, :]
        
        return pos_encodings


class AtomFeatureEncoder(nn.Module):
    """Atom feature encoder.
    
    Encodes atom features including coordinates, atom types, and other properties.
    """
    
    feature_dim: int
    
    @nn.compact
    def __call__(
        self, 
        atom_pos: jnp.ndarray, 
        atom_mask: jnp.ndarray,
        atom_types: jnp.ndarray, 
        train: bool = False,
    ) -> jnp.ndarray:
        """Encode atom features.
        
        Args:
            atom_pos: Atom positions of shape [B, N, 3]
            atom_mask: Atom mask of shape [B, N]
            atom_types: Atom types of shape [B, N]
            train: Whether in training mode
            
        Returns:
            Atom features of shape [B, N, feature_dim]
        """
        batch_size, num_atoms = atom_pos.shape[:2]
        
        # Embed atom types
        atom_type_embedding = nn.Embed(
            num_embeddings=22,  # Number of atom types 
            features=self.feature_dim // 2
        )(atom_types)
        
        # Project atom positions
        pos_projection = nn.Dense(features=self.feature_dim // 2)(atom_pos)
        
        # Combine embeddings
        atom_features = jnp.concatenate([atom_type_embedding, pos_projection], axis=-1)
        
        # Apply final projection
        atom_features = nn.Dense(features=self.feature_dim)(atom_features)
        atom_features = nn.relu(atom_features)
        atom_features = nn.Dense(features=self.feature_dim)(atom_features)
        
        # Apply mask
        atom_features = atom_features * atom_mask[:, :, None]
        
        return atom_features


class MSAEncoder(nn.Module):
    """Encoder for Multiple Sequence Alignments (MSA).
    
    Processes MSA inputs to extract features for the structure prediction network.
    """
    
    hidden_dim: int
    num_heads: int = 4
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self, 
        msa_tokens: jnp.ndarray, 
        msa_mask: jnp.ndarray,
        train: bool = False,
        rngs: Optional[dict] = None,
    ) -> jnp.ndarray:
        """Encode MSA features.
        
        Args:
            msa_tokens: MSA token features of shape [B, N_seq, L, C]
            msa_mask: MSA mask of shape [B, N_seq, L]
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Sequence features of shape [B, L, hidden_dim]
        """
        batch_size, num_seqs, seq_len, _ = msa_tokens.shape
        
        # Reshape for processing each sequence
        flat_msa = msa_tokens.reshape(-1, seq_len, msa_tokens.shape[-1])
        flat_mask = msa_mask.reshape(-1, seq_len)
        
        # Project inputs
        msa_features = nn.Dense(features=self.hidden_dim)(flat_msa)
        
        # Apply attention layers
        for i in range(2):  # Simplified version with 2 layers
            # Self-attention
            attn_dropout = None if rngs is None else {"dropout": rngs.get("dropout")}
            
            # Layer norm
            msa_features = nn.LayerNorm()(msa_features)
            
            # Multi-head attention
            attn_output = nn.MultiHeadAttention(
                num_heads=self.num_heads,
                qkv_features=self.hidden_dim,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=msa_features,
                inputs_kv=msa_features,
                mask=flat_mask[:, None, None, :],  # [B*N_seq, 1, 1, L]
                deterministic=not train,
                rngs=attn_dropout,
            )
            
            # Residual connection
            msa_features = msa_features + attn_output
            
            # FFN
            msa_features = nn.LayerNorm()(msa_features)
            ffn_features = nn.Dense(features=self.hidden_dim * 4)(msa_features)
            ffn_features = nn.relu(ffn_features)
            ffn_features = nn.Dropout(
                rate=self.dropout_rate if train else 0.0
            )(ffn_features, deterministic=not train, rng=rngs.get("dropout") if rngs else None)
            ffn_output = nn.Dense(features=self.hidden_dim)(ffn_features)
            
            # Residual connection
            msa_features = msa_features + ffn_output
        
        # Reshape back and pool across sequences
        msa_features = msa_features.reshape(batch_size, num_seqs, seq_len, self.hidden_dim)
        msa_mask = msa_mask[:, :, :, None]  # [B, N_seq, L, 1]
        
        # Mean pooling across sequences
        pooled_features = jnp.sum(msa_features * msa_mask, axis=1) / (
            jnp.sum(msa_mask, axis=1) + 1e-8
        )
        
        return pooled_features