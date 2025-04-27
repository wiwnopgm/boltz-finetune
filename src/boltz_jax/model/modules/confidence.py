"""JAX implementation of the confidence module for Boltz."""

from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from boltz_jax.model.modules.trunk import InputEmbedder, MSAModule, PairformerModule


class ConfidenceModule(nn.Module):
    """Confidence prediction module for structure quality assessment.
    
    Predicts per-residue and per-pair confidence scores for protein structures.
    """
    
    token_s: int
    token_z: int
    hidden_dim: int = 128
    compute_pae: bool = True
    imitate_trunk: bool = False
    pairformer_args: Optional[Dict] = None
    full_embedder_args: Optional[Dict] = None
    msa_args: Optional[Dict] = None
    num_layers: int = 4
    dropout_rate: float = 0.1
    use_s_diffusion: bool = False
    
    def setup(self):
        """Initialize the confidence module components."""
        # If imitating trunk, we need to create similar structure as main trunk
        if self.imitate_trunk:
            assert self.pairformer_args is not None
            assert self.full_embedder_args is not None
            
            # Create embedder
            self.input_embedder = InputEmbedder(**self.full_embedder_args)
            
            # Create MSA module if needed
            if self.msa_args is not None:
                self.msa_module = MSAModule(
                    token_z=self.token_z,
                    s_input_dim=self.full_embedder_args.get("token_s", 384) + 2 * 21 + 1 + 4,
                    **self.msa_args,
                )
            
            # Create pairformer module
            self.pairformer = PairformerModule(
                token_s=self.token_s,
                token_z=self.token_z,
                **self.pairformer_args,
            )
        else:
            # Simpler model that takes pre-computed token representations
            # and atom coordinates
            
            # Process token and pair representations
            self.token_proj = nn.Dense(features=self.hidden_dim)
            self.pair_proj = nn.Dense(features=self.hidden_dim)
            
            # Process atom coordinates if available
            self.coord_embedder = nn.Dense(features=self.hidden_dim)
            
            # Attention layers for confidence prediction
            self.layers = [ConfidenceLayer(
                hidden_dim=self.hidden_dim, 
                dropout_rate=self.dropout_rate,
            ) for _ in range(self.num_layers)]
        
        # Output heads
        self.plddt_head = nn.Sequential([
            nn.LayerNorm(),
            nn.Dense(features=self.hidden_dim),
            nn.relu,
            nn.Dense(features=1),
            nn.sigmoid,
        ])
        
        # PAE prediction if needed
        if self.compute_pae:
            self.pae_head = nn.Sequential([
                nn.LayerNorm(),
                nn.Dense(features=self.hidden_dim),
                nn.relu,
                nn.Dense(features=1),
                nn.sigmoid,
            ])
    
    def __call__(
        self,
        s: Optional[jnp.ndarray] = None,
        z: Optional[jnp.ndarray] = None,
        coords: Optional[jnp.ndarray] = None,
        feats: Optional[Dict[str, jnp.ndarray]] = None,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Predict confidence scores for protein structure.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            coords: Atom coordinates [B, N, 3]
            feats: Dictionary of features
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Dictionary with confidence scores
        """
        if self.imitate_trunk:
            return self._forward_imitate_trunk(feats, train, rngs)
        else:
            assert s is not None and z is not None and coords is not None
            return self._forward_simple(s, z, coords, feats, train, rngs)
    
    def _forward_imitate_trunk(
        self,
        feats: Dict[str, jnp.ndarray],
        train: bool,
        rngs: Optional[Dict[str, jnp.ndarray]],
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass with trunk imitation.
        
        Args:
            feats: Dictionary of features
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Dictionary with confidence scores
        """
        batch_size = feats["aatype"].shape[0]
        seq_len = feats["aatype"].shape[1]
        
        # Initialize representations
        s_input_dim = (
            self.token_s + 2 * 21 + 1 + 4
        )
        s = jnp.zeros((batch_size, seq_len, self.token_s))
        z = jnp.zeros((batch_size, seq_len, seq_len, self.token_z))
        
        # Apply input embedder
        s, z = self.input_embedder(
            s=s,
            z=z,
            feats=feats,
            train=train,
            rngs=rngs,
        )
        
        # Apply MSA module if needed
        if hasattr(self, "msa_module") and "msa_tokens" in feats:
            msa_rngs = None
            if rngs is not None and "dropout" in rngs:
                msa_rngs = {"dropout": rngs["dropout"]}
            
            s, z = self.msa_module(
                s=s,
                z=z,
                msa_tokens=feats["msa_tokens"],
                train=train,
                rngs=msa_rngs,
            )
        
        # Apply pairformer
        pairformer_rngs = None
        if rngs is not None and "dropout" in rngs:
            pairformer_rngs = {"dropout": rngs["dropout"]}
        
        s, z = self.pairformer(
            s=s,
            z=z,
            feats=feats,
            train=train,
            rngs=pairformer_rngs,
        )
        
        # Predict pLDDT
        plddt = self.plddt_head(s)
        
        # Predict PAE if needed
        if self.compute_pae:
            pae = self.pae_head(z)
        else:
            pae = None
        
        # Prepare output
        out = {"plddt": plddt}
        if pae is not None:
            out["pae"] = pae
        
        return out
    
    def _forward_simple(
        self,
        s: jnp.ndarray,
        z: jnp.ndarray,
        coords: jnp.ndarray,
        feats: Optional[Dict[str, jnp.ndarray]],
        train: bool,
        rngs: Optional[Dict[str, jnp.ndarray]],
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass with simple architecture.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            coords: Atom coordinates [B, N, 3]
            feats: Dictionary of features
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Dictionary with confidence scores
        """
        batch_size, seq_len = s.shape[:2]
        
        # Process token representations
        token_feats = self.token_proj(s)
        
        # Process coordinates if they exist and we have residue indices
        if feats is not None and "residue_index" in feats:
            # Filter to CA atoms only
            if "atom_types" in feats:
                atom_types = feats["atom_types"]
                ca_mask = (atom_types == 1)  # Assuming CA is type 1
            else:
                # Just use the first atom per residue
                residue_index = feats["residue_index"]
                ca_mask = jnp.zeros_like(residue_index, dtype=bool)
                for i in range(seq_len):
                    # Mark first atom of each residue
                    res_mask = (residue_index == i)
                    first_atom_idx = jnp.argmax(res_mask.astype(jnp.int32), axis=1)
                    for b in range(batch_size):
                        if jnp.any(res_mask[b]):
                            ca_mask = ca_mask.at[b, first_atom_idx[b]].set(True)
            
            # Extract CA coords
            ca_coords = coords * ca_mask[:, :, None]
            
            # Embed coordinates
            coord_feats = self.coord_embedder(ca_coords)
            
            # Add coordinate features to token features based on residue index
            residue_index = feats["residue_index"]
            for b in range(batch_size):
                for n in range(coords.shape[1]):
                    if ca_mask[b, n]:
                        res_idx = residue_index[b, n]
                        if 0 <= res_idx < seq_len:
                            token_feats = token_feats.at[b, res_idx].add(coord_feats[b, n])
        
        # Process through confidence layers
        for layer in self.layers:
            token_feats = layer(
                token_feats,
                train=train,
                rngs=rngs,
            )
        
        # Predict pLDDT
        plddt = self.plddt_head(token_feats)
        
        # Prepare output
        out = {"plddt": plddt}
        
        # Predict PAE if needed
        if self.compute_pae:
            # Process pair representations
            pair_feats = self.pair_proj(z)
            
            # Combine with token features
            token_i = token_feats[:, :, None, :]  # [B, L, 1, H]
            token_j = token_feats[:, None, :, :]  # [B, 1, L, H]
            
            # Simple combination
            combined_feats = pair_feats + token_i + token_j
            
            # Predict PAE
            pae = self.pae_head(combined_feats)
            out["pae"] = pae
        
        return out


class ConfidenceLayer(nn.Module):
    """Single layer for confidence prediction.
    
    Processes token representations for confidence prediction.
    """
    
    hidden_dim: int
    dropout_rate: float = 0.1
    num_heads: int = 8
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Process token representations.
        
        Args:
            x: Token representations [B, L, H]
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Updated token representations [B, L, H]
        """
        # Self-attention
        y = nn.LayerNorm()(x)
        
        dropout_rng = None
        if rngs is not None and "dropout" in rngs:
            dropout_rng = rngs["dropout"]
        
        y = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.hidden_dim,
            dropout_rate=self.dropout_rate if train else 0.0,
        )(
            inputs_q=y,
            inputs_kv=y,
            deterministic=not train,
            rngs={"dropout": dropout_rng} if dropout_rng is not None else None,
        )
        x = x + y
        
        # FFN
        y = nn.LayerNorm()(x)
        y = nn.Sequential([
            nn.Dense(features=self.hidden_dim * 4),
            nn.relu,
            nn.Dropout(rate=self.dropout_rate if train else 0.0),
            nn.Dense(features=self.hidden_dim),
        ])(
            y, 
            deterministic=not train,
            rngs={"dropout": dropout_rng} if dropout_rng is not None else None,
        )
        x = x + y
        
        return x 