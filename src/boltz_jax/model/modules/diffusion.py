"""JAX implementation of the diffusion module for Boltz."""

from typing import Dict, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp


class ScoreModel(nn.Module):
    """Score model for diffusion-based structure prediction.
    
    Predicts denoising score for atom coordinates.
    """
    
    token_z: int
    token_s: int
    atom_z: int
    atom_s: int
    num_layers: int = 8
    dropout_rate: float = 0.1
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    atom_feature_dim: int = 128
    
    @nn.compact
    def __call__(
        self, 
        s: jnp.ndarray, 
        z: jnp.ndarray, 
        atom_pos: jnp.ndarray,
        atom_mask: jnp.ndarray,
        residue_index: jnp.ndarray,
        t: jnp.ndarray,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> jnp.ndarray:
        """Predict denoising score for atom coordinates.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            atom_pos: Atom positions [B, N, 3]
            atom_mask: Atom mask [B, N]
            residue_index: Residue indices for atoms [B, N]
            t: Diffusion timestep [B]
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Score prediction [B, N, 3]
        """
        batch_size, num_atoms = atom_pos.shape[:2]
        seq_len = s.shape[1]
        
        # Encode diffusion timestep
        t_encoding = self._get_timestep_embedding(t, self.atom_feature_dim)
        
        # Project atom positions
        atom_features = nn.Dense(features=self.atom_feature_dim)(atom_pos)
        
        # Add timestep embedding to each atom
        atom_features = atom_features + t_encoding[:, None, :]
        
        # Process with layers that integrate token/pair information
        for i in range(self.num_layers):
            # Apply atom self-attention
            atom_features = nn.LayerNorm()(atom_features)
            
            # Self-attention among atoms
            atom_attn = nn.MultiHeadAttention(
                num_heads=8,
                qkv_features=self.atom_feature_dim,
                dropout_rate=self.dropout_rate if train else 0.0,
            )(
                inputs_q=atom_features,
                inputs_kv=atom_features,
                mask=atom_mask[:, None, None, :],  # [B, 1, 1, N]
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            atom_features = atom_features + atom_attn
            
            # MLP
            atom_features = nn.LayerNorm()(atom_features)
            atom_mlp = nn.Sequential([
                nn.Dense(features=self.atom_feature_dim * 4),
                nn.relu,
                nn.Dropout(rate=self.dropout_rate if train else 0.0),
                nn.Dense(features=self.atom_feature_dim),
            ])(
                atom_features, 
                deterministic=not train,
                rngs={"dropout": rngs["dropout"]} if rngs is not None else None,
            )
            atom_features = atom_features + atom_mlp
            
            # Exchange information with token representations
            atom_features_norm = nn.LayerNorm()(atom_features)
            
            # Fetch token features for each atom based on residue index
            # Note: This is a simplified version, in a real implementation
            # you would use a more sophisticated attention mechanism
            token_features = jnp.zeros((batch_size, num_atoms, self.token_s))
            
            # For each batch and atom, get the corresponding token features
            for b in range(batch_size):
                for n in range(num_atoms):
                    res_idx = residue_index[b, n]
                    if 0 <= res_idx < seq_len:
                        token_features = token_features.at[b, n].set(s[b, res_idx])
            
            # Project token features to atom dimension
            token_features_proj = nn.Dense(features=self.atom_feature_dim)(token_features)
            
            # Add to atom features
            atom_features = atom_features + token_features_proj
            
            # Get pair features for neighboring residues
            # Again, simplified version
            pair_features = jnp.zeros((batch_size, num_atoms, num_atoms, self.token_z))
            
            # For each batch and atom pair, get corresponding pair features
            # This is obviously computationally expensive in Python,
            # but in JAX it would be optimized or implemented differently
            for b in range(batch_size):
                for n1 in range(num_atoms):
                    res_idx1 = residue_index[b, n1]
                    if 0 <= res_idx1 < seq_len:
                        for n2 in range(num_atoms):
                            res_idx2 = residue_index[b, n2]
                            if 0 <= res_idx2 < seq_len:
                                pair_features = pair_features.at[b, n1, n2].set(
                                    z[b, res_idx1, res_idx2]
                                )
            
            # Project and pool pair features
            pair_features_flat = pair_features.reshape(
                batch_size * num_atoms, num_atoms, self.token_z
            )
            pair_mask = jnp.tile(atom_mask[:, None, :], (1, num_atoms, 1))
            pair_mask_flat = pair_mask.reshape(batch_size * num_atoms, num_atoms)
            
            # Simple aggregation with masking
            pair_features_summed = jnp.sum(
                pair_features_flat * pair_mask_flat[:, :, None], axis=1
            )
            pair_features_summed = pair_features_summed.reshape(
                batch_size, num_atoms, self.token_z
            )
            
            # Project to atom dimension
            pair_features_proj = nn.Dense(features=self.atom_feature_dim)(pair_features_summed)
            
            # Add to atom features
            atom_features = atom_features + pair_features_proj
        
        # Final projection to 3D coordinate corrections
        atom_features = nn.LayerNorm()(atom_features)
        score_pred = nn.Dense(features=3)(atom_features)
        
        # Apply mask to the output
        score_pred = score_pred * atom_mask[:, :, None]
        
        return score_pred
    
    def _get_timestep_embedding(self, t: jnp.ndarray, dim: int) -> jnp.ndarray:
        """Compute sinusoidal timestep embeddings.
        
        Args:
            t: Timestep tensor [B]
            dim: Output dimension
            
        Returns:
            Timestep embedding [B, dim]
        """
        half_dim = dim // 2
        emb = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
        
        if dim % 2 == 1:  # Zero pad if needed
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
            
        return emb


class AtomDiffusion(nn.Module):
    """Atom diffusion module for structure prediction.
    
    Implements diffusion-based atom coordinate generation.
    """
    
    score_model_args: Dict
    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8.0
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    accumulate_token_repr: bool = False
    
    def setup(self):
        """Initialize the diffusion model components."""
        self.score_model = ScoreModel(**self.score_model_args)
    
    def __call__(
        self,
        s: jnp.ndarray,
        z: jnp.ndarray,
        feats: Dict[str, jnp.ndarray],
        num_sampling_steps: Optional[int] = None,
        multiplicity: int = 1,
        num_samples: int = 1,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Run the diffusion process for structure prediction.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            feats: Dictionary of features
            num_sampling_steps: Number of sampling steps
            multiplicity: Multiplicity for training
            num_samples: Number of diffusion samples
            train: Whether in training mode
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Dictionary with diffusion outputs
        """
        batch_size = s.shape[0]
        
        # Extract features
        atom_pos = feats.get("atom_pos", None)
        atom_mask = feats.get("atom_mask", None)
        residue_index = feats.get("residue_index", None)
        
        # Handle training vs. inference
        if train:
            return self._train_forward(
                s=s,
                z=z,
                atom_pos=atom_pos,
                atom_mask=atom_mask,
                residue_index=residue_index,
                multiplicity=multiplicity,
                rngs=rngs,
            )
        else:
            return self._sample(
                s=s,
                z=z,
                atom_mask=atom_mask,
                residue_index=residue_index,
                num_steps=num_sampling_steps or 200,
                num_samples=num_samples,
                rngs=rngs,
            )
    
    def _train_forward(
        self,
        s: jnp.ndarray,
        z: jnp.ndarray,
        atom_pos: jnp.ndarray,
        atom_mask: jnp.ndarray,
        residue_index: jnp.ndarray,
        multiplicity: int = 1,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass during training.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            atom_pos: Atom positions [B, N, 3]
            atom_mask: Atom mask [B, N]
            residue_index: Residue indices for atoms [B, N]
            multiplicity: Multiplicity for training
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Dictionary with training outputs and loss
        """
        batch_size = s.shape[0]
        
        # Sample noise level (timestep)
        if rngs is not None and "diffusion" in rngs:
            t_rng, noise_rng = jax.random.split(rngs["diffusion"])
        else:
            t_rng = jax.random.PRNGKey(0)
            noise_rng = jax.random.PRNGKey(1)
        
        # Sample log-normal timestep
        log_t = jax.random.normal(t_rng, (batch_size,)) * self.P_std + self.P_mean
        t = jnp.exp(log_t)
        
        # Compute diffusion variance
        sigma = self._t_to_sigma(t)
        
        # Sample noise
        noise = jax.random.normal(
            noise_rng, shape=atom_pos.shape
        ) * self.noise_scale
        
        # Create noisy positions
        noisy_pos = atom_pos + sigma[:, None, None] * noise
        
        # Predict score
        score_pred = self.score_model(
            s=s,
            z=z,
            atom_pos=noisy_pos,
            atom_mask=atom_mask,
            residue_index=residue_index,
            t=t,
            train=train,
            rngs=rngs,
        )
        
        # Compute loss (simplified, in practice would use proper weighting)
        loss = jnp.sum(
            jnp.square(score_pred - (-noise / sigma[:, None, None])) * atom_mask[:, :, None]
        ) / jnp.sum(atom_mask)
        
        return {
            "loss": loss,
            "coords": atom_pos,
            "noisy_coords": noisy_pos,
            "score_pred": score_pred,
        }
    
    def _sample(
        self,
        s: jnp.ndarray,
        z: jnp.ndarray,
        atom_mask: jnp.ndarray,
        residue_index: jnp.ndarray,
        num_steps: int = 200,
        num_samples: int = 1,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Sample atom coordinates using diffusion process.
        
        Args:
            s: Token representations [B, L, d_s]
            z: Pair representations [B, L, L, d_z]
            atom_mask: Atom mask [B, N]
            residue_index: Residue indices for atoms [B, N]
            num_steps: Number of sampling steps
            num_samples: Number of diffusion samples
            rngs: Optional dict of PRNGKeys
            
        Returns:
            Dictionary with sampled coordinates
        """
        batch_size = s.shape[0]
        num_atoms = atom_mask.shape[1]
        
        # Initialize random positions based on a high variance
        if rngs is not None and "diffusion" in rngs:
            rng = rngs["diffusion"]
        else:
            rng = jax.random.PRNGKey(0)
        
        # Sample initial noisy positions
        pos = jax.random.normal(
            rng, shape=(batch_size, num_atoms, 3)
        ) * self.sigma_max
        
        # Apply mask
        pos = pos * atom_mask[:, :, None]
        
        # Setup timesteps (from sigma_max to sigma_min)
        sigmas = self._get_sampling_timesteps(num_steps)
        
        # Sample using DDPM-like algorithm
        for i in range(num_steps):
            sigma = sigmas[i]
            
            # Convert sigma to timestep
            t = self._sigma_to_t(jnp.ones((batch_size,)) * sigma)
            
            # Predict score
            score = self.score_model(
                s=s,
                z=z,
                atom_pos=pos,
                atom_mask=atom_mask,
                residue_index=residue_index,
                t=t,
                train=False,
                rngs=rngs,
            )
            
            # Get next sigma
            next_sigma = sigmas[i + 1] if i < num_steps - 1 else 0.0
            
            # Update positions using score
            step_size = self.step_scale * (sigma - next_sigma)
            pos = pos + step_size * score
            
            # Add noise for the next step (except the last one)
            if i < num_steps - 1:
                noise = jax.random.normal(
                    jax.random.fold_in(rng, i),
                    shape=pos.shape,
                )
                pos = pos + jnp.sqrt(step_size) * noise * atom_mask[:, :, None]
        
        return {
            "coords": pos,
            "final_score": score,
        }
    
    def _t_to_sigma(self, t: jnp.ndarray) -> jnp.ndarray:
        """Convert diffusion timestep to noise standard deviation.
        
        Args:
            t: Diffusion timestep [B]
            
        Returns:
            Noise standard deviation [B]
        """
        log_sigma_max = jnp.log(self.sigma_max)
        log_sigma_min = jnp.log(self.sigma_min)
        
        # Scale t to log sigma
        log_sigma = log_sigma_max + (log_sigma_min - log_sigma_max) * (
            1.0 - 2.0 * jax.nn.sigmoid(-self.gamma_0 - self.gamma_min * jnp.log(t))
        )
        
        return jnp.exp(log_sigma)
    
    def _sigma_to_t(self, sigma: jnp.ndarray) -> jnp.ndarray:
        """Convert noise standard deviation to diffusion timestep.
        
        Args:
            sigma: Noise standard deviation [B]
            
        Returns:
            Diffusion timestep [B]
        """
        log_sigma_max = jnp.log(self.sigma_max)
        log_sigma_min = jnp.log(self.sigma_min)
        log_sigma = jnp.log(sigma)
        
        # Inverse of _t_to_sigma
        x = (log_sigma - log_sigma_max) / (log_sigma_min - log_sigma_max) - 1.0
        x = x / 2.0
        
        # Inverse of sigmoid
        y = -jnp.log(1.0 / x - 1.0)
        
        # Solve for t
        t = jnp.exp(-(y + self.gamma_0) / self.gamma_min)
        
        return t
    
    def _get_sampling_timesteps(self, num_steps: int) -> jnp.ndarray:
        """Get sequence of sigma values for sampling.
        
        Args:
            num_steps: Number of sampling steps
            
        Returns:
            Array of sigma values from max to min, plus zero [num_steps+1]
        """
        # For simplicity, using a linear spacing in log space
        log_sigma_max = jnp.log(self.sigma_max)
        log_sigma_min = jnp.log(self.sigma_min)
        
        log_sigmas = jnp.linspace(log_sigma_max, log_sigma_min, num_steps)
        sigmas = jnp.exp(log_sigmas)
        
        # Add final sigma=0
        sigmas = jnp.concatenate([sigmas, jnp.array([0.0])])
        
        return sigmas 