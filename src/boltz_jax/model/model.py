"""JAX implementation of the Boltz1 model."""

from typing import Any, Dict, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training import train_state

from boltz_jax.data import const
from boltz_jax.model.modules.confidence import ConfidenceModule
from boltz_jax.model.modules.diffusion import AtomDiffusion
from boltz_jax.model.modules.encoders import RelativePositionEncoder
from boltz_jax.model.modules.trunk import (
    DistogramModule,
    InputEmbedder,
    MSAModule,
    PairformerModule,
)


class Boltz1(nn.Module):
    """JAX implementation of the Boltz1 model.
    
    This is the main model for protein structure prediction, adapted from the 
    PyTorch Lightning implementation to use JAX/Flax.
    """
    
    atom_s: int
    atom_z: int
    token_s: int
    token_z: int
    num_bins: int
    training_args: Dict[str, Any]
    validation_args: Dict[str, Any]
    embedder_args: Dict[str, Any]
    msa_args: Dict[str, Any]
    pairformer_args: Dict[str, Any]
    score_model_args: Dict[str, Any]
    diffusion_process_args: Dict[str, Any]
    diffusion_loss_args: Dict[str, Any]
    confidence_model_args: Dict[str, Any]
    atom_feature_dim: int = 128
    confidence_prediction: bool = False
    confidence_imitate_trunk: bool = False
    alpha_pae: float = 0.0
    structure_prediction_training: bool = True
    atoms_per_window_queries: int = 32
    atoms_per_window_keys: int = 128
    nucleotide_rmsd_weight: float = 5.0
    ligand_rmsd_weight: float = 10.0
    no_msa: bool = False
    no_atom_encoder: bool = False
    min_dist: float = 2.0
    max_dist: float = 22.0
    predict_args: Optional[Dict[str, Any]] = None

    def setup(self):
        """Initialize the model components."""
        # Input projections
        s_input_dim = (
            self.token_s + 2 * const.num_tokens + 1 + len(const.pocket_contact_info)
        )
        self.s_init = nn.Dense(
            features=self.token_s, 
            use_bias=False, 
            name="s_init"
        )
        self.z_init_1 = nn.Dense(
            features=self.token_z, 
            use_bias=False, 
            name="z_init_1"
        )
        self.z_init_2 = nn.Dense(
            features=self.token_z, 
            use_bias=False, 
            name="z_init_2"
        )

        # Input embeddings
        full_embedder_args = {
            "atom_s": self.atom_s,
            "atom_z": self.atom_z,
            "token_s": self.token_s,
            "token_z": self.token_z,
            "atoms_per_window_queries": self.atoms_per_window_queries,
            "atoms_per_window_keys": self.atoms_per_window_keys,
            "atom_feature_dim": self.atom_feature_dim,
            "no_atom_encoder": self.no_atom_encoder,
            **self.embedder_args,
        }
        self.input_embedder = InputEmbedder(**full_embedder_args)
        self.rel_pos = RelativePositionEncoder(self.token_z)
        self.token_bonds = nn.Dense(
            features=self.token_z, 
            use_bias=False, 
            name="token_bonds"
        )

        # Normalization layers
        self.s_norm = nn.LayerNorm(name="s_norm")
        self.z_norm = nn.LayerNorm(name="z_norm")

        # Recycling projections
        self.s_recycle = nn.Dense(
            features=self.token_s, 
            use_bias=False, 
            name="s_recycle"
        )
        self.z_recycle = nn.Dense(
            features=self.token_z, 
            use_bias=False, 
            name="z_recycle"
        )
        # Note: JAX init is handled differently than PyTorch
        # The equivalent of gating_init_ will be implemented in a separate module

        # Pairwise stack
        self.no_msa = self.no_msa
        if not self.no_msa:
            self.msa_module = MSAModule(
                token_z=self.token_z,
                s_input_dim=s_input_dim,
                **self.msa_args,
            )
        self.pairformer_module = PairformerModule(
            self.token_s, 
            self.token_z, 
            **self.pairformer_args
        )

        # Output modules
        use_accumulate_token_repr = (
            self.confidence_prediction 
            and "use_s_diffusion" in self.confidence_model_args 
            and self.confidence_model_args["use_s_diffusion"]
        )
        self.structure_module = AtomDiffusion(
            score_model_args={
                "token_z": self.token_z,
                "token_s": self.token_s,
                "atom_z": self.atom_z,
                "atom_s": self.atom_s,
                "atoms_per_window_queries": self.atoms_per_window_queries,
                "atoms_per_window_keys": self.atoms_per_window_keys,
                "atom_feature_dim": self.atom_feature_dim,
                **self.score_model_args,
            },
            accumulate_token_repr=use_accumulate_token_repr,
            **self.diffusion_process_args,
        )
        self.distogram_module = DistogramModule(self.token_z, self.num_bins)
        
        if self.confidence_prediction:
            if self.confidence_imitate_trunk:
                self.confidence_module = ConfidenceModule(
                    self.token_s,
                    self.token_z,
                    compute_pae=self.alpha_pae > 0,
                    imitate_trunk=True,
                    pairformer_args=self.pairformer_args,
                    full_embedder_args=full_embedder_args,
                    msa_args=self.msa_args,
                    **self.confidence_model_args,
                )
            else:
                self.confidence_module = ConfidenceModule(
                    self.token_s,
                    self.token_z,
                    compute_pae=self.alpha_pae > 0,
                    **self.confidence_model_args,
                )

    def __call__(
        self,
        feats: Dict[str, jnp.ndarray],
        recycling_steps: int = 0,
        num_sampling_steps: Optional[int] = None,
        multiplicity_diffusion_train: int = 1,
        diffusion_samples: int = 1,
        run_confidence_sequentially: bool = False,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass of the model.
        
        Args:
            feats: Dictionary of input features
            recycling_steps: Number of recycling steps
            num_sampling_steps: Number of sampling steps for diffusion
            multiplicity_diffusion_train: Multiplicity for diffusion training
            diffusion_samples: Number of diffusion samples
            run_confidence_sequentially: Whether to run confidence sequentially
            train: Whether in training mode
            rngs: JAX PRNGKey for stochastic operations
            
        Returns:
            Dictionary of output features
        """
        # Process input features and initialize states
        out = {}
        
        # Prepare recycling features
        s_prev = jnp.zeros((feats["aatype"].shape[0], feats["aatype"].shape[1], self.token_s))
        z_prev = jnp.zeros((feats["aatype"].shape[0], feats["aatype"].shape[1], feats["aatype"].shape[1], self.token_z))
        
        # Recycling loop
        for recycle_idx in range(recycling_steps + 1):
            # Compute token representation with recycling
            token_repr = self._compute_token_repr(feats, s_prev, z_prev, train=train, rngs=rngs)
            s, z = token_repr["s"], token_repr["z"]
            
            # Store token representations
            out[f"s_{recycle_idx}"] = s
            out[f"z_{recycle_idx}"] = z
            
            # Final recycling update
            if recycle_idx < recycling_steps:
                s_prev = self.s_recycle(s)
                z_prev = self.z_recycle(z)
        
        # Use the final token representations
        s, z = out[f"s_{recycling_steps}"], out[f"z_{recycling_steps}"]
        
        # Compute distogram
        out["distogram_logits"] = self.distogram_module(z)
        
        # Structure prediction
        structure_out = self.structure_module(
            s=s, 
            z=z, 
            feats=feats,
            num_sampling_steps=num_sampling_steps,
            multiplicity=multiplicity_diffusion_train,
            num_samples=diffusion_samples,
            train=train,
            rngs=rngs,
        )
        out.update(structure_out)
        
        # Confidence prediction
        if self.confidence_prediction:
            confidence_rngs = None
            if rngs is not None:
                confidence_rngs = {"dropout": rngs["confidence_dropout"]}
                
            if self.confidence_imitate_trunk:
                confidence_out = self.confidence_module(
                    feats=feats,
                    train=train,
                    rngs=confidence_rngs,
                )
            else:
                confidence_out = self.confidence_module(
                    s=s,
                    z=z,
                    coords=out["coords"],
                    feats=feats,
                    train=train,
                    rngs=confidence_rngs,
                )
            out.update(confidence_out)
            
        return out
    
    def _compute_token_repr(
        self,
        feats: Dict[str, jnp.ndarray],
        s_prev: jnp.ndarray,
        z_prev: jnp.ndarray,
        train: bool = False,
        rngs: Optional[Dict[str, jnp.ndarray]] = None,
    ) -> Dict[str, jnp.ndarray]:
        """Compute token representations with recycling.
        
        Args:
            feats: Dictionary of input features
            s_prev: Previous single token representations
            z_prev: Previous pair token representations
            train: Whether in training mode
            rngs: JAX PRNGKey for stochastic operations
            
        Returns:
            Dictionary with token representations s and z
        """
        batch_size = feats["aatype"].shape[0]
        seq_len = feats["aatype"].shape[1]
        
        # Initialize input representations
        s = self.s_init(feats["tokens"])
        z1 = self.z_init_1(feats["tokens"])
        z2 = jnp.swapaxes(self.z_init_2(feats["tokens"]), 1, 2)
        z = z1[:, :, None, :] + z2[:, None, :, :]
        
        # Add relative position encoding
        z += self.rel_pos(seq_len)
        
        # Add previous representations
        s += s_prev
        z += z_prev
        
        # Apply input embedder
        s, z = self.input_embedder(
            s=s, 
            z=z, 
            feats=feats,
            train=train,
            rngs=rngs,
        )
        
        # Apply MSA module
        if not self.no_msa:
            msa_rngs = None
            if rngs is not None:
                msa_rngs = {"dropout": rngs["msa_dropout"]}
            s, z = self.msa_module(
                s=s,
                z=z, 
                msa_tokens=feats["msa_tokens"],
                train=train,
                rngs=msa_rngs,
            )
        
        # Apply pairformer module
        pairformer_rngs = None
        if rngs is not None:
            pairformer_rngs = {"dropout": rngs["pairformer_dropout"]}
        s, z = self.pairformer_module(
            s=s,
            z=z,
            feats=feats,
            train=train,
            rngs=pairformer_rngs,
        )
        
        # Normalize
        s = self.s_norm(s)
        z = self.z_norm(z)
        
        return {"s": s, "z": z}


class Boltz1TrainState(train_state.TrainState):
    """Train state for the Boltz1 model with EMA support.
    
    This class extends the Flax TrainState to include EMA parameters.
    """
    ema_params: Optional[Dict[str, Any]] = None
    ema_decay: float = 0.999
    
    def apply_gradients(self, *, grads, **kwargs):
        """Apply gradients and update EMA parameters."""
        next_state = super().apply_gradients(grads=grads, **kwargs)
        
        # Update EMA parameters if they exist
        if self.ema_params is not None:
            next_ema_params = jax.tree_map(
                lambda ema, param: self.ema_decay * ema + (1 - self.ema_decay) * param,
                self.ema_params,
                next_state.params,
            )
            return next_state.replace(ema_params=next_ema_params)
        
        return next_state
    
    def initialize_ema(self):
        """Initialize EMA parameters from current parameters."""
        return self.replace(ema_params=self.params) 