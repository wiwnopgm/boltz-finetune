from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

from boltz.data import const
from boltz.model.layers.attention import AttentionPairBias
from boltz.model.layers.dropout import get_dropout_mask
from boltz.model.layers.outer_product_mean import OuterProductMean
from boltz.model.layers.pair_averaging import PairWeightedAveraging
from boltz.model.layers.transition import Transition
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz.model.modules.encoders import AtomAttentionEncoder


# Define RNA-specific constants if not already defined in const
if not hasattr(const, "rna_num_tokens"):
    # 4 nucleotides (A, U, G, C) + special tokens
    setattr(const, "rna_num_tokens", 4)

# RNA nucleotide indices
RNA_A = 0
RNA_U = 1
RNA_G = 2
RNA_C = 3


class RNABaseConstraints(nn.Module):
    """Module that enforces RNA base-pairing constraints."""
    
    def __init__(self, temperature: float = 2.0):
        """Initialize RNA base-pairing constraints.
        
        Parameters
        ----------
        temperature : float
            Temperature parameter for softening the constraints.
        """
        super().__init__()
        self.temperature = temperature
        
    def forward(self, one_hot_seq: Tensor) -> Tensor:
        """Apply RNA base-pairing constraints.
        
        Parameters
        ----------
        one_hot_seq : Tensor
            One-hot encoded RNA sequence [batch, seq_len, 4]
            where indices are [A, U, G, C]
            
        Returns
        -------
        Tensor
            Base-pairing probability matrix [batch, seq_len, seq_len]
        """
        # Extract positions of each nucleotide
        A_pos = one_hot_seq[:, :, RNA_A].unsqueeze(-1)  # [batch, seq_len, 1]
        U_pos = one_hot_seq[:, :, RNA_U].unsqueeze(-1)  # [batch, seq_len, 1]
        G_pos = one_hot_seq[:, :, RNA_G].unsqueeze(-1)  # [batch, seq_len, 1]
        C_pos = one_hot_seq[:, :, RNA_C].unsqueeze(-1)  # [batch, seq_len, 1]
        
        # Create pairing matrices
        # Watson-Crick: A-U pairs
        AU_pairs = torch.matmul(A_pos, U_pos.transpose(1, 2))  # [batch, seq_len, seq_len]
        # Watson-Crick: G-C pairs
        GC_pairs = torch.matmul(G_pos, C_pos.transpose(1, 2))  # [batch, seq_len, seq_len]
        # Wobble: G-U pairs
        GU_pairs = torch.matmul(G_pos, U_pos.transpose(1, 2))  # [batch, seq_len, seq_len]
        
        # Combined pairing probability (with G-U having lower weight)
        # Watson-Crick pairs (A-U, G-C) get full weight
        # Wobble pairs (G-U) get 0.8 weight as they're less stable
        pairing_prob = AU_pairs + GC_pairs + 0.8 * GU_pairs
        
        # Apply temperature scaling for softer constraints
        if self.temperature > 0:
            pairing_prob = pairing_prob / self.temperature
            
        return pairing_prob


class RNAInputEmbedder(nn.Module):
    """RNA Input embedder."""

    def __init__(
        self,
        atom_s: int,
        atom_z: int,
        token_s: int,
        token_z: int,
        atoms_per_window_queries: int,
        atoms_per_window_keys: int,
        atom_feature_dim: int,
        atom_encoder_depth: int,
        atom_encoder_heads: int,
        no_atom_encoder: bool = False,
        use_secondary_structure: bool = True,
    ) -> None:
        """Initialize the RNA input embedder.

        Parameters
        ----------
        atom_s : int
            The atom single representation dimension.
        atom_z : int
            The atom pair representation dimension.
        token_s : int
            The single token representation dimension.
        token_z : int
            The pair token representation dimension.
        atoms_per_window_queries : int
            The number of atoms per window for queries.
        atoms_per_window_keys : int
            The number of atoms per window for keys.
        atom_feature_dim : int
            The atom feature dimension.
        atom_encoder_depth : int
            The atom encoder depth.
        atom_encoder_heads : int
            The atom encoder heads.
        no_atom_encoder : bool, optional
            Whether to use the atom encoder, by default False
        use_secondary_structure : bool, optional
            Whether to use RNA secondary structure features, by default True
        """
        super().__init__()
        self.token_s = token_s
        self.no_atom_encoder = no_atom_encoder
        self.use_secondary_structure = use_secondary_structure

        if not no_atom_encoder:
            self.atom_attention_encoder = AtomAttentionEncoder(
                atom_s=atom_s,
                atom_z=atom_z,
                token_s=token_s,
                token_z=token_z,
                atoms_per_window_queries=atoms_per_window_queries,
                atoms_per_window_keys=atoms_per_window_keys,
                atom_feature_dim=atom_feature_dim,
                atom_encoder_depth=atom_encoder_depth,
                atom_encoder_heads=atom_encoder_heads,
                structure_prediction=False,
            )
        
        # RNA-specific base constraints
        self.base_constraints = RNABaseConstraints()

    def forward(self, feats: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass.

        Parameters
        ----------
        feats : Dict[str, Tensor]
            Input features

        Returns
        -------
        Tuple[Tensor, Tensor]
            The embedded tokens and base-pairing constraints.
        """
        # Load relevant features
        nuc_type = feats["nuc_type"]  # RNA nucleotide type
        profile = feats["profile"]
        deletion_mean = feats["deletion_mean"].unsqueeze(-1)
        
        # RNA-specific features
        if self.use_secondary_structure and "sec_structure" in feats:
            sec_structure = feats["sec_structure"]  # Secondary structure features
        else:
            # Create empty tensor if not available
            sec_structure = torch.zeros(
                (nuc_type.shape[0], nuc_type.shape[1], 0),
                device=nuc_type.device,
            )
        
        # Add site-specific features if available
        pocket_feature = feats.get("pocket_feature", 
                                 torch.zeros((nuc_type.shape[0], nuc_type.shape[1], 0), 
                                             device=nuc_type.device))

        # Compute input embedding using atom encoder if available
        if self.no_atom_encoder:
            a = torch.zeros(
                (nuc_type.shape[0], nuc_type.shape[1], self.token_s),
                device=nuc_type.device,
            )
        else:
            a, _, _, _, _ = self.atom_attention_encoder(feats)
            
        # Concatenate all features

        # 
        s = torch.cat([a, nuc_type, profile, deletion_mean, sec_structure, pocket_feature], dim=-1)
        
        # Compute base-pairing constraints if nucleotide one-hot encoding is available
        # Use one-hot nucleotide representation (nuc_type) if it has the right shape
        if nuc_type.shape[-1] == const.rna_num_tokens:  # One-hot encoding
            base_pairing = self.base_constraints(nuc_type)
        else:
            # Otherwise, create a placeholder - this should be implemented properly in production
            base_pairing = torch.zeros(
                (nuc_type.shape[0], nuc_type.shape[1], nuc_type.shape[1]),
                device=nuc_type.device,
            )
            
        return s, base_pairing


class RNAMSALayer(nn.Module):
    """MSA layer with RNA-specific adaptations."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        use_watson_crick_constraints: bool = True,
    ) -> None:
        """Initialize the RNA MSA layer.

        Parameters match the original MSALayer with added RNA-specific parameters.
        """
        super().__init__()
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_watson_crick_constraints = use_watson_crick_constraints
        self.token_z = token_z
        
        # MSA sequence processing - same as original
        self.msa_transition = Transition(dim=msa_s, hidden=msa_s * 4)
        
        # Use standard pair weighted averaging
        self.pair_weighted_averaging = PairWeightedAveraging(
            c_m=msa_s,
            c_z=token_z,
            c_h=32,
            num_heads=8,
        )

        # Triangle functions - same as original
        self.tri_mul_out = TriangleMultiplicationOutgoing(token_z)
        self.tri_mul_in = TriangleMultiplicationIncoming(token_z)
        self.tri_att_start = TriangleAttentionStartingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            token_z, pairwise_head_width, pairwise_num_heads, inf=1e9
        )
        self.z_transition = Transition(
            dim=token_z,
            hidden=token_z * 4,
        )
        
        # Use standard outer product mean
        self.outer_product_mean = OuterProductMean(
            c_in=msa_s,
            c_hidden=32,
            c_out=token_z,
        )
        
        # Base-pairing projection 
        self.base_pair_proj = nn.Linear(1, token_z // 4) if use_watson_crick_constraints else None

    def forward(
        self,
        z: Tensor,
        m: Tensor,
        token_mask: Tensor,
        msa_mask: Tensor,
        base_pairing: Tensor = None,
        chunk_heads_pwa: bool = False,
        chunk_size_transition_z: int = None,
        chunk_size_transition_msa: int = None,
        chunk_size_outer_product: int = None,
        chunk_size_tri_attn: int = None,
    ) -> Tuple[Tensor, Tensor]:
        """Perform the forward pass.
        
        Parameters match the original MSALayer with added RNA-specific parameters.
        """
        # Apply base-pairing bias to z before using it in attention
        if self.use_watson_crick_constraints and base_pairing is not None:
            # Add a small bias to the pairwise representation where base pairing occurs
            z_bp = z + 0.1 * base_pairing.unsqueeze(-1) * z
            
            # Use this adjusted z for pair weighted averaging, but keep original z
            z_attn = z_bp
        else:
            z_attn = z

        

        # Communication to MSA stack - similar to original
        msa_dropout = get_dropout_mask(self.msa_dropout, m, self.training)

        m = m + msa_dropout * self.pair_weighted_averaging(
            m, z_attn, token_mask, chunk_heads_pwa
        )
        m = m + self.msa_transition(m, chunk_size_transition_msa)

        # Communication to pairwise stack - same as original
        z = z + self.outer_product_mean(m, msa_mask, chunk_size_outer_product)

        # RNA-specific: Inject Watson-Crick base-pairing information if available
        if self.use_watson_crick_constraints and base_pairing is not None:
            # Project the base-pairing constraints
            bp_projected = self.base_pair_proj(base_pairing.unsqueeze(-1))
            # Expand to match part of z's channel dimension
            bp_expanded = bp_projected.expand(-1, -1, -1, self.token_z // 4)
            # Repeat the pattern for the remaining channels
            bp_repeated = bp_expanded.repeat(1, 1, 1, 4)
            # Apply a small weight to this constraint
            z = z + 0.1 * bp_repeated * token_mask.unsqueeze(-1)

        # Pairwise stack operations - same as original
        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_out(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_mul_in(z, mask=token_mask)

        dropout = get_dropout_mask(self.z_dropout, z, self.training)
        z = z + dropout * self.tri_att_start(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
        )

        dropout = get_dropout_mask(self.z_dropout, z, self.training, columnwise=True)
        z = z + dropout * self.tri_att_end(
            z,
            mask=token_mask,
            chunk_size=chunk_size_tri_attn,
        )

        z = z + self.z_transition(z, chunk_size_transition_z)

        return z, m


class RNAMSAModule(nn.Module):
    """RNA MSA module specifically designed for RNA sequences."""

    def __init__(
        self,
        msa_s: int,
        token_z: int,
        s_input_dim: int,
        msa_blocks: int,
        msa_dropout: float,
        z_dropout: float,
        pairwise_head_width: int = 32,
        pairwise_num_heads: int = 4,
        activation_checkpointing: bool = False,
        use_paired_feature: bool = True,  # Default to True for RNA
        use_secondary_structure: bool = True,  # Default to True for RNA
        use_watson_crick_constraints: bool = True,  # RNA-specific base-pairing
        base_pair_weight: float = 1.0,  # Weight for base-pairing constraints
        offload_to_cpu: bool = False,
        **kwargs,
    ) -> None:
        """Initialize the RNA MSA module.

        Parameters
        ----------
        msa_s : int
            The MSA embedding size.
        token_z : int
            The token pairwise embedding size.
        s_input_dim : int
            The input sequence dimension.
        msa_blocks : int
            The number of MSA blocks.
        msa_dropout : float
            The MSA dropout.
        z_dropout : float
            The pairwise dropout.
        pairwise_head_width : int, optional
            The pairwise head width, by default 32
        pairwise_num_heads : int, optional
            The number of pairwise heads, by default 4
        activation_checkpointing : bool, optional
            Whether to use activation checkpointing, by default False
        use_paired_feature : bool, optional
            Whether to use the paired feature, by default True for RNA
        use_secondary_structure : bool, optional
            Whether to use RNA secondary structure features, by default True
        use_watson_crick_constraints : bool, optional
            Whether to use Watson-Crick base-pairing constraints, by default True
        base_pair_weight : float, optional
            Weight for base-pairing constraints, by default 1.0
        offload_to_cpu : bool, optional
            Whether to offload to CPU, by default False
        """
        super().__init__()
        self.msa_blocks = msa_blocks
        self.msa_dropout = msa_dropout
        self.z_dropout = z_dropout
        self.use_paired_feature = use_paired_feature
        self.use_secondary_structure = use_secondary_structure
        self.use_watson_crick_constraints = use_watson_crick_constraints
        self.base_pair_weight = base_pair_weight

        self.s_proj = nn.Linear(s_input_dim, msa_s, bias=False)
        
        # Calculate input feature dimension for the MSA projection
        msa_feature_dim = const.rna_num_tokens + 2  # Base nucleotides + deletions
        if use_paired_feature:
            msa_feature_dim += 1  # Add pairing information
        if use_secondary_structure:
            msa_feature_dim += 3  # Add secondary structure features (stem, loop, bulge)
            
        self.msa_proj = nn.Linear(
            msa_feature_dim,
            msa_s,
            bias=False,
        )

        print(f"msa_feature_dim: {msa_feature_dim}")
        print(f"msa_s: {msa_s}")
        
        # Initialize MSA layers
        self.layers = nn.ModuleList()
        for i in range(msa_blocks):
            if activation_checkpointing:
                self.layers.append(
                    checkpoint_wrapper(
                        RNAMSALayer(
                            msa_s,
                            token_z,
                            msa_dropout,
                            z_dropout,
                            pairwise_head_width,
                            pairwise_num_heads,
                            use_watson_crick_constraints=use_watson_crick_constraints,
                        ),
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            else:
                self.layers.append(
                    RNAMSALayer(
                        msa_s,
                        token_z,
                        msa_dropout,
                        z_dropout,
                        pairwise_head_width,
                        pairwise_num_heads,
                        use_watson_crick_constraints=use_watson_crick_constraints,
                    )
                )

    def forward(
        self,
        z: Tensor,
        emb: Tensor,
        feats: Dict[str, Tensor],
        base_pairing: Tensor = None,
    ) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        emb : Tensor
            The input embeddings
        feats : Dict[str, Tensor]
            Input features
        base_pairing : Tensor, optional
            Base-pairing constraints, by default None

        Returns
        -------
        Tensor
            The output pairwise embeddings.
        """
        # Set chunk sizes for optimization
        if not self.training:
            if z.shape[1] > const.chunk_size_threshold:
                chunk_heads_pwa = True
                chunk_size_transition_z = 64
                chunk_size_transition_msa = 32
                chunk_size_outer_product = 4
                chunk_size_tri_attn = 128
            else:
                chunk_heads_pwa = False
                chunk_size_transition_z = None
                chunk_size_transition_msa = None
                chunk_size_outer_product = None
                chunk_size_tri_attn = 512
        else:
            chunk_heads_pwa = False
            chunk_size_transition_z = None
            chunk_size_transition_msa = None
            chunk_size_outer_product = None
            chunk_size_tri_attn = None

        # Get device from input tensor to ensure all tensors are on the same device
        device = z.device

        # Load relevant features
        rna_msa = feats["rna_msa"].to(device)  # RNA sequence alignment [batch, num_seqs, seq_len, 4]
        has_deletion = feats["has_deletion"].to(device).unsqueeze(-1)  # [batch, num_seqs, seq_len, 1]
        deletion_value = feats["deletion_value"].to(device).unsqueeze(-1)  # [batch, num_seqs, seq_len, 1]
        
        # Get pairing information if used
        if self.use_paired_feature:
            is_paired = feats["msa_paired"].to(device).unsqueeze(-1)  # [batch, num_seqs, seq_len, 1]
        
        # Get secondary structure features if used
        if self.use_secondary_structure:
            # These features would represent probability of being in different structural elements
            # Convert 3D tensors to 4D by adding a feature dimension
            zeros_template = torch.zeros_like(has_deletion).to(device)
            stem_prob = feats.get("stem_prob", zeros_template).to(device).unsqueeze(-1)  # [batch, num_seqs, seq_len, 1]
            loop_prob = feats.get("loop_prob", zeros_template).to(device).unsqueeze(-1)  # [batch, num_seqs, seq_len, 1]
            bulge_prob = feats.get("bulge_prob", zeros_template).to(device).unsqueeze(-1)  # [batch, num_seqs, seq_len, 1]
        
        # Get masks
        msa_mask = feats["msa_mask"].to(device)
        token_mask = feats["token_pad_mask"].to(device).float()
        token_mask = token_mask[:, :, None] * token_mask[:, None, :]

        # Compute RNA MSA embeddings by combining features
        feature_list = [rna_msa, has_deletion, deletion_value]
        
        if self.use_paired_feature:
            feature_list.append(is_paired)
            
        if self.use_secondary_structure:
            feature_list.extend([stem_prob, loop_prob, bulge_prob])
        
        # All tensors in feature_list should now have 4 dimensions [batch, num_seqs, seq_len, feat_dim]
        m = torch.cat(feature_list, dim=-1)

        # Project input embeddings and MSA features
        m = self.msa_proj(m)
        m = m + self.s_proj(emb).unsqueeze(1)
        
        # Apply Watson-Crick constraints if enabled
        if self.use_watson_crick_constraints and base_pairing is not None:
            # Ensure base_pairing is on the correct device
            base_pairing = base_pairing.to(device)
            
            # Incorporate base-pairing constraints into pairwise representation
            # Scale it by the base pair weight hyperparameter
            base_pair_bias = self.base_pair_weight * base_pairing
            
            # Add base-pairing bias to the pairwise representation
            # We expand the dimensions to match z's shape
            bias_expanded = base_pair_bias.unsqueeze(-1).expand(-1, -1, -1, z.shape[-1])
            z = z + bias_expanded

        # Process through RNA MSA blocks
        for i in range(self.msa_blocks):
            z, m = self.layers[i](
                z,
                m,
                token_mask,
                msa_mask,
                base_pairing=base_pairing if self.use_watson_crick_constraints else None,
                chunk_heads_pwa=chunk_heads_pwa,
                chunk_size_transition_z=chunk_size_transition_z,
                chunk_size_transition_msa=chunk_size_transition_msa,
                chunk_size_outer_product=chunk_size_outer_product,
                chunk_size_tri_attn=chunk_size_tri_attn,
            )
        return z


class RNADistogramModule(nn.Module):
    """RNA-specific distogram module that predicts distances between nucleotides."""

    def __init__(self, token_z: int, num_bins: int) -> None:
        """Initialize the RNA distogram module.

        Parameters
        ----------
        token_z : int
            The token pairwise embedding size.
        num_bins : int
            The number of bins.
        """
        super().__init__()
        self.distogram = nn.Linear(token_z, num_bins)
        
        # RNA-specific projection to enhance base-pairing distances
        self.base_pair_projection = nn.Linear(token_z, token_z)
        
        # Special projection for Watson-Crick pairs
        self.wc_projection = nn.Sequential(
            nn.Linear(token_z, token_z // 2),
            nn.ReLU(),
            nn.Linear(token_z // 2, token_z)
        )
        
        # Base-pairing projection
        self.direct_bp_projection = nn.Linear(1, token_z // 4)
        
        # Learnable weights for different structural elements
        self.stem_weight = nn.Parameter(torch.ones(1) * 1.2)  # Slightly higher weight for stems
        self.loop_weight = nn.Parameter(torch.ones(1) * 0.8)  # Slightly lower weight for loops
        self.bulge_weight = nn.Parameter(torch.ones(1) * 0.9)  # Medium weight for bulges

    def forward(self, z: Tensor, structure_info: Dict[str, Tensor] = None) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        z : Tensor
            The pairwise embeddings
        structure_info : Dict[str, Tensor], optional
            Additional structure information, by default None
            Can include:
                - 'stem_prob': probability of being in a stem
                - 'loop_prob': probability of being in a loop
                - 'bulge_prob': probability of being in a bulge
                - 'base_pairing': base-pairing probabilities

        Returns
        -------
        Tensor
            The predicted RNA distogram.
        """
        # RNA distances are often more constrained by base-pairing
        z_bp = self.base_pair_projection(z)
        z = z + z_bp
        
        # Apply Watson-Crick specific projection
        z_wc = self.wc_projection(z)
        
        # If we have structural information, use it to weight the representation
        if structure_info is not None:
            # Process direct base-pairing information if available
            if 'base_pairing' in structure_info:
                base_pairing = structure_info['base_pairing']
                # Project and expand base-pairing information
                bp_projected = self.direct_bp_projection(base_pairing.unsqueeze(-1))
                # Match z's channels dimension
                channels = z.size(-1)
                
                # Fix: properly handle the 4D tensor expansion
                # bp_projected shape is [batch, seq_len, seq_len, token_z//4]
                # We need to repeat this last dimension to match z's channel dimension
                repeat_factor = channels // bp_projected.size(-1)
                bp_repeated = bp_projected.repeat(1, 1, 1, repeat_factor)
                
                # Add to z with a smaller weight to not overpower learned features
                z = z + 0.15 * bp_repeated
            
            if 'stem_prob' in structure_info:
                # Expand stem probability to match z's shape
                stem_prob = structure_info['stem_prob']
                # Make sure stem_prob has the right shape for broadcasting
                if len(stem_prob.shape) == 3:  # [batch, seq_len, seq_len]
                    stem_expanded = stem_prob.unsqueeze(-1)
                else:  # [batch, seq_len]
                    # Create pairwise structure matrix
                    seq_len = stem_prob.size(1)
                    stem_expanded = stem_prob.unsqueeze(2).expand(-1, -1, seq_len).unsqueeze(-1)
                
                # Apply stem weight - stems have more constrained distances
                z = z + self.stem_weight * stem_expanded * z_wc
                
            if 'loop_prob' in structure_info:
                # Apply lower weight to loop regions
                loop_prob = structure_info['loop_prob']
                # Make sure loop_prob has the right shape for broadcasting
                if len(loop_prob.shape) == 3:  # [batch, seq_len, seq_len]
                    loop_expanded = loop_prob.unsqueeze(-1)
                else:  # [batch, seq_len]
                    # Create pairwise structure matrix
                    seq_len = loop_prob.size(1)
                    loop_expanded = loop_prob.unsqueeze(2).expand(-1, -1, seq_len).unsqueeze(-1)
                
                # Loops are more flexible
                z = z + self.loop_weight * loop_expanded * z
                
            if 'bulge_prob' in structure_info:
                # Use bulge information to adjust distances
                bulge_prob = structure_info['bulge_prob']
                # Make sure bulge_prob has the right shape for broadcasting
                if len(bulge_prob.shape) == 3:  # [batch, seq_len, seq_len]
                    bulge_expanded = bulge_prob.unsqueeze(-1)
                else:  # [batch, seq_len]
                    # Create pairwise structure matrix
                    seq_len = bulge_prob.size(1)
                    bulge_expanded = bulge_prob.unsqueeze(2).expand(-1, -1, seq_len).unsqueeze(-1)
                
                # Bulges have intermediate flexibility between stems and loops
                z = z + self.bulge_weight * bulge_expanded * z_wc
        
        # Symmetrize the representation for distance prediction
        z = z + z.transpose(1, 2)
        
        return self.distogram(z) 