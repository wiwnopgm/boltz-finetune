"""JAX implementation of confidence loss functions for protein structure prediction."""

from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp


def plddt_loss(
    pred_plddt: jnp.ndarray,
    true_ca_lddt: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute loss for per-residue confidence (pLDDT) prediction.
    
    Args:
        pred_plddt: Predicted pLDDT values [B, L, 1]
        true_ca_lddt: True CA LDDT values [B, L]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        MSE loss for pLDDT prediction
    """
    # Squeeze last dimension of predictions if necessary
    pred_plddt = pred_plddt.squeeze(-1)  # [B, L]
    
    # Compute MSE loss
    squared_error = (pred_plddt - true_ca_lddt) ** 2
    
    # Apply mask and compute mean
    masked_se = squared_error * mask
    loss = jnp.sum(masked_se) / (jnp.sum(mask) + eps)
    
    return loss


def pae_loss(
    pred_pae: jnp.ndarray,
    true_pae: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute loss for predicted aligned error (PAE).
    
    Args:
        pred_pae: Predicted PAE values [B, L, L, 1]
        true_pae: True PAE values [B, L, L]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        MSE loss for PAE prediction
    """
    # Squeeze last dimension of predictions if necessary
    pred_pae = pred_pae.squeeze(-1)  # [B, L, L]
    
    # Create 2D mask from 1D mask
    mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
    
    # Compute MSE loss
    squared_error = (pred_pae - true_pae) ** 2
    
    # Apply mask and compute mean
    masked_se = squared_error * mask_2d
    loss = jnp.sum(masked_se) / (jnp.sum(mask_2d) + eps)
    
    return loss


def pde_loss(
    pred_pde: jnp.ndarray,
    true_pde: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute loss for predicted distance error (PDE).
    
    Args:
        pred_pde: Predicted PDE values [B, L, L, 1]
        true_pde: True PDE values [B, L, L]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        MSE loss for PDE prediction
    """
    return pae_loss(pred_pde, true_pde, mask, eps)


def masked_binary_crossentropy(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute binary cross-entropy loss with masking.
    
    Args:
        pred: Predicted values in [0, 1] range
        target: Target values in {0, 1}
        mask: Mask indicating valid positions
        eps: Small value for numerical stability
        
    Returns:
        Binary cross-entropy loss
    """
    # Clip prediction values for numerical stability
    pred = jnp.clip(pred, eps, 1.0 - eps)
    
    # Compute BCE loss
    bce = -target * jnp.log(pred) - (1.0 - target) * jnp.log(1.0 - pred)
    
    # Apply mask
    masked_bce = bce * mask
    loss = jnp.sum(masked_bce) / (jnp.sum(mask) + eps)
    
    return loss


def confidence_loss(
    outputs: Dict[str, jnp.ndarray],
    batch: Dict[str, jnp.ndarray],
    lddt_weight: float = 1.0,
    pae_weight: float = 0.1,
    pde_weight: float = 0.1,
    resolved_weight: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute combined confidence loss for all prediction heads.
    
    Args:
        outputs: Dictionary of model outputs
        batch: Dictionary of input batch data
        lddt_weight: Weight for LDDT loss
        pae_weight: Weight for PAE loss
        pde_weight: Weight for PDE loss
        resolved_weight: Weight for resolved residue prediction
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (total_loss, loss_dict)
    """
    loss_dict = {}
    total_loss = 0.0
    
    # Get masks
    mask = batch["atom_mask"] if "atom_mask" in batch else batch["seq_mask"]
    
    # pLDDT loss
    if "plddt" in outputs and "lddt" in batch:
        plddt_l = plddt_loss(outputs["plddt"], batch["lddt"], mask, eps)
        loss_dict["plddt_loss"] = plddt_l
        total_loss += lddt_weight * plddt_l
    
    # PAE loss
    if "pae" in outputs and "pae" in batch:
        pae_l = pae_loss(outputs["pae"], batch["pae"], mask, eps)
        loss_dict["pae_loss"] = pae_l
        total_loss += pae_weight * pae_l
    
    # PDE loss
    if "pde" in outputs and "pde" in batch:
        pde_l = pde_loss(outputs["pde"], batch["pde"], mask, eps)
        loss_dict["pde_loss"] = pde_l
        total_loss += pde_weight * pde_l
    
    # Resolved residue loss
    if "resolved" in outputs and "resolved" in batch:
        resolved_l = masked_binary_crossentropy(
            outputs["resolved"],
            batch["resolved"],
            mask,
            eps,
        )
        loss_dict["resolved_loss"] = resolved_l
        total_loss += resolved_weight * resolved_l
    
    loss_dict["total_loss"] = total_loss
    return total_loss, loss_dict


def compute_lddt(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    mask: jnp.ndarray,
    cutoffs: jnp.ndarray = jnp.array([0.5, 1.0, 2.0, 4.0]),
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Compute Local Distance Difference Test (LDDT) scores.
    
    Args:
        pred_coords: Predicted coordinates [B, L, 3]
        true_coords: Ground truth coordinates [B, L, 3]
        mask: Sequence mask [B, L]
        cutoffs: Distance cutoffs for LDDT calculation
        eps: Small value for numerical stability
        
    Returns:
        LDDT scores [B, L]
    """
    # Get batch and sequence dimensions
    batch_size, seq_len = mask.shape
    
    # Create pairwise distance matrices
    pred_dmat = jnp.sqrt(
        jnp.sum(
            (pred_coords[:, :, None, :] - pred_coords[:, None, :, :]) ** 2,
            axis=-1,
        ) + eps
    )  # [B, L, L]
    
    true_dmat = jnp.sqrt(
        jnp.sum(
            (true_coords[:, :, None, :] - true_coords[:, None, :, :]) ** 2,
            axis=-1,
        ) + eps
    )  # [B, L, L]
    
    # Create 2D mask
    mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
    
    # Exclude self-distances
    self_mask = 1.0 - jnp.eye(seq_len)[None]  # [1, L, L]
    mask_2d = mask_2d * self_mask  # [B, L, L]
    
    # Compute LDDT for different cutoffs
    lddt_scores = []
    for cutoff in cutoffs:
        # Reference distance mask
        ref_mask = (true_dmat < cutoff) * mask_2d  # [B, L, L]
        
        # Compute absolute distance differences
        dist_diff = jnp.abs(true_dmat - pred_dmat) * ref_mask
        
        # Count distances within tolerance
        within_tolerance = (dist_diff < cutoff / 2) * ref_mask
        
        # Sum per residue
        residue_counts = jnp.sum(ref_mask, axis=-1)  # [B, L]
        residue_within = jnp.sum(within_tolerance, axis=-1)  # [B, L]
        
        # Compute score
        score = residue_within / (residue_counts + eps)
        lddt_scores.append(score)
    
    # Average over cutoffs
    lddt = jnp.mean(jnp.stack(lddt_scores, axis=0), axis=0)  # [B, L]
    
    # Apply original mask
    lddt = lddt * mask
    
    return lddt


def compute_pae(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Compute Predicted Aligned Error (PAE).
    
    Args:
        pred_coords: Predicted coordinates [B, L, 3]
        true_coords: Ground truth coordinates [B, L, 3]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        PAE matrix [B, L, L]
    """
    # Create 2D mask
    mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
    
    # Get batch and sequence dimensions
    batch_size, seq_len = mask.shape
    
    # Compute pairwise distances for predictions and ground truth
    pred_dmat = jnp.sqrt(
        jnp.sum(
            (pred_coords[:, :, None, :] - pred_coords[:, None, :, :]) ** 2,
            axis=-1,
        ) + eps
    )  # [B, L, L]
    
    true_dmat = jnp.sqrt(
        jnp.sum(
            (true_coords[:, :, None, :] - true_coords[:, None, :, :]) ** 2,
            axis=-1,
        ) + eps
    )  # [B, L, L]
    
    # Compute absolute error in pairwise distances
    pae = jnp.abs(pred_dmat - true_dmat) * mask_2d
    
    return pae 