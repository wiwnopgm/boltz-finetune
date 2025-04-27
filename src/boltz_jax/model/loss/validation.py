"""JAX implementation of validation metrics and losses for model evaluation."""

from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp

from boltz_jax.model.loss.confidence import compute_lddt, compute_pae


def weighted_minimum_rmsd(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    mask: jnp.ndarray,
    weights: Optional[jnp.ndarray] = None,
    eps: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute minimum RMSD between predicted and true coordinates with optional weights.
    
    Implements Kabsch algorithm to find the optimal rotation.
    
    Args:
        pred_coords: Predicted coordinates [B, N, 3]
        true_coords: Ground truth coordinates [B, N, 3]
        mask: Atom mask [B, N]
        weights: Optional per-atom weights [B, N]
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (rmsd, rotated_coords)
    """
    batch_size = pred_coords.shape[0]
    
    # Use uniform weights if not provided
    if weights is None:
        weights = mask
    else:
        weights = weights * mask
    
    # Apply mask to coordinates
    pred_coords_masked = pred_coords * mask[:, :, None]
    true_coords_masked = true_coords * mask[:, :, None]
    
    # Normalize weights to sum to 1
    weights_normalized = weights / (jnp.sum(weights, axis=-1, keepdims=True) + eps)
    
    # Center coordinates
    pred_centroid = jnp.sum(
        pred_coords_masked * weights_normalized[:, :, None], axis=1, keepdims=True
    )
    true_centroid = jnp.sum(
        true_coords_masked * weights_normalized[:, :, None], axis=1, keepdims=True
    )
    
    pred_centered = pred_coords_masked - pred_centroid
    true_centered = true_coords_masked - true_centroid
    
    # Apply mask again to handle potential numerical issues
    pred_centered = pred_centered * mask[:, :, None]
    true_centered = true_centered * mask[:, :, None]
    
    # Function to compute optimal rotation for one example
    def _get_optimal_rotation(pred, true, wts):
        # Compute correlation matrix
        wts_sqrt = jnp.sqrt(wts)[:, None]
        covariance = jnp.matmul(
            jnp.transpose(pred * wts_sqrt), 
            true * wts_sqrt
        )
        
        # Compute SVD
        u, _, vh = jnp.linalg.svd(covariance)
        
        # Compute rotation matrix
        # Handle reflection case to ensure proper rotation (det R = 1)
        det = jnp.linalg.det(jnp.matmul(vh.T, u.T))
        correction = jnp.array([[1.0, 0.0, 0.0], 
                               [0.0, 1.0, 0.0], 
                               [0.0, 0.0, det]])
        rotation = jnp.matmul(vh.T, jnp.matmul(correction, u.T))
        
        return rotation
    
    # Compute optimal rotation for each batch element
    rotations = []
    for i in range(batch_size):
        rotation = _get_optimal_rotation(
            pred_centered[i], true_centered[i], weights_normalized[i]
        )
        rotations.append(rotation)
    
    # Stack rotations
    rotation_matrices = jnp.stack(rotations, axis=0)
    
    # Apply rotation to centered predictions
    pred_aligned = jnp.matmul(pred_centered, rotation_matrices)
    
    # Translate back
    pred_aligned = pred_aligned + true_centroid
    
    # Compute RMSD
    squared_diff = jnp.sum(
        (pred_aligned - true_coords_masked) ** 2, axis=-1
    )
    rmsd = jnp.sqrt(
        jnp.sum(squared_diff * weights_normalized, axis=-1)
    )
    
    return rmsd, pred_aligned


def factored_lddt_loss(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    mask: jnp.ndarray,
    factor: float = 10.0,
    pred_lddt: Optional[jnp.ndarray] = None,
    eps: float = 1e-10,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute LDDT loss with optional weighting by predicted pLDDT.
    
    Args:
        pred_coords: Predicted coordinates [B, L, 3]
        true_coords: Ground truth coordinates [B, L, 3]
        mask: Sequence mask [B, L]
        factor: Scaling factor for loss
        pred_lddt: Optional predicted pLDDT values for weighting [B, L]
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Compute true LDDT scores
    true_lddt = compute_lddt(pred_coords, true_coords, mask, eps=eps)
    
    # Compute loss using predicted confidence if available
    if pred_lddt is not None:
        # Squeeze last dimension if necessary
        pred_lddt = pred_lddt.squeeze(-1)
        
        # Factor the loss by predicted confidence
        confidence_weight = jax.nn.sigmoid(factor * (pred_lddt - 0.5))
        loss_weight = mask * confidence_weight
        
        # Compute loss
        squared_error = (pred_lddt - true_lddt) ** 2
        loss = jnp.sum(squared_error * loss_weight) / (jnp.sum(loss_weight) + eps)
    else:
        # Regular MSE loss
        loss = jnp.sum((1.0 - true_lddt) ** 2 * mask) / (jnp.sum(mask) + eps)
    
    # Compute average LDDT for monitoring
    avg_lddt = jnp.sum(true_lddt) / (jnp.sum(mask) + eps)
    
    metrics = {
        "loss": loss,
        "avg_lddt": avg_lddt,
    }
    
    return loss, metrics


def factored_token_lddt_dist_loss(
    pred_coords: jnp.ndarray,
    true_coords: jnp.ndarray,
    mask: jnp.ndarray,
    factor: float = 10.0,
    pred_token_lddt: Optional[jnp.ndarray] = None,
    pred_dist: Optional[jnp.ndarray] = None,
    eps: float = 1e-10,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute combined LDDT and distance loss with confidence weighting.
    
    Args:
        pred_coords: Predicted coordinates [B, L, 3]
        true_coords: Ground truth coordinates [B, L, 3]
        mask: Sequence mask [B, L]
        factor: Scaling factor for confidence weighting
        pred_token_lddt: Optional predicted per-token LDDT [B, L]
        pred_dist: Optional predicted distogram logits [B, L, L, bins]
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    # Compute LDDT loss
    lddt_loss, lddt_metrics = factored_lddt_loss(
        pred_coords,
        true_coords,
        mask,
        factor,
        pred_token_lddt,
        eps,
    )
    
    # Initialize with LDDT loss and metrics
    total_loss = lddt_loss
    metrics = lddt_metrics
    
    # Add distogram loss if available
    if pred_dist is not None:
        # Compute pairwise distances for ground truth coordinates
        true_dmat = jnp.sqrt(
            jnp.sum(
                (true_coords[:, :, None, :] - true_coords[:, None, :, :]) ** 2,
                axis=-1,
            ) + eps
        )  # [B, L, L]
        
        # Create 2D mask
        mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
        
        # Convert true distances to one-hot encoded bins
        bin_size = (self.max_bin - self.min_bin) / self.num_bins
        bin_edges = jnp.arange(self.num_bins) * bin_size + self.min_bin
        
        true_bins = jnp.sum(true_dmat[..., None] > bin_edges, axis=-1)
        true_bins = jnp.clip(true_bins, 0, self.num_bins - 1)
        true_onehot = jax.nn.one_hot(true_bins, self.num_bins)  # [B, L, L, num_bins]
        
        # Compute cross-entropy loss
        dist_loss = -jnp.sum(
            true_onehot * jax.nn.log_softmax(pred_dist, axis=-1) * mask_2d[:, :, :, None],
            axis=[-1, -2, -3],
        ) / (jnp.sum(mask_2d) + eps)
        
        # Add to total loss
        total_loss += dist_loss
        metrics["dist_loss"] = dist_loss
    
    return total_loss, metrics


def compute_plddt_mae(
    pred_plddt: jnp.ndarray,
    true_lddt: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Compute Mean Absolute Error for pLDDT predictions.
    
    Args:
        pred_plddt: Predicted pLDDT values [B, L, 1]
        true_lddt: True LDDT values [B, L]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        MAE for pLDDT prediction
    """
    # Squeeze last dimension of predictions if necessary
    pred_plddt = pred_plddt.squeeze(-1)  # [B, L]
    
    # Compute absolute error
    abs_error = jnp.abs(pred_plddt - true_lddt)
    
    # Apply mask and compute mean
    masked_ae = abs_error * mask
    mae = jnp.sum(masked_ae) / (jnp.sum(mask) + eps)
    
    return mae


def compute_pae_mae(
    pred_pae: jnp.ndarray,
    true_pae: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Compute Mean Absolute Error for PAE predictions.
    
    Args:
        pred_pae: Predicted PAE values [B, L, L, 1]
        true_pae: True PAE values [B, L, L]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        MAE for PAE prediction
    """
    # Squeeze last dimension of predictions if necessary
    pred_pae = pred_pae.squeeze(-1)  # [B, L, L]
    
    # Create 2D mask
    mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
    
    # Compute absolute error
    abs_error = jnp.abs(pred_pae - true_pae)
    
    # Apply mask and compute mean
    masked_ae = abs_error * mask_2d
    mae = jnp.sum(masked_ae) / (jnp.sum(mask_2d) + eps)
    
    return mae


def compute_pde_mae(
    pred_pde: jnp.ndarray,
    true_pde: jnp.ndarray,
    mask: jnp.ndarray,
    eps: float = 1e-10,
) -> jnp.ndarray:
    """Compute Mean Absolute Error for PDE predictions.
    
    Args:
        pred_pde: Predicted PDE values [B, L, L, 1]
        true_pde: True PDE values [B, L, L]
        mask: Sequence mask [B, L]
        eps: Small value for numerical stability
        
    Returns:
        MAE for PDE prediction
    """
    return compute_pae_mae(pred_pde, true_pde, mask, eps)


def create_pseudo_beta(
    coords: jnp.ndarray,
    mask: jnp.ndarray,
    residue_index: jnp.ndarray,
    atom_mask: jnp.ndarray,
    eps: float = 1e-10,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Create pseudo-beta coordinates for residues from atom coordinates.
    
    Args:
        coords: Atom coordinates [B, N, 3]
        mask: Sequence mask [B, L]
        residue_index: Residue index for each atom [B, N]
        atom_mask: Atom mask [B, N]
        eps: Small value for numerical stability
        
    Returns:
        Tuple of (pseudo_beta_coords, pseudo_beta_mask)
    """
    batch_size, num_res = mask.shape
    
    # Initialize pseudo-beta coordinates
    pseudo_beta = jnp.zeros((batch_size, num_res, 3))
    pseudo_beta_mask = jnp.zeros((batch_size, num_res))
    
    # Determine CA atom indices (assuming atom type is encoded in the input)
    # In a real implementation, you would have atom types
    # For simplicity, we'll use the first atom of each residue as CA
    for b in range(batch_size):
        for res_idx in range(num_res):
            # Find atoms for this residue
            residue_mask = (residue_index[b] == res_idx) & atom_mask[b]
            if jnp.any(residue_mask):
                # Use the first atom (CA) for this residue
                atom_idx = jnp.argmax(residue_mask.astype(jnp.int32))
                pseudo_beta = pseudo_beta.at[b, res_idx].set(coords[b, atom_idx])
                pseudo_beta_mask = pseudo_beta_mask.at[b, res_idx].set(1.0)
    
    # Apply sequence mask
    pseudo_beta_mask = pseudo_beta_mask * mask
    
    return pseudo_beta, pseudo_beta_mask 