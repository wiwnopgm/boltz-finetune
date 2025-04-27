"""JAX implementation of distogram loss functions for protein structure prediction."""

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def distogram_loss(
    pred_logits: jnp.ndarray,
    target_positions: jnp.ndarray,
    mask: jnp.ndarray,
    min_bin: float = 2.0,
    max_bin: float = 22.0,
    num_bins: int = 64,
    eps: float = 1e-6,
) -> Dict[str, jnp.ndarray]:
    """Compute distogram loss between predicted and target distances.
    
    Args:
        pred_logits: Predicted distogram logits [B, L, L, num_bins]
        target_positions: Target atom positions [B, L, 3]
        mask: Sequence mask [B, L]
        min_bin: Minimum distance bin
        max_bin: Maximum distance bin
        num_bins: Number of distance bins
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with loss components
    """
    # Create 2D mask from 1D mask
    batch_size, seq_len = mask.shape
    mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
    
    # Compute ground truth pairwise distances
    ca_position = target_positions  # Using CA atom positions
    
    # Compute distances between all CA atoms
    dmat = jnp.sqrt(
        jnp.sum(
            (ca_position[:, :, None, :] - ca_position[:, None, :, :]) ** 2,
            axis=-1,
        ) + eps
    )  # [B, L, L]
    
    # Create bin edges and convert distances to one-hot encoded bins
    bin_size = (max_bin - min_bin) / num_bins
    bin_edges = jnp.arange(num_bins) * bin_size + min_bin
    
    # Create ground truth distogram (one-hot encoded)
    true_bins = jnp.sum(dmat[..., None] > bin_edges, axis=-1)
    true_bins = jnp.clip(true_bins, 0, num_bins - 1)
    
    # Convert to one-hot
    true_onehot = jax.nn.one_hot(true_bins, num_bins)  # [B, L, L, num_bins]
    
    # Compute cross-entropy loss
    loss_per_position = -jnp.sum(
        true_onehot * jax.nn.log_softmax(pred_logits, axis=-1),
        axis=-1,
    )  # [B, L, L]
    
    # Apply mask and compute mean loss
    loss_per_position = loss_per_position * mask_2d
    loss = jnp.sum(loss_per_position) / (jnp.sum(mask_2d) + eps)
    
    # Compute accuracy (for monitoring)
    pred_bins = jnp.argmax(pred_logits, axis=-1)
    correct = (pred_bins == true_bins) * mask_2d
    accuracy = jnp.sum(correct) / (jnp.sum(mask_2d) + eps)
    
    return {
        "loss": loss,
        "accuracy": accuracy,
    }


def symmetric_distogram_loss(
    pred_logits: jnp.ndarray,
    target_positions: jnp.ndarray,
    mask: jnp.ndarray,
    min_bin: float = 2.0,
    max_bin: float = 22.0,
    num_bins: int = 64,
    eps: float = 1e-6,
) -> Dict[str, jnp.ndarray]:
    """Compute symmetric distogram loss that handles chain symmetry.
    
    Args:
        pred_logits: Predicted distogram logits [B, L, L, num_bins]
        target_positions: Target atom positions [B, L, 3]
        mask: Sequence mask [B, L]
        min_bin: Minimum distance bin
        max_bin: Maximum distance bin
        num_bins: Number of distance bins
        eps: Small value for numerical stability
        
    Returns:
        Dictionary with loss components
    """
    # Create 2D mask from 1D mask
    batch_size, seq_len = mask.shape
    mask_2d = mask[:, None, :] * mask[:, :, None]  # [B, L, L]
    
    # Make distogram prediction symmetric
    # In JAX we manually symmetrize by averaging with the transpose
    pred_logits_symmetric = (pred_logits + jnp.transpose(pred_logits, (0, 2, 1, 3))) / 2.0
    
    # Compute standard distogram loss
    return distogram_loss(
        pred_logits_symmetric,
        target_positions,
        mask,
        min_bin,
        max_bin,
        num_bins,
        eps,
    )


def categorical_kl_divergence(
    logits: jnp.ndarray,
    target_distribution: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Compute KL divergence between two categorical distributions.
    
    Args:
        logits: Predicted logits [B, ..., num_categories]
        target_distribution: Target probability distribution [B, ..., num_categories]
        mask: Optional mask [B, ...] (same shape as inputs except for last dimension)
        eps: Small value for numerical stability
        
    Returns:
        KL divergence loss
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = jnp.sum(target_distribution * (jnp.log(target_distribution + eps) - log_probs),
                 axis=-1)
    
    if mask is not None:
        loss = loss * mask
        return jnp.sum(loss) / (jnp.sum(mask) + eps)
    else:
        return jnp.mean(loss)


def gaussian_bin_to_distogram(
    pred_logits: jnp.ndarray,
    min_bin: float = 2.0,
    max_bin: float = 22.0,
    num_bins: int = 64,
    sigma: float = 1.0,
) -> jnp.ndarray:
    """Convert bin logits to expected distances using Gaussian smoothing.
    
    Args:
        pred_logits: Predicted distogram logits [B, L, L, num_bins]
        min_bin: Minimum distance bin
        max_bin: Maximum distance bin
        num_bins: Number of distance bins
        sigma: Standard deviation for Gaussian smoothing
        
    Returns:
        Expected distances [B, L, L]
    """
    # Convert logits to probabilities
    probs = jax.nn.softmax(pred_logits, axis=-1)  # [B, L, L, num_bins]
    
    # Create bin centers
    bin_width = (max_bin - min_bin) / num_bins
    bin_centers = jnp.arange(num_bins) * bin_width + min_bin + bin_width / 2
    
    # Compute expected distance
    expected_distance = jnp.sum(probs * bin_centers[None, None, None, :], axis=-1)
    
    return expected_distance 