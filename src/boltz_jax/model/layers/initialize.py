"""JAX implementation of initialization functions for neural network layers."""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


def gating_init(shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32,
                rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """Initialize weights for gating connections with small values.
    
    Args:
        shape: Shape of the parameter
        dtype: Data type of the parameter
        rng: Random number generator key
        
    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    # Initialize with small values near zero
    std = 1e-3
    limit = np.sqrt(3.0) * std
    return jax.random.uniform(rng, shape, dtype, -limit, limit)


def final_init(shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32,
               rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """Initialize weights for final layer with small values.
    
    Args:
        shape: Shape of the parameter
        dtype: Data type of the parameter
        rng: Random number generator key
        
    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    # Initialize with small values near zero (smaller than gating)
    std = 1e-4
    limit = np.sqrt(3.0) * std
    return jax.random.uniform(rng, shape, dtype, -limit, limit)


def glorot_uniform(shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32,
                  rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """Initialize weights with Glorot uniform initialization (Xavier uniform).
    
    Args:
        shape: Shape of the parameter
        dtype: Data type of the parameter
        rng: Random number generator key
        
    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    # Calculate fan_in and fan_out
    fan_in = shape[0] if len(shape) == 2 else np.prod(shape[:-1])
    fan_out = shape[1] if len(shape) == 2 else shape[-1]
    
    # Calculate limits for uniform distribution
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(rng, shape, dtype, -limit, limit)


def orthogonal(shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32,
               rng: Optional[jax.random.PRNGKey] = None, gain: float = 1.0) -> jnp.ndarray:
    """Initialize weights with orthogonal initialization.
    
    Args:
        shape: Shape of the parameter
        dtype: Data type of the parameter
        rng: Random number generator key
        gain: Scaling factor for the orthogonal matrix
        
    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    # Generate a random matrix
    a = jax.random.normal(rng, (shape[0], shape[1]))
    
    # Compute the QR factorization
    q, r = jnp.linalg.qr(a)
    
    # Make q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = jnp.diag(r)
    ph = jnp.sign(d)
    q = q * ph
    
    # Reshape to the desired shape
    q = q * gain
    return q.astype(dtype)


def time_embedding_init(shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32,
                       rng: Optional[jax.random.PRNGKey] = None) -> jnp.ndarray:
    """Initialize weights for time embedding.
    
    Args:
        shape: Shape of the parameter
        dtype: Data type of the parameter
        rng: Random number generator key
        
    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    # Using higher variance initialization for time embeddings
    std = 0.02
    return std * jax.random.normal(rng, shape, dtype)


def normal_init(shape: Tuple[int, ...], dtype: jnp.dtype = jnp.float32,
               rng: Optional[jax.random.PRNGKey] = None, std: float = 0.01) -> jnp.ndarray:
    """Initialize weights with normal distribution.
    
    Args:
        shape: Shape of the parameter
        dtype: Data type of the parameter
        rng: Random number generator key
        std: Standard deviation of the normal distribution
        
    Returns:
        The initialized parameters
    """
    if rng is None:
        rng = jax.random.PRNGKey(0)
        
    return std * jax.random.normal(rng, shape, dtype)


class GatingDense(nn.Module):
    """Dense layer with gating initialization for better training stability.
    
    Attributes:
        features: Number of output features
        use_bias: Whether to include a bias term
        kernel_init: Initializer for the kernel
        bias_init: Initializer for the bias
        precision: Numerical precision
    """
    features: int
    use_bias: bool = True
    kernel_init: Callable = gating_init
    bias_init: Callable = nn.initializers.zeros
    precision: Any = None
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply the gating Dense layer.
        
        Args:
            inputs: Input array
            
        Returns:
            Output array
        """
        inputs = jnp.asarray(inputs, jnp.float32)
        kernel = self.param('kernel',
                           self.kernel_init,
                           (inputs.shape[-1], self.features))
        
        y = jnp.dot(inputs, kernel)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
            y = y + bias
        return y


class FinalDense(nn.Module):
    """Dense layer with final-layer initialization (smaller values).
    
    Attributes:
        features: Number of output features
        use_bias: Whether to include a bias term
        kernel_init: Initializer for the kernel
        bias_init: Initializer for the bias
        precision: Numerical precision
    """
    features: int
    use_bias: bool = True
    kernel_init: Callable = final_init
    bias_init: Callable = nn.initializers.zeros
    precision: Any = None
    
    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Apply the final Dense layer.
        
        Args:
            inputs: Input array
            
        Returns:
            Output array
        """
        inputs = jnp.asarray(inputs, jnp.float32)
        kernel = self.param('kernel',
                           self.kernel_init,
                           (inputs.shape[-1], self.features))
        
        y = jnp.dot(inputs, kernel)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,))
            y = y + bias
        return y


# Function versions of the initializers to use with Flax's standard layers
def make_gating_init(scale: float = 1e-3) -> Callable:
    """Create a gating initializer function with specified scale.
    
    Args:
        scale: Scale factor for initialization
        
    Returns:
        Initializer function
    """
    def init_fn(key, shape, dtype=jnp.float32):
        std = scale
        limit = np.sqrt(3.0) * std
        return jax.random.uniform(key, shape, dtype, -limit, limit)
    return init_fn 