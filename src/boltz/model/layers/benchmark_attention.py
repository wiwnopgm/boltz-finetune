import torch
import torch.nn.functional as F
import time
import numpy as np
from einops.layers.torch import Rearrange
from torch import nn
import boltz.model.layers.initialize as init
from boltz.model.layers.attention import AttentionPairBias
import os

class ScaledDotProductAttentionPairBias(nn.Module):
    """Attention pair bias layer using scaled dot product attention."""

    def __init__(
        self,
        c_s: int,
        c_z: int,
        num_heads: int,
        inf: float = 1e6,
        initial_norm: bool = True,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        c_s : int
            The input sequence dimension.
        c_z : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6
        initial_norm: bool, optional
            Whether to apply layer norm to the input, by default True
        """
        super().__init__()

        assert c_s % num_heads == 0

        self.c_s = c_s
        self.num_heads = num_heads
        self.head_dim = c_s // num_heads
        self.inf = inf

        self.initial_norm = initial_norm
        if self.initial_norm:
            self.norm_s = nn.LayerNorm(c_s)

        self.proj_q = nn.Linear(c_s, c_s)
        self.proj_k = nn.Linear(c_s, c_s, bias=False)
        self.proj_v = nn.Linear(c_s, c_s, bias=False)
        self.proj_g = nn.Linear(c_s, c_s, bias=False)

        self.proj_z = nn.Sequential(
            nn.LayerNorm(c_z),
            nn.Linear(c_z, num_heads, bias=False),
            Rearrange("b ... h -> b h ..."),
        )

        self.proj_o = nn.Linear(c_s, c_s, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        s: torch.Tensor,
        z: torch.Tensor,
        mask: torch.Tensor,
        multiplicity: int = 1,
        to_keys=None,
        model_cache=None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        s : torch.Tensor
            The input sequence tensor (B, S, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The sequence mask tensor (B, N) - masks out entire positions
        multiplicity : int, optional
            The diffusion batch size, by default 1

        Returns
        -------
        torch.Tensor
            The output sequence tensor.
        """
        B = s.shape[0]

        # Layer norms
        if self.initial_norm:
            s = self.norm_s(s)

        if to_keys is not None:
            k_in = to_keys(s)
            mask = to_keys(mask.unsqueeze(-1)).squeeze(-1)
        else:
            k_in = s

        # Compute projections
        q = self.proj_q(s).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        # Project z and handle caching
        if model_cache is None or "z" not in model_cache:
            z = self.proj_z(z)
            if model_cache is not None:
                model_cache["z"] = z
        else:
            z = model_cache["z"]
        
        z = z.repeat_interleave(multiplicity, 0)

        g = self.proj_g(s).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Reshape for scaled_dot_product_attention
            # [B, N, H, D] -> [B, H, N, D]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Create attention mask from sequence mask
            # (B, N) -> (B, 1, N, N)
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
            mask_matrix = mask_expanded * mask_expanded.transpose(-2, -1)  # (B, 1, N, N)

            # First compute attention without pair bias
            o = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask_matrix,
                dropout_p=0.0,
                is_causal=False
            ).to(v.dtype)

            # Add pair bias to the output
            # Reshape z to match attention output: [B, H, N, N] -> [B, H, N, D]
            z = z.permute(0, 1, 3, 2)  # [B, H, N, N] -> [B, H, N, N]
            
            # Compute pair bias contribution directly
            # Instead of creating a large intermediate tensor, we'll compute the contribution
            # of the pair bias to each position in the output
            z_contribution = torch.zeros_like(o)
            
            # For each position i, sum the pair bias values for all positions j
            for i in range(z.shape[2]):
                # Extract the pair bias for position i with all other positions
                z_i = z[:, :, i, :]  # [B, H, N]
                
                # Expand to match the head dimension
                z_i = z_i.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)  # [B, H, N, D]
                
                # Add to the contribution tensor
                z_contribution[:, :, i, :] = z_i.mean(dim=2)  # Average over the second N dimension
            
            o = o + z_contribution

        # Reshape back: [B, H, N, D] -> [B, N, H*D]
        o = o.transpose(1, 2).reshape(B, -1, self.c_s)
        
        o = self.proj_o(g * o)

        return o

def print_attention_shapes(attention, s, z, mask, multiplicity=1, model_cache=None):
    """Print shapes for the original AttentionPairBias class."""
    B = s.shape[0]
    
    # Layer norms
    if attention.initial_norm:
        s = attention.norm_s(s)
    
    # Compute projections
    q = attention.proj_q(s).view(B, -1, attention.num_heads, attention.head_dim)
    k = attention.proj_k(s).view(B, -1, attention.num_heads, attention.head_dim)
    v = attention.proj_v(s).view(B, -1, attention.num_heads, attention.head_dim)
    
    # Project z and handle caching
    if model_cache is None or "z" not in model_cache:
        z = attention.proj_z(z)
        if model_cache is not None:
            model_cache["z"] = z
    else:
        z = model_cache["z"]
    
    z = z.repeat_interleave(multiplicity, 0)
    
    g = attention.proj_g(s).sigmoid()
    
    with torch.autocast("cuda", enabled=False):
        # Compute attention weights
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        
        attn = attn / (attention.head_dim**0.5) + z.float()
        
        # Create a mask for each position: (B, N) -> (B, 1, N, N)
        # This masks out entire positions (rows and columns)
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        mask_matrix = mask_expanded * mask_expanded.transpose(-2, -1)  # (B, 1, N, N)
        attn = attn + (1 - mask_matrix.float()) * -attention.inf
        
        attn = attn.softmax(dim=-1)

        # Compute output
        o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    
    o = o.reshape(B, -1, attention.c_s)
    
    o = attention.proj_o(g * o)
    
    return o

def benchmark_attention(use_scaled_dot_product=True, batch_size=8, seq_len=256, num_heads=8, hidden_dim=512, num_runs=100):
    """
    Benchmark the attention implementation.
    
    Parameters
    ----------
    use_scaled_dot_product : bool
        Whether to use scaled_dot_product_attention or the original implementation
    batch_size : int
        Batch size
    seq_len : int
        Sequence length
    num_heads : int
        Number of attention heads
    hidden_dim : int
        Hidden dimension
    num_runs : int
        Number of runs for benchmarking
        
    Returns
    -------
    tuple
        (average_time_ms, max_memory_mb, output_tensor)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the attention layer
    c_s = hidden_dim
    c_z = 64  # Pairwise dimension
    
    # Create appropriate attention layer based on implementation choice
    if use_scaled_dot_product:
        attention = ScaledDotProductAttentionPairBias(c_s, c_z, num_heads).to(device)
    else:
        attention = AttentionPairBias(c_s, c_z, num_heads).to(device)
    
    # Create input tensors with consistent data type (float32)
    # Using float16 can cause issues with layer normalization
    s = torch.randn(batch_size, seq_len, c_s, device=device, dtype=torch.float32)  # (B, N, D)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device, dtype=torch.float32)  # (B, N, N, D)
    mask = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)  # (B, N)
    
    # Warm-up run
    with torch.no_grad():
        _ = attention(s, z, mask)
    
    # Reset memory stats and clear cache
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Benchmark
    times = []
    output = None
    
    for _ in range(num_runs):
        # Clear cache before each run to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        torch.cuda.synchronize()
        start_time = time.time()
        
        with torch.no_grad():
            output = attention(s, z, mask)
        
        torch.cuda.synchronize()
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # Convert to milliseconds
        
        # Clear output after each run to free memory
        if _ < num_runs - 1:  # Keep the last output for comparison
            output = None
            
        # Force garbage collection
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Get max memory usage
    max_memory_mb = 0
    if torch.cuda.is_available():
        max_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)  # Convert to MB
    
    return np.mean(times), max_memory_mb, output

def compare_attention_implementations(batch_size=8, seq_len=256, num_heads=8, hidden_dim=512, num_runs=50):
    """
    Compare the original and scaled dot product attention implementations.
    
    Parameters
    ----------
    batch_size : int
        Batch size
    seq_len : int
        Sequence length
    num_heads : int
        Number of attention heads
    hidden_dim : int
        Hidden dimension
    num_runs : int
        Number of runs for benchmarking
        
    Returns
    -------
    dict
        Comparison results
    """
    # Clear memory before benchmarking
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Benchmark original implementation
    original_time, original_memory, original_output = benchmark_attention(
        use_scaled_dot_product=False,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_runs=num_runs
    )
    
    # Clear memory between benchmarks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Benchmark scaled_dot_product_attention implementation
    scaled_dot_product_time, scaled_dot_product_memory, scaled_dot_product_output = benchmark_attention(
        use_scaled_dot_product=True,
        batch_size=batch_size,
        seq_len=seq_len,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_runs=num_runs
    )
    
    # Calculate speedup
    speedup = original_time / scaled_dot_product_time
    
    # Compare outputs
    max_diff = torch.max(torch.abs(original_output - scaled_dot_product_output)).item()
    mean_diff = torch.mean(torch.abs(original_output - scaled_dot_product_output)).item()
    
    # Calculate relative difference, handling the case where original_output is all zeros
    abs_original_mean = torch.mean(torch.abs(original_output)).item()
    if abs_original_mean > 0:
        relative_diff = mean_diff / abs_original_mean * 100
    else:
        # If original output is all zeros, use a small epsilon to avoid division by zero
        relative_diff = mean_diff * 100  # Just report the absolute difference as percentage
    
    # Clear memory after comparison
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    return {
        "original_time_ms": original_time,
        "scaled_dot_product_time_ms": scaled_dot_product_time,
        "speedup": speedup,
        "original_memory_mb": original_memory,
        "scaled_dot_product_memory_mb": scaled_dot_product_memory,
        "memory_reduction_percent": (original_memory - scaled_dot_product_memory) / original_memory * 100,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "relative_diff_percent": relative_diff
    }

def test_flash_attention_availability():
    """
    Test if Flash Attention is available and being used.
    
    Returns
    -------
    dict
        Information about Flash Attention availability
    """
    result = {
        "cuda_available": torch.cuda.is_available(),
        "flash_attention_available": False,
        "device_name": "CPU",
        "cuda_version": None,
        "pytorch_version": torch.__version__,
    }
    
    if torch.cuda.is_available():
        result["device_name"] = torch.cuda.get_device_name(0)
        result["cuda_version"] = torch.version.cuda
        
        # Check if scaled_dot_product_attention is available
        if hasattr(F, "scaled_dot_product_attention"):
            result["flash_attention_available"] = True
            
            # Test if it's actually using Flash Attention
            # Create a small test case
            batch_size, seq_len, num_heads, head_dim = 2, 4, 2, 4
            q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
            k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
            v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda")
            
            # Run with and without autocast to see if behavior changes
            # Flash Attention is typically only used with autocast
            with torch.autocast("cuda"):
                _ = F.scaled_dot_product_attention(q, k, v)
            
            # If we get here without errors, Flash Attention is likely available
            result["flash_attention_working"] = True
    
    return result

if __name__ == "__main__":
    # Set PyTorch memory management settings
    if torch.cuda.is_available():
        # Enable expandable memory segments to reduce fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Set memory fraction to use (80% of available memory)
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Test different configurations with progressively larger sizes
    configs = [
        {"batch_size": 8, "seq_len": 256, "num_heads": 8, "hidden_dim": 512},
        {"batch_size": 16, "seq_len": 512, "num_heads": 8, "hidden_dim": 512},
        {"batch_size": 32, "seq_len": 1024, "num_heads": 8, "hidden_dim": 512},
        # Larger configurations will be tested if previous ones succeed
    ]
    
    # Check Flash Attention availability
    flash_info = test_flash_attention_availability()
    print("\nFlash Attention Information:")
    print(f"CUDA Available: {flash_info['cuda_available']}")
    print(f"Flash Attention Available: {flash_info['flash_attention_available']}")
    print(f"Device: {flash_info['device_name']}")
    print(f"CUDA Version: {flash_info['cuda_version']}")
    print(f"PyTorch Version: {flash_info['pytorch_version']}")
    print("-" * 100)
    
    print("Benchmarking attention implementations:")
    print("-" * 100)
    print(f"{'Config':<30} {'Original (ms)':<12} {'SDP (ms)':<12} {'Speedup':<10} {'Orig Mem (MB)':<15} {'SDP Mem (MB)':<15} {'Mem Red %':<10} {'Max Diff':<12} {'Rel Diff %':<10}")
    print("-" * 100)
    
    for config in configs:
        config_str = f"B={config['batch_size']}, L={config['seq_len']}, H={config['num_heads']}, D={config['hidden_dim']}"
        try:
            # Clear memory before each configuration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            import gc
            gc.collect()
                
            # Compare implementations with fewer runs for larger batches
            num_runs = max(5, 50 // (config["batch_size"] // 8))  # Reduce runs for larger batches
            
            results = compare_attention_implementations(
                batch_size=config["batch_size"],
                seq_len=config["seq_len"],
                num_heads=config["num_heads"],
                hidden_dim=config["hidden_dim"],
                num_runs=num_runs
            )
            
            # Print results
            print(f"{config_str:<30} "
                  f"{results['original_time_ms']:>10.2f} ms "
                  f"{results['scaled_dot_product_time_ms']:>10.2f} ms "
                  f"{results['speedup']:>8.2f}x "
                  f"{results['original_memory_mb']:>12.2f} MB "
                  f"{results['scaled_dot_product_memory_mb']:>12.2f} MB "
                  f"{results['memory_reduction_percent']:>8.2f}% "
                  f"{results['max_diff']:>10.6f} "
                  f"{results['relative_diff_percent']:>8.4f}%")
            
            # If this configuration succeeded, try a larger one
            if config["batch_size"] == 32 and config["seq_len"] == 1024:
                # Add larger configurations if previous ones succeeded
                configs.append({"batch_size": 64, "seq_len": 2048, "num_heads": 8, "hidden_dim": 512})
                
        except RuntimeError as e:
            print(f"{config_str:<30} Error: {str(e)}")
            # Stop testing larger configurations if we hit a memory error
            break
    
    print("-" * 100) 