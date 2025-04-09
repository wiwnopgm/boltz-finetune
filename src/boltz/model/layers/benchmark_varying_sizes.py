import torch
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import os
from boltz.model.layers.attention import AttentionPairBias

def benchmark_varying_sizes(use_scaled_dot_product=True, batch_sizes=None, seq_lens=None, num_heads=8, hidden_dim=512, num_runs=50):
    """
    Benchmark the attention implementation with varying input sizes.
    
    Parameters
    ----------
    use_scaled_dot_product : bool
        Whether to use scaled_dot_product_attention or the original implementation
    batch_sizes : list
        List of batch sizes to test
    seq_lens : list
        List of sequence lengths to test
    num_heads : int
        Number of attention heads
    hidden_dim : int
        Hidden dimension
    num_runs : int
        Number of runs for benchmarking
        
    Returns
    -------
    dict
        Dictionary with benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [4, 8, 16, 32, 64]
    if seq_lens is None:
        seq_lens = [128, 256, 512, 1024, 2048]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the attention layer
    c_s = hidden_dim
    c_z = 64  # Pairwise dimension
    attention = AttentionPairBias(c_s, c_z, num_heads).to(device)
    
    results = {
        'batch_sizes': batch_sizes,
        'seq_lens': seq_lens,
        'times': np.zeros((len(batch_sizes), len(seq_lens)))
    }
    
    for i, batch_size in enumerate(batch_sizes):
        for j, seq_len in enumerate(seq_lens):
            print(f"Benchmarking B={batch_size}, L={seq_len}...")
            
            # Create input tensors
            s = torch.randn(batch_size, seq_len, c_s, device=device)
            z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device)
            mask = torch.ones(batch_size, seq_len, seq_len, device=device)
            
            # Warm-up run
            with torch.no_grad():
                if use_scaled_dot_product:
                    _ = attention_forward_scaled_dot_product(attention, s, z, mask)
                else:
                    _ = attention_forward_original(attention, s, z, mask)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                torch.cuda.synchronize()
                start_time = time.time()
                
                with torch.no_grad():
                    if use_scaled_dot_product:
                        _ = attention_forward_scaled_dot_product(attention, s, z, mask)
                    else:
                        _ = attention_forward_original(attention, s, z, mask)
                
                torch.cuda.synchronize()
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            results['times'][i, j] = np.mean(times)
    
    return results

def attention_forward_original(attention, s, z, mask):
    """Original attention forward pass."""
    B = s.shape[0]
    
    # Layer norms
    if attention.initial_norm:
        s = attention.norm_s(s)
    
    # Compute projections
    q = attention.proj_q(s).view(B, -1, attention.num_heads, attention.head_dim)
    k = attention.proj_k(s).view(B, -1, attention.num_heads, attention.head_dim)
    v = attention.proj_v(s).view(B, -1, attention.num_heads, attention.head_dim)
    
    # Project z
    z = attention.proj_z(z)
    
    g = attention.proj_g(s).sigmoid()
    
    with torch.autocast("cuda", enabled=False):
        # Compute attention weights
        attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
        attn = attn / (attention.head_dim**0.5) + z.float()
        attn = attn + (1 - mask[:, None, None].float()) * -attention.inf
        attn = attn.softmax(dim=-1)
        
        # Compute output
        o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
    o = o.reshape(B, -1, attention.c_s)
    o = attention.proj_o(g * o)
    
    return o

def attention_forward_scaled_dot_product(attention, s, z, mask):
    """Attention forward pass using scaled_dot_product_attention."""
    B = s.shape[0]
    
    # Layer norms
    if attention.initial_norm:
        s = attention.norm_s(s)
    
    # Compute projections
    q = attention.proj_q(s).view(B, -1, attention.num_heads, attention.head_dim)
    k = attention.proj_k(s).view(B, -1, attention.num_heads, attention.head_dim)
    v = attention.proj_v(s).view(B, -1, attention.num_heads, attention.head_dim)
    
    # Project z
    z = attention.proj_z(z)
    
    g = attention.proj_g(s).sigmoid()
    
    # Reshape for scaled_dot_product_attention
    # q, k, v: [B, N, H, D] -> [B, H, N, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    
    # z: [B, H, N, N] -> [B, H, N, N]
    # mask: [B, N, N] -> [B, 1, 1, N, N]
    attention_mask = mask[:, None, None, :, :].expand(B, attention.num_heads, 1, -1, -1)
    
    with torch.autocast("cuda", enabled=False):
        # First compute the attention scores with the pair bias
        attn = torch.matmul(q.float(), k.float().transpose(-2, -1)) / (attention.head_dim**0.5)
        attn = attn + z.float()
        attn = attn + (1 - attention_mask.float()) * -attention.inf
        
        # Then use scaled_dot_product_attention with the pre-computed attention scores
        o = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None,  # We've already applied the mask in the attention scores
            dropout_p=0.0,
            is_causal=False,
            scale=None,  # We've already scaled in the attention scores
            attn_bias=attn  # Use our pre-computed attention scores with pair bias
        ).to(v.dtype)
    
    # Reshape back: [B, H, N, D] -> [B, N, H*D]
    o = o.transpose(1, 2).reshape(B, -1, attention.c_s)
    o = attention.proj_o(g * o)
    
    return o

def visualize_varying_sizes(original_results, scaled_results):
    """Visualize the benchmark results with varying input sizes."""
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot 1: Execution time comparison for different batch sizes
    batch_sizes = original_results['batch_sizes']
    seq_lens = original_results['seq_lens']
    
    # Create a heatmap for original implementation
    im1 = ax1.imshow(original_results['times'], cmap='viridis', aspect='auto')
    ax1.set_title('Original Implementation')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Batch Size')
    ax1.set_xticks(np.arange(len(seq_lens)))
    ax1.set_yticks(np.arange(len(batch_sizes)))
    ax1.set_xticklabels(seq_lens)
    ax1.set_yticklabels(batch_sizes)
    
    # Add colorbar
    cbar1 = ax1.figure.colorbar(im1, ax=ax1)
    cbar1.ax.set_ylabel('Execution Time (ms)')
    
    # Create a heatmap for scaled_dot_product_attention implementation
    im2 = ax2.imshow(scaled_results['times'], cmap='viridis', aspect='auto')
    ax2.set_title('Scaled Dot Product Attention')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Batch Size')
    ax2.set_xticks(np.arange(len(seq_lens)))
    ax2.set_yticks(np.arange(len(batch_sizes)))
    ax2.set_xticklabels(seq_lens)
    ax2.set_yticklabels(batch_sizes)
    
    # Add colorbar
    cbar2 = ax2.figure.colorbar(im2, ax=ax2)
    cbar2.ax.set_ylabel('Execution Time (ms)')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "attention_varying_sizes.png"))
    print(f"Results saved to {output_dir}/attention_varying_sizes.png")
    
    # Create a figure for speedup
    plt.figure(figsize=(10, 8))
    
    # Calculate speedup
    speedup = original_results['times'] / scaled_results['times']
    
    # Create a heatmap for speedup
    im = plt.imshow(speedup, cmap='RdYlGn', aspect='auto')
    plt.title('Speedup of Scaled Dot Product Attention')
    plt.xlabel('Sequence Length')
    plt.ylabel('Batch Size')
    plt.xticks(np.arange(len(seq_lens)), seq_lens)
    plt.yticks(np.arange(len(batch_sizes)), batch_sizes)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Speedup (x)')
    
    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lens)):
            text = plt.text(j, i, f"{speedup[i, j]:.2f}x", ha="center", va="center", color="black")
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, "attention_speedup_heatmap.png"))
    print(f"Speedup heatmap saved to {output_dir}/attention_speedup_heatmap.png")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    print(f"{'Batch Size':<10} {'Seq Length':<10} {'Original (ms)':<15} {'Scaled (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for i, batch_size in enumerate(batch_sizes):
        for j, seq_len in enumerate(seq_lens):
            original_time = original_results['times'][i, j]
            scaled_time = scaled_results['times'][i, j]
            speedup = original_time / scaled_time
            print(f"{batch_size:<10} {seq_len:<10} {original_time:>10.2f} ms {scaled_time:>10.2f} ms {speedup:>8.2f}x")
    
    print("-" * 80)
    print(f"Average speedup: {np.mean(speedup):.2f}x")
    print(f"Maximum speedup: {np.max(speedup):.2f}x")
    print(f"Minimum speedup: {np.min(speedup):.2f}x")

if __name__ == "__main__":
    # Benchmark with varying batch sizes and sequence lengths
    batch_sizes = [4, 8, 16, 32, 64]
    seq_lens = [128, 256, 512, 1024, 2048]
    
    print("Benchmarking original implementation...")
    original_results = benchmark_varying_sizes(
        use_scaled_dot_product=False,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        num_runs=20
    )
    
    print("Benchmarking scaled_dot_product_attention implementation...")
    scaled_results = benchmark_varying_sizes(
        use_scaled_dot_product=True,
        batch_sizes=batch_sizes,
        seq_lens=seq_lens,
        num_runs=20
    )
    
    # Visualize the results
    visualize_varying_sizes(original_results, scaled_results) 