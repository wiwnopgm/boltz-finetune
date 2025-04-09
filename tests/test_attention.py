import torch
import pytest
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode
)

def test_triangle_attention_basic():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_in = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensor [batch_size, seq_len, seq_len, c_in]
    x = torch.randn(batch_size, seq_len, seq_len, c_in)
    
    # Create mask [batch_size, seq_len, seq_len]
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    # Test starting node attention
    starting_attn = TriangleAttentionStartingNode(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads
    )
    
    # Forward pass
    out_starting = starting_attn(x, mask=mask)
    
    # Check output shape
    assert out_starting.shape == (batch_size, seq_len, seq_len, c_in)
    
    # Test ending node attention
    ending_attn = TriangleAttentionEndingNode(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads
    )
    
    # Forward pass
    out_ending = ending_attn(x, mask=mask)
    
    # Check output shape
    assert out_ending.shape == (batch_size, seq_len, seq_len, c_in)

def test_triangle_attention_chunked():
    # Test parameters
    batch_size = 2
    seq_len = 20
    c_in = 32
    c_hidden = 16
    no_heads = 4
    chunk_size = 4
    
    # Create dummy input tensor
    x = torch.randn(batch_size, seq_len, seq_len, c_in)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    # Create attention module
    attn = TriangleAttentionStartingNode(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads
    )
    
    # Test with chunking
    out_chunked = attn(x, mask=mask, chunk_size=chunk_size)
    
    # Test without chunking
    out_unchunked = attn(x, mask=mask)
    
    # Check shapes match
    assert out_chunked.shape == out_unchunked.shape
    
    # Check outputs are different (since attention has random weights)
    assert not torch.allclose(out_chunked, out_unchunked, atol=1e-6)

def test_triangle_attention_memory_efficient():
    # Test parameters
    batch_size = 2
    seq_len = 16
    c_in = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensor
    x = torch.randn(batch_size, seq_len, seq_len, c_in)
    mask = torch.ones(batch_size, seq_len, seq_len)
    
    # Create attention module
    attn = TriangleAttentionStartingNode(
        c_in=c_in,
        c_hidden=c_hidden,
        no_heads=no_heads
    )
    
    # Test with memory efficient kernel
    out_mem_eff = attn(
        x, 
        mask=mask,
        use_memory_efficient_kernel=True
    )
    
    # Test without memory efficient kernel
    out_normal = attn(x, mask=mask)
    
    # Check shapes match
    assert out_mem_eff.shape == out_normal.shape
    
    # Check outputs are different (since attention has random weights)
    assert not torch.allclose(out_mem_eff, out_normal, atol=1e-6)

if __name__ == "__main__":
    pytest.main([__file__]) 