import torch
import pytest
from boltz.model.layers.triangular_attention.primitives import Attention

def test_attention_basic():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Forward pass
    out = attn(q_x, kv_x)
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, c_q)

def test_attention_with_biases():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create bias tensors [batch_size, no_heads, seq_len, seq_len]
    bias1 = torch.randn(batch_size, no_heads, seq_len, seq_len)
    bias2 = torch.randn(batch_size, no_heads, seq_len, seq_len)
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Forward pass with biases
    out = attn(q_x, kv_x, biases=[bias1, bias2])
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, c_q)

def test_attention_no_gating():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create attention module without gating
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=False
    )
    
    # Forward pass
    out = attn(q_x, kv_x)
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, c_q)

def test_attention_different_dimensions():
    # Test parameters
    batch_size = 2
    seq_len_q = 10
    seq_len_kv = 15
    c_q = 32
    c_k = 64
    c_v = 48
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors with different dimensions
    q_x = torch.randn(batch_size, seq_len_q, c_q)
    kv_x = torch.randn(batch_size, seq_len_kv, c_k)
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Forward pass
    out = attn(q_x, kv_x)
    
    # Check output shape
    assert out.shape == (batch_size, seq_len_q, c_q)

def test_attention_memory_efficient():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Forward pass with memory efficient kernel
    out = attn(
        q_x, 
        kv_x,
        use_memory_efficient_kernel=True
    )
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, c_q)

def test_attention_lma():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Forward pass with LMA
    out = attn(
        q_x, 
        kv_x,
        use_lma=True,
        lma_q_chunk_size=4,
        lma_kv_chunk_size=4
    )
    
    # Check output shape
    assert out.shape == (batch_size, seq_len, c_q)

def test_attention_invalid_options():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Test invalid combination of attention options
    with pytest.raises(ValueError):
        attn(
            q_x,
            kv_x,
            use_memory_efficient_kernel=True,
            use_deepspeed_evo_attention=True
        )

def test_attention_too_many_biases():
    # Test parameters
    batch_size = 2
    seq_len = 10
    c_q = 32
    c_k = 32
    c_v = 32
    c_hidden = 16
    no_heads = 4
    
    # Create dummy input tensors
    q_x = torch.randn(batch_size, seq_len, c_q)
    kv_x = torch.randn(batch_size, seq_len, c_k)
    
    # Create multiple bias tensors
    biases = [
        torch.randn(batch_size, no_heads, seq_len, seq_len)
        for _ in range(3)
    ]
    
    # Create attention module
    attn = Attention(
        c_q=c_q,
        c_k=c_k,
        c_v=c_v,
        c_hidden=c_hidden,
        no_heads=no_heads,
        gating=True
    )
    
    # Test with too many biases for memory efficient kernel
    with pytest.raises(ValueError):
        attn(
            q_x,
            kv_x,
            biases=biases,
            use_memory_efficient_kernel=True
        )

if __name__ == "__main__":
    pytest.main([__file__]) 