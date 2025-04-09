import torch
import sys
import os

# Add the src directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from boltz.model.layers.attention import AttentionPairBias

def test_attention_pair_bias():
    # Set parameters
    batch_size = 2
    seq_len = 4
    c_s = 64  # sequence dimension
    c_z = 32  # pairwise dimension
    num_heads = 4
    
    # Create device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create attention layer
    attention = AttentionPairBias(c_s, c_z, num_heads).to(device)
    
    # Create input tensors
    s = torch.randn(batch_size, seq_len, c_s, device=device)  # (B, N, D)
    z = torch.randn(batch_size, seq_len, seq_len, c_z, device=device)  # (B, N, N, D)
    mask = torch.ones(batch_size, seq_len, device=device)  # (B, N)
    
    print("\nInput shapes:")
    print(f"s shape: {s.shape}")
    print(f"z shape: {z.shape}")
    print(f"mask shape: {mask.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = attention(s, z, mask)
    
    print("\nOutput shape:")
    print(f"output shape: {output.shape}")
    
    # Test with different mask values
    print("\nTesting with different mask values:")
    mask[0, 1] = 0  # Mask out position 1 in first batch
    output_masked = attention(s, z, mask)
    print(f"output_masked shape: {output_masked.shape}")
    
    # Test with multiplicity
    print("\nTesting with multiplicity:")
    output_mult = attention(s, z, mask, multiplicity=2)
    print(f"output_mult shape: {output_mult.shape}")
    
    return output, output_masked, output_mult

if __name__ == "__main__":
    output, output_masked, output_mult = test_attention_pair_bias() 