# Add a function to check for NaNs in tensors
def check_nan(tensor, name, step="", print_stats=False):
    """Helper to check and print info about potential NaN values."""
    if not isinstance(tensor, torch.Tensor):
        return False
        
    has_nan = torch.isnan(tensor).any().item()
    if has_nan:
        nan_count = torch.isnan(tensor).sum().item()
        total_count = tensor.numel()
        print(f"[NaN in Pairformer] {step} - {name}: {nan_count}/{total_count} NaN values ({100*nan_count/total_count:.2f}%)")
        if print_stats:
            # Calculate stats for non-NaN values
            valid_values = tensor[~torch.isnan(tensor)]
            if len(valid_values) > 0:
                print(f"  Valid values - Min: {valid_values.min().item():.6f}, Max: {valid_values.max().item():.6f}, Mean: {valid_values.mean().item():.6f}")
            else:
                print("  No valid values (all NaNs)")
    return has_nan

# Add debugging to the forward method of the Pairformer class
def forward(self, s, z, mask=None, pair_mask=None, is_fixed=False):
    print("\n=== Starting Pairformer forward pass ===")
    # Check input tensors
    check_nan(s, "s", "input")
    check_nan(z, "z", "input")
    if mask is not None:
        check_nan(mask, "mask", "input")
    if pair_mask is not None:
        check_nan(pair_mask, "pair_mask", "input")
    
    # Create attention masks
    if mask is None:
        print("No mask provided, using default")
        s_mask = None
    else:
        print(f"Creating s_mask from mask shape: {mask.shape}")
        s_mask = mask.float().reshape(mask.shape[0], 1, 1, mask.shape[1])
        check_nan(s_mask, "s_mask", "after reshaping")
    
    # Create pair attention masks
    if mask is not None and pair_mask is None:
        print("Creating pair_mask from mask")
        pair_mask = mask.float()[:, :, None] * mask.float()[:, None, :]
        check_nan(pair_mask, "pair_mask", "after creation")
        
    if pair_mask is not None:
        z_mask = pair_mask.float().reshape(
            pair_mask.shape[0], 1, pair_mask.shape[1], pair_mask.shape[2]
        )
        check_nan(z_mask, "z_mask", "after reshaping")
    else:
        print("No pair_mask provided, using default")
        z_mask = None
    
    # Store the input to compute residual
    s_input = s
    check_nan(s_input, "s_input", "stored for residual")
    z_input = z
    check_nan(z_input, "z_input", "stored for residual")
    
    for i, (s_block, z_block) in enumerate(zip(self.s_blocks, self.z_blocks)):
        print(f"\n--- Pairformer block {i+1}/{len(self.s_blocks)} ---")
        
        # Apply s-block (sequence attention)
        print("Running sequence attention block")
        s_block_out = s_block(s, s_mask, bias=None, is_fixed=is_fixed)
        check_nan(s_block_out, "s_block_out", f"after s_block {i}")
        s = s + s_block_out  # Residual connection
        check_nan(s, "s", f"after residual in block {i}")
        
        # Apply z-block (pair attention)
        print("Running pair attention block")
        z_attn_bias = self.s_to_z_attn_bias[i](s)
        check_nan(z_attn_bias, "z_attn_bias", f"after s_to_z_attn_bias in block {i}")
        
        z_block_out = z_block(z, z_mask, bias=z_attn_bias, is_fixed=is_fixed)
        check_nan(z_block_out, "z_block_out", f"after z_block {i}")
        z = z + z_block_out  # Residual connection
        check_nan(z, "z", f"after residual in block {i}")
    
    # Apply the final layer norms
    s = self.s_norm(s)
    check_nan(s, "s", "after final norm")
    z = self.z_norm(z)
    check_nan(z, "z", "after final norm")
    
    # Compute the return values
    s_out = s + s_input  # Global residual
    check_nan(s_out, "s_out", "after global residual")
    z_out = z + z_input  # Global residual
    check_nan(z_out, "z_out", "after global residual")
    
    print("=== Completed Pairformer forward pass ===\n")
    return s_out, z_out

# Now add debugging to the attention block classes
def attention_forward(self, x, mask=None, bias=None, is_fixed=False):
    print(f"Running attention with x shape: {x.shape}")
    
    check_nan(x, "x", "input to attention")
    if mask is not None:
        check_nan(mask, "mask", "input to attention")
    if bias is not None:
        check_nan(bias, "bias", "input to attention")
    
    qkv = self.qkv(x)
    check_nan(qkv, "qkv", "after qkv projection")
    
    q, k, v = qkv.chunk(3, dim=-1)
    check_nan(q, "q", "after chunking")
    check_nan(k, "k", "after chunking")
    check_nan(v, "v", "after chunking")
    
    # Reshape for multi-head attention
    batch_size, seq_len, _ = q.shape
    head_dim = q.shape[-1] // self.n_heads
    
    q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
    k = k.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
    v = v.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
    
    check_nan(q, "q", "after reshaping")
    check_nan(k, "k", "after reshaping")
    check_nan(v, "v", "after reshaping")
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    check_nan(attn_scores, "attn_scores", "after q@k^T")
    
    # Add bias if provided (this is where NaNs might appear)
    if bias is not None:
        attn_scores = attn_scores + bias
        check_nan(attn_scores, "attn_scores", "after adding bias")
    
    # Apply mask if provided
    if mask is not None:
        attn_scores = attn_scores + (1.0 - mask) * -10000.0
        check_nan(attn_scores, "attn_scores", "after masking")
    
    # Apply softmax to get attention weights
    attn_weights = F.softmax(attn_scores, dim=-1)
    check_nan(attn_weights, "attn_weights", "after softmax", print_stats=True)
    
    # Apply dropout
    attn_weights = self.dropout(attn_weights)
    check_nan(attn_weights, "attn_weights", "after dropout")
    
    # Apply attention weights to values
    context = torch.matmul(attn_weights, v)
    check_nan(context, "context", "after attn_weights@v")
    
    # Reshape back
    context = context.transpose(1, 2).reshape(batch_size, seq_len, -1)
    check_nan(context, "context", "after reshaping")
    
    # Apply output projection
    output = self.out_proj(context)
    check_nan(output, "output", "after out_proj")
    
    return output 