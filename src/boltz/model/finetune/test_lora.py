from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

from boltz.model.modules.diffusion import DiffusionModule
from boltz.model.modules.lora import LoRADiffusionModule


def create_dummy_data(batch_size=4, seq_len=32, token_s=384, token_z=128, atom_s=384, atom_z=128):
    """Create dummy data for testing"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy inputs with the correct dimensions for SingleConditioning
    # s_inputs and s_trunk should each have dimension token_s
    # They get concatenated inside SingleConditioning to make the full input
    s_inputs = torch.randn(batch_size, seq_len, token_s, device=device)
    s_trunk = torch.randn(batch_size, seq_len, token_s, device=device)
    z_trunk = torch.randn(batch_size, seq_len, seq_len, token_z, device=device)
    r_noisy = torch.randn(batch_size * seq_len, 3, device=device)  # 3D coordinates
    times = torch.randn(batch_size, device=device)
    relative_position_encoding = torch.randn(batch_size, seq_len, seq_len, token_z, device=device)
    
    # Create dummy features
    feats = {
        "coords": torch.randn(batch_size * seq_len, 3, device=device),
        "atom_pad_mask": torch.ones(batch_size, seq_len, device=device),
        "token_pad_mask": torch.ones(batch_size, seq_len, device=device),
        "atom_resolved_mask": torch.ones(batch_size, seq_len, device=device),
        "atom_to_token": torch.eye(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1, 1),
        "mol_type": torch.zeros(batch_size, seq_len, device=device),
    }
    
    return {
        "s_inputs": s_inputs,
        "s_trunk": s_trunk,
        "z_trunk": z_trunk,
        "r_noisy": r_noisy,
        "times": times,
        "relative_position_encoding": relative_position_encoding,
        "feats": feats,
        "multiplicity": 1,
    }


def test_lora_forward_pass():
    """Test forward pass with LoRA-adapted model"""
    print("Testing forward pass with LoRA-adapted model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU")
    
    # Create a small diffusion module
    model = DiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
    ).to(device)
    
    # Create LoRA-adapted model
    lora_model = LoRADiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ).to(device)
    
    # Copy weights from original model
    lora_model.load_state_dict(model.state_dict(), strict=False)
    
    # Create dummy data
    dummy_data = create_dummy_data(batch_size=2, seq_len=16)
    
    # Forward pass with original model
    with torch.no_grad():
        original_output = model(**dummy_data)
    
    # Forward pass with LoRA model
    with torch.no_grad():
        lora_output = lora_model(**dummy_data)
    
    # Check that outputs have the same shape
    assert original_output["r_update"].shape == lora_output["r_update"].shape, "Output shapes don't match"
    assert original_output["token_a"].shape == lora_output["token_a"].shape, "Token shapes don't match"
    
    # Check that outputs are different (LoRA should modify the output)
    assert not torch.allclose(original_output["r_update"], lora_output["r_update"]), "LoRA should modify the output"
    
    print("Forward pass test passed!")


def test_lora_training():
    """Test training with LoRA-adapted model"""
    print("Testing training with LoRA-adapted model...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU")
    
    # Create a small diffusion module
    model = DiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
    ).to(device)
    
    # Create LoRA-adapted model
    lora_model = LoRADiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ).to(device)
    
    # Copy weights from original model
    lora_model.load_state_dict(model.state_dict(), strict=False)
    
    # Create optimizer for LoRA parameters only
    optimizer = optim.AdamW(
        [p for n, p in lora_model.named_parameters() if "lora" in n],
        lr=1e-4,
    )
    
    # Create dummy data
    dummy_data = create_dummy_data(batch_size=2, seq_len=16)
    
    # Get initial output
    with torch.no_grad():
        initial_output = lora_model(**dummy_data)
        initial_loss = initial_output["loss"]
    
    # Train for a few steps
    for i in range(5):
        # Forward pass
        output = lora_model(**dummy_data)
        loss = output["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    # Get final output
    with torch.no_grad():
        final_output = lora_model(**dummy_data)
        final_loss = final_output["loss"]
    
    # Check that loss decreased
    assert final_loss < initial_loss, "Loss should decrease during training"
    
    print("Training test passed!")


def test_lora_save_load():
    """Test saving and loading LoRA weights"""
    print("Testing saving and loading LoRA weights...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA not available, using CPU")
    
    # Create a small diffusion module
    model = DiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
    ).to(device)
    
    # Create LoRA-adapted model
    lora_model = LoRADiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ).to(device)
    
    # Copy weights from original model
    lora_model.load_state_dict(model.state_dict(), strict=False)
    
    # Create optimizer for LoRA parameters only
    optimizer = optim.AdamW(
        [p for n, p in lora_model.named_parameters() if "lora" in n],
        lr=1e-4,
    )
    
    # Create dummy data
    dummy_data = create_dummy_data(batch_size=2, seq_len=16)
    
    # Train for a few steps
    for i in range(5):
        # Forward pass
        output = lora_model(**dummy_data)
        loss = output["loss"]
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Step {i+1}, Loss: {loss.item():.4f}")
    
    # Get output after training
    with torch.no_grad():
        trained_output = lora_model(**dummy_data)
    
    # Save LoRA weights
    lora_model.save_lora_weights("test_lora_weights.pt")
    
    # Create a new LoRA model
    new_lora_model = LoRADiffusionModule(
        token_s=384,
        token_z=128,
        atom_s=384,
        atom_z=128,
        atoms_per_window_queries=32,
        atoms_per_window_keys=128,
        sigma_data=16,
        dim_fourier=256,
        atom_encoder_depth=3,
        atom_encoder_heads=4,
        token_transformer_depth=6,  # Reduced for testing
        token_transformer_heads=8,
        atom_decoder_depth=3,
        atom_decoder_heads=4,
        atom_feature_dim=128,
        conditioning_transition_layers=2,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.0,
    ).to(device)
    
    # Copy weights from original model
    new_lora_model.load_state_dict(model.state_dict(), strict=False)
    
    # Load LoRA weights
    new_lora_model.load_lora_weights("test_lora_weights.pt")
    
    # Get output with loaded weights
    with torch.no_grad():
        loaded_output = new_lora_model(**dummy_data)
    
    # Check that outputs match
    assert torch.allclose(trained_output["r_update"], loaded_output["r_update"]), "Outputs don't match after loading"
    assert torch.allclose(trained_output["token_a"], loaded_output["token_a"]), "Token outputs don't match after loading"
    
    # Clean up
    if os.path.exists("test_lora_weights.pt"):
        os.remove("test_lora_weights.pt")
    
    print("Save/load test passed!")


def main():
    """Run all tests"""
    print("Running LoRA tests...")
    
    # Test forward pass
    test_lora_forward_pass()
    
    # Test training
    test_lora_training()
    
    # Test saving and loading
    test_lora_save_load()
    
    print("All tests passed!")


if __name__ == "__main__":
    main() 