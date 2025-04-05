from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from boltz.model.modules.lora import LoRADiffusionModule
from boltz.model.modules.diffusion import DiffusionModule


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for diffusion module")
    
    # Model arguments
    parser.add_argument("--token_s", type=int, default=384, help="Token single representation dimension")
    parser.add_argument("--token_z", type=int, default=128, help="Token pair representation dimension")
    parser.add_argument("--atom_s", type=int, default=384, help="Atom single representation dimension")
    parser.add_argument("--atom_z", type=int, default=128, help="Atom pair representation dimension")
    parser.add_argument("--atoms_per_window_queries", type=int, default=32, help="Atoms per window for queries")
    parser.add_argument("--atoms_per_window_keys", type=int, default=128, help="Atoms per window for keys")
    parser.add_argument("--sigma_data", type=int, default=16, help="Standard deviation of data distribution")
    parser.add_argument("--dim_fourier", type=int, default=256, help="Dimension of fourier embedding")
    parser.add_argument("--atom_encoder_depth", type=int, default=3, help="Depth of atom encoder")
    parser.add_argument("--atom_encoder_heads", type=int, default=4, help="Number of heads in atom encoder")
    parser.add_argument("--token_transformer_depth", type=int, default=24, help="Depth of token transformer")
    parser.add_argument("--token_transformer_heads", type=int, default=8, help="Number of heads in token transformer")
    parser.add_argument("--atom_decoder_depth", type=int, default=3, help="Depth of atom decoder")
    parser.add_argument("--atom_decoder_heads", type=int, default=4, help="Number of heads in atom decoder")
    parser.add_argument("--atom_feature_dim", type=int, default=128, help="Atom feature dimension")
    parser.add_argument("--conditioning_transition_layers", type=int, default=2, help="Number of transition layers for conditioning")
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--pretrained_model_path", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--save_lora_path", type=str, default="lora_weights.pt", help="Path to save LoRA weights")
    
    return parser.parse_args()


def load_pretrained_model(args):
    """Load pretrained diffusion module"""
    model = DiffusionModule(
        token_s=args.token_s,
        token_z=args.token_z,
        atom_s=args.atom_s,
        atom_z=args.atom_z,
        atoms_per_window_queries=args.atoms_per_window_queries,
        atoms_per_window_keys=args.atoms_per_window_keys,
        sigma_data=args.sigma_data,
        dim_fourier=args.dim_fourier,
        atom_encoder_depth=args.atom_encoder_depth,
        atom_encoder_heads=args.atom_encoder_heads,
        token_transformer_depth=args.token_transformer_depth,
        token_transformer_heads=args.token_transformer_heads,
        atom_decoder_depth=args.atom_decoder_depth,
        atom_decoder_heads=args.atom_decoder_heads,
        atom_feature_dim=args.atom_feature_dim,
        conditioning_transition_layers=args.conditioning_transition_layers,
    )
    
    if args.pretrained_model_path:
        model.load_state_dict(torch.load(args.pretrained_model_path))
        
    return model


def create_lora_model(pretrained_model, args):
    """Create LoRA-adapted model from pretrained model"""
    lora_model = LoRADiffusionModule(
        token_s=args.token_s,
        token_z=args.token_z,
        atom_s=args.atom_s,
        atom_z=args.atom_z,
        atoms_per_window_queries=args.atoms_per_window_queries,
        atoms_per_window_keys=args.atoms_per_window_keys,
        sigma_data=args.sigma_data,
        dim_fourier=args.dim_fourier,
        atom_encoder_depth=args.atom_encoder_depth,
        atom_encoder_heads=args.atom_encoder_heads,
        token_transformer_depth=args.token_transformer_depth,
        token_transformer_heads=args.token_transformer_heads,
        atom_decoder_depth=args.atom_decoder_depth,
        atom_decoder_heads=args.atom_decoder_heads,
        atom_feature_dim=args.atom_feature_dim,
        conditioning_transition_layers=args.conditioning_transition_layers,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    
    # Copy weights from pretrained model
    lora_model.load_state_dict(pretrained_model.state_dict(), strict=False)
    
    return lora_model


def train_lora_model(model, train_loader, args):
    """Train LoRA-adapted model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Only optimize LoRA parameters
    optimizer = optim.AdamW(
        [p for n, p in model.named_parameters() if "lora" in n],
        lr=args.learning_rate,
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            
            # Compute loss
            loss = outputs["loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
        # Save LoRA weights
        model.save_lora_weights(args.save_lora_path)
        
    return model


def main():
    args = parse_args()
    
    # Load pretrained model
    pretrained_model = load_pretrained_model(args)
    
    # Create LoRA-adapted model
    lora_model = create_lora_model(pretrained_model, args)
    
    # Create dummy data loader (replace with actual data loader)
    # train_loader = DataLoader(...)
    
    # Train LoRA-adapted model
    # trained_model = train_lora_model(lora_model, train_loader, args)
    
    print("LoRA fine-tuning setup complete. Replace the dummy data loader with your actual data loader to start training.")


if __name__ == "__main__":
    main() 