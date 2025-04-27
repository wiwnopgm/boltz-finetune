import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from boltz.model.modules.lora import (
    LoRALinear,
    LoRAEmbedding,
    LoRAAttentionPairBias,
    LoRADiffusionModule
)

def create_example_dataset(num_samples=1000, seq_length=10, feature_dim=64):
    """Create a dummy dataset for fine-tuning demonstration"""
    # Input features
    X = torch.randn(num_samples, seq_length, feature_dim)
    
    # Target outputs (for example, binary classification)
    y = torch.randint(0, 2, (num_samples,))
    
    return TensorDataset(X, y)

def convert_model_to_lora(model, lora_rank=8, lora_alpha=16, lora_dropout=0.1):
    """
    Convert a standard model to use LoRA for efficient fine-tuning.
    This shows a pattern for adapting existing models to use LoRA.
    """
    # Replace Linear layers with LoRALinear
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Replace with LoRA equivalent
            setattr(model, name, LoRALinear(
                in_features=module.in_features, 
                out_features=module.out_features,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            ))
            # Copy original weights
            model._modules[name].weight.data.copy_(module.weight.data)
            if module.bias is not None:
                model._modules[name].bias.data.copy_(module.bias.data)
                
        # Replace Embedding layers with LoRAEmbedding
        elif isinstance(module, nn.Embedding):
            setattr(model, name, LoRAEmbedding(
                num_embeddings=module.num_embeddings,
                embedding_dim=module.embedding_dim,
                r=lora_rank,
                lora_alpha=lora_alpha
            ))
            # Copy original weights
            model._modules[name].weight.data.copy_(module.weight.data)
            
        # Recursively apply to sub-modules
        elif isinstance(module, nn.Module):
            convert_model_to_lora(module, lora_rank, lora_alpha, lora_dropout)
    
    return model

def get_trainable_params(model):
    """Get only the trainable parameters from a model with LoRA layers"""
    return [p for p in model.parameters() if p.requires_grad]

def fine_tune(model, train_dataloader, val_dataloader=None, epochs=3, learning_rate=1e-3):
    """Fine-tune a model with LoRA layers"""
    # Only optimize trainable params (LoRA weights)
    optimizer = optim.Adam(get_trainable_params(model), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {total_loss/len(train_dataloader):.4f}")
        
        # Validation
        if val_dataloader:
            model.eval()
            val_loss = 0
            correct = 0
            
            with torch.no_grad():
                for data, target in val_dataloader:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
            
            val_loss /= len(val_dataloader)
            accuracy = 100. * correct / len(val_dataloader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
            
            model.train()
    
    return model

def save_lora_weights(model, path):
    """Save only the LoRA weights from a model"""
    lora_state_dict = {}
    
    # Collect LoRA weights
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding, LoRAAttentionPairBias)):
            # Save LoRA A and B matrices
            if hasattr(module, 'lora_A'):
                lora_state_dict[f"{name}.lora_A"] = module.lora_A
            if hasattr(module, 'lora_B'):
                lora_state_dict[f"{name}.lora_B"] = module.lora_B
                
            # For attention, save multiple LoRA matrices
            if isinstance(module, LoRAAttentionPairBias):
                for lora_key in ['lora_q_A', 'lora_q_B', 'lora_k_A', 'lora_k_B', 'lora_v_A', 'lora_v_B']:
                    if hasattr(module, lora_key):
                        lora_state_dict[f"{name}.{lora_key}"] = getattr(module, lora_key)
    
    # Save to disk
    torch.save(lora_state_dict, path)
    print(f"LoRA weights saved to {path}")

def load_lora_weights(model, path):
    """Load LoRA weights into a model with LoRA layers"""
    lora_state_dict = torch.load(path)
    
    # Load weights into model
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAEmbedding, LoRAAttentionPairBias)):
            # Load LoRA A and B matrices
            if hasattr(module, 'lora_A') and f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            if hasattr(module, 'lora_B') and f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
                
            # For attention, load multiple LoRA matrices
            if isinstance(module, LoRAAttentionPairBias):
                for lora_key in ['lora_q_A', 'lora_q_B', 'lora_k_A', 'lora_k_B', 'lora_v_A', 'lora_v_B']:
                    if hasattr(module, lora_key) and f"{name}.{lora_key}" in lora_state_dict:
                        getattr(module, lora_key).data = lora_state_dict[f"{name}.{lora_key}"]
    
    print(f"LoRA weights loaded from {path}")
    return model


if __name__ == "__main__":
    # Define a simple model architecture
    class SimpleModel(nn.Module):
        def __init__(self, input_dim=64, hidden_dim=128, num_classes=2):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, hidden_dim)
            self.activation = nn.ReLU()
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, num_classes)
            
        def forward(self, x):
            # Input shape: [batch_size, seq_length, input_dim]
            # We'll use the first token for classification
            x = x[:, 0, :]
            x = self.linear1(x)
            x = self.activation(x)
            x = self.linear2(x)
            x = self.activation(x)
            x = self.linear3(x)
            return x
    
    # Create a model
    print("Creating base model...")
    model = SimpleModel()
    
    # Print total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    
    # Convert to LoRA
    print("Converting model to use LoRA...")
    model = convert_model_to_lora(model, lora_rank=4, lora_alpha=8)
    
    # Print number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters after LoRA conversion: {trainable_params}")
    print(f"Parameter reduction: {trainable_params/total_params*100:.2f}% of original")
    
    # Create dummy dataset
    print("Creating dataset...")
    dataset = create_example_dataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32)
    
    # Fine-tune with LoRA
    print("Fine-tuning with LoRA...")
    model = fine_tune(model, train_dataloader, val_dataloader, epochs=3)
    
    # Save LoRA weights
    print("Saving LoRA weights...")
    save_lora_weights(model, "lora_weights.pt")
    
    # Create a new model and load LoRA weights
    print("Creating new model and loading LoRA weights...")
    new_model = SimpleModel()
    new_model = convert_model_to_lora(new_model, lora_rank=4, lora_alpha=8)
    new_model = load_lora_weights(new_model, "lora_weights.pt")
    
    # Set model to eval mode (merges LoRA weights)
    print("Setting model to eval mode (merges LoRA weights)...")
    new_model.eval()
    
    print("Done!") 