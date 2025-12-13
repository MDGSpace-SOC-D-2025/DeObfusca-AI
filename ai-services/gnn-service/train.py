"""
Training script for GNN sanitizer.

This script trains a Gated Graph Neural Network to detect junk instructions
in obfuscated binaries.

Dataset: OLLVM obfuscated binaries with ground truth labels
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
import os
import json
from tqdm import tqdm
import pickle

from app import GNNSanitizer

class ObfuscatedBinaryDataset(Dataset):
    """
    Custom dataset for obfuscated binary programs.
    
    Each sample contains:
    - P-Code instructions as node features
    - CFG edges
    - Ground truth labels (0=real, 1=junk)
    """
    def __init__(self, root, transform=None):
        super(ObfuscatedBinaryDataset, self).__init__(root, transform)
        self.data_files = [f for f in os.listdir(root) if f.endswith('.json')]
    
    def len(self):
        return len(self.data_files)
    
    def get(self, idx):
        # Load sample
        with open(os.path.join(self.root, self.data_files[idx])) as f:
            sample = json.load(f)
        
        # Convert to graph
        x = torch.tensor(sample['node_features'], dtype=torch.float)
        edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
        y = torch.tensor(sample['labels'], dtype=torch.long)
        
        return Data(x=x, edge_index=edge_index, y=y)

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(loader, desc='Training'):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index)
        loss = criterion(out, batch.y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += (pred == batch.y).sum().item()
        total += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validation'):
            batch = batch.to(device)
            
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def main():
    # Config
    DATASET_PATH = '/data/training/obfuscated_binaries'
    MODEL_SAVE_PATH = '/app/models/gnn_sanitizer.pth'
    EMBEDDING_PATH = '/app/models/instruction_embeddings.pkl'
    
    INPUT_DIM = 100  # Instruction embedding dimension
    HIDDEN_DIM = 128
    OUTPUT_DIM = 2  # Binary classification
    NUM_LAYERS = 6
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Load dataset
    dataset = ObfuscatedBinaryDataset(DATASET_PATH)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = GNNSanitizer(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        num_layers=NUM_LAYERS
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model saved with validation accuracy: {val_acc:.4f}")
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main()
