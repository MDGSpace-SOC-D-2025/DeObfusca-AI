

# ...existing code...

import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

# If model.py and other dependencies are in the same directory, ensure import works
sys.path.append(os.path.dirname(__file__))

# Constants (used if not using argparse/main)
EMBED_DIM = 256
NUM_LAYERS = 6
NUM_HEADS = 8
DROPOUT = 0.1
BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
PATIENCE = 5
NUM_WORKERS = 2
KAGGLE_PRESET = 'medium'
SEED = 42

# Load vocabulary
if VOCAB_PATH and Path(VOCAB_PATH).exists():
    vocab = InstructionVocabulary.load(VOCAB_PATH)
else:
    vocab = InstructionVocabulary()
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    vocab.save(Path(OUTPUT_DIR) / 'vocab.pkl')

# Create datasets
train_dataset = JunkInstructionDataset(DATA_DIR, vocab, split='train')
val_dataset = JunkInstructionDataset(DATA_DIR, vocab, split='val', augment=False)

# Create dataloaders
train_loader = PyGDataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# Use JunkInstructionDataset from model.py
        
        # Parse labels (0=real, 1=junk)
        labels = sample.get('labels', [0] * len(instructions))
        # Support label formats where 'labels' may be a mask or string
        if isinstance(labels, str):
            # Not expected; default to zeros
            labels = [0] * len(instructions)
        if len(labels) < len(instructions):
            labels = labels + [0] * (len(instructions) - len(labels))
        y = torch.tensor(labels[:len(instructions)], dtype=torch.long)
        
        # Create positions
        positions = torch.arange(len(instructions), dtype=torch.long)
        
        # Create additional features (64 dimensions)
        additional_features = self._create_additional_features(sample, len(instructions))
        
        # Data augmentation
        if self.augment:
            x, edge_index, edge_type, y, positions, additional_features = self._augment(
                x, edge_index, edge_type, y, positions, additional_features
            )
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            y=y,
            positions=positions,
            additional_features=additional_features,
            num_nodes=len(x)
        )
    
    def _create_sequential_edges(self, num_nodes: int) -> torch.Tensor:
        if num_nodes <= 1:
            return torch.zeros(2, 0, dtype=torch.long)
        sources = list(range(num_nodes - 1))
        targets = list(range(1, num_nodes))
        return torch.tensor([sources, targets], dtype=torch.long)
    
    def _create_additional_features(self, sample: Dict, num_nodes: int) -> torch.Tensor:
        features = torch.zeros(num_nodes, 64)
        inst_types = sample.get('instruction_types', [0] * num_nodes)
        for i, t in enumerate(inst_types[:num_nodes]):
            if t < 8:
                features[i, t] = 1.0
        features[0, 8] = 1.0
        exits = sample.get('exit_nodes', [num_nodes - 1])
        for e in exits:
            if e < num_nodes:
                features[e, 9] = 1.0
        block_ids = sample.get('block_ids', list(range(num_nodes)))
        for i, bid in enumerate(block_ids[:num_nodes]):
            features[i, 10 + (bid % 8)] = 1.0
        edges = sample.get('edges', [])
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        for s, t in edges:
            if t < num_nodes:
                in_degree[t] += 1
            if s < num_nodes:
                out_degree[s] += 1
        for i in range(num_nodes):
            features[i, 18] = min(in_degree[i] / 5.0, 1.0)
            features[i, 19] = min(out_degree[i] / 5.0, 1.0)
        operand_counts = sample.get('operand_counts', [0] * num_nodes)
        for i, count in enumerate(operand_counts[:num_nodes]):
            if count < 12:
                features[i, 20 + count] = 1.0
        return features
    
    def _augment(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        y: torch.Tensor,
        positions: torch.Tensor,
        additional_features: torch.Tensor
    ) -> Tuple:
        """Apply data augmentation."""
        # Random node dropout (10% chance, drop 10% of non-essential nodes)
        if random.random() < 0.1 and len(x) > 10:
            keep_prob = 0.9
            keep_mask = torch.rand(len(x)) < keep_prob
            keep_mask[0] = True  # Always keep entry node
            keep_mask[-1] = True  # Always keep exit node
            
            # Create node mapping
            keep_indices = torch.where(keep_mask)[0]
            node_map = {old.item(): new for new, old in enumerate(keep_indices)}
            
            x = x[keep_mask]
            y = y[keep_mask]
            positions = torch.arange(len(x), dtype=torch.long)
            additional_features = additional_features[keep_mask]
            
            # Remap edges
            valid_edges = []
            valid_types = []
            for i in range(edge_index.size(1)):
                s, t = edge_index[0, i].item(), edge_index[1, i].item()
                if s in node_map and t in node_map:
                    valid_edges.append([node_map[s], node_map[t]])
                    valid_types.append(edge_type[i].item())
            
            if valid_edges:
                edge_index = torch.tensor(valid_edges, dtype=torch.long).t().contiguous()
                edge_type = torch.tensor(valid_types, dtype=torch.long)
            else:
                edge_index = self._create_sequential_edges(len(x))
                edge_type = torch.zeros(edge_index.size(1), dtype=torch.long)
        
        # Random edge dropout (5% chance)
        if random.random() < 0.05 and edge_index.size(1) > 5:
            keep_prob = 0.9
            keep_mask = torch.rand(edge_index.size(1)) < keep_prob
            edge_index = edge_index[:, keep_mask]
            edge_type = edge_type[keep_mask]
        
        return x, edge_index, edge_type, y, positions, additional_features


def collate_fn(batch: List[Data]) -> Batch:
    """Custom collate function for PyG batching."""
    return Batch.from_data_list(batch)


class FocalLoss(nn.Module):

    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        
        # Compute focal weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        return (focal_weight * ce_loss).mean()


class Trainer:

    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: str
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Loss function
        self.criterion = FocalLoss(alpha=0.25, gamma=2.0)
        self.graph_criterion = nn.BCELoss()
        
        # Scheduler
        total_steps = len(train_loader) * config['epochs']
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps,
            anneal_strategy='cos'
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history = defaultdict(list)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        
        total_loss = 0.0
        total_node_loss = 0.0
        total_graph_loss = 0.0
        total_correct = 0
        total_nodes = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config['epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                output = self.model(
                    x=batch.x,
                    edge_index=batch.edge_index,
                    edge_type=batch.edge_type,
                    positions=batch.positions,
                    additional_features=batch.additional_features,
                    batch=batch.batch
                )
                
                # Node-level loss
                node_loss = self.criterion(output['node_logits'], batch.y)
                
                # Graph-level loss (if labels available)
                if hasattr(batch, 'graph_label'):
                    graph_loss = self.graph_criterion(
                        output['graph_score'],
                        batch.graph_label.float()
                    )
                else:
                    # Use ratio of junk nodes as proxy
                    junk_ratios = []
                    for i in range(batch.num_graphs):
                        mask = batch.batch == i
                        junk_ratio = batch.y[mask].float().mean()
                        junk_ratios.append(junk_ratio)
                    
                    proxy_labels = torch.stack(junk_ratios)
                    graph_loss = self.graph_criterion(
                        output['graph_score'],
                        (proxy_labels > 0.1).float()
                    )
                
                # Combined loss
                loss = node_loss + 0.1 * graph_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            total_node_loss += node_loss.item()
            total_graph_loss += graph_loss.item()
            
            preds = output['node_logits'].argmax(dim=-1)
            total_correct += (preds == batch.y).sum().item()
            total_nodes += batch.y.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{total_correct / total_nodes:.4f}"
            })
        
        num_batches = len(self.train_loader)
        return {
            'loss': total_loss / num_batches,
            'node_loss': total_node_loss / num_batches,
            'graph_loss': total_graph_loss / num_batches,
            'accuracy': total_correct / total_nodes
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_nodes = 0
        
        all_preds = []
        all_labels = []
        
        for batch in self.val_loader:
            batch = batch.to(self.device)
            
            output = self.model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                positions=batch.positions,
                additional_features=batch.additional_features,
                batch=batch.batch
            )
            
            # Node-level loss
            loss = self.criterion(output['node_logits'], batch.y)
            total_loss += loss.item()
            
            # Predictions
            preds = output['node_logits'].argmax(dim=-1)
            total_correct += (preds == batch.y).sum().item()
            total_nodes += batch.y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Precision, Recall, F1 for junk class
        junk_preds = all_preds == 1
        junk_labels = all_labels == 1
        
        tp = (junk_preds & junk_labels).sum()
        fp = (junk_preds & ~junk_labels).sum()
        fn = (~junk_preds & junk_labels).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': total_correct / total_nodes,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self) -> Dict[str, List[float]]:
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config['epochs']):
                print(f"  ✓ New best model saved!")
            else:
                self.patience_counter += 1
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Save history
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.history.items()}, f)
        
        return dict(self.history)
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': dict(self.history)
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from {path}")


def create_synthetic_data(output_dir: str, num_samples: int = 1000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vocab = InstructionVocabulary()
    samples = []
    
    # Common instruction patterns
    normal_patterns = [
        ['PUSH', 'MOV', 'MOV', 'ADD', 'POP', 'RET'],
        ['LOAD', 'INT_ADD', 'STORE', 'RETURN'],
        ['MOV', 'CMP', 'JE', 'MOV', 'JMP', 'MOV', 'RET'],
        ['PUSH', 'CALL', 'ADD', 'POP', 'RET'],
        ['LOAD', 'INT_MULT', 'INT_ADD', 'STORE', 'RETURN'],
    ]
    
    junk_patterns = [
        ['NOP', 'NOP', 'NOP'],
        ['PUSH', 'POP'],  # Dead push-pop
        ['MOV', 'MOV'],  # Redundant mov
        ['XOR', 'XOR'],  # Self-xor pattern
        ['ADD', 'SUB'],  # Arithmetic cancellation
    ]
    
    for i in range(num_samples):
        # Create base function
        base = random.choice(normal_patterns).copy()
        
        # Inject junk instructions (20% chance per position)
        instructions = []
        labels = []
        
        for inst in base:
            # Maybe inject junk before
            if random.random() < 0.2:
                junk = random.choice(junk_patterns)
                instructions.extend(junk)
                labels.extend([1] * len(junk))  # 1 = junk
            
            instructions.append(inst)
            labels.append(0)  # 0 = real
        
        # Create edges (sequential + some random branches)
        edges = [(i, i + 1) for i in range(len(instructions) - 1)]
        edge_types = [0] * len(edges)  # 0 = sequential
        
        # Add some branch edges
        if len(instructions) > 5 and random.random() < 0.3:
            src = random.randint(0, len(instructions) - 3)
            dst = random.randint(src + 2, len(instructions) - 1)
            edges.append((src, dst))
            edge_types.append(1)  # 1 = branch
        
        samples.append({
            'instructions': instructions,
            'edges': edges,
            'edge_types': edge_types,
            'labels': labels,
            'function_name': f'func_{i:04d}'
        })
    
    # Split into train/val/test
    random.shuffle(samples)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]
    
    # Save
    for split, split_samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        with open(split_dir / 'data.json', 'w') as f:
            json.dump(split_samples, f)
    
    # Save vocabulary
    vocab.save(output_dir / 'vocab.pkl')
    
    print(f"Created synthetic dataset:")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    print(f"  Test: {len(test_samples)} samples")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Train GNN Junk Instruction Detector')
    
    parser.add_argument('--data_dir', type=str, default='./data/gnn',
                        help='Path to training data directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/gnn',
                        help='Path to save checkpoints')
    parser.add_argument('--vocab_path', type=str, default=None,
                        help='Path to vocabulary file')
    
    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6,
                        help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='DataLoader num_workers (Kaggle should use small value)')
    parser.add_argument('--create_synthetic', action='store_true',
                        help='Create synthetic data for testing')
    parser.add_argument('--num_synthetic', type=int, default=1000,
                        help='Number of synthetic samples to create')
    parser.add_argument('--kaggle_preset', type=str, default='medium', choices=['small','medium','large'],
                        help='Model size preset for Kaggle runs (small uses less memory)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create synthetic data if requested
    if args.create_synthetic:
        args.data_dir = create_synthetic_data(args.data_dir, args.num_synthetic)
    
    # Load vocabulary
    if args.vocab_path and Path(args.vocab_path).exists():
        vocab = InstructionVocabulary.load(args.vocab_path)
    else:
        vocab = InstructionVocabulary()
        # Save vocabulary
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        vocab.save(Path(args.output_dir) / 'vocab.pkl')
    
    # Create datasets
    train_dataset = JunkInstructionDataset(args.data_dir, vocab, split='train')
    val_dataset = JunkInstructionDataset(args.data_dir, vocab, split='val', augment=False)
    
    # Create dataloaders
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    # Create model (use Kaggle preset if requested)
    try:
        from model import create_kaggle_model
        model = create_kaggle_model(preset=args.kaggle_preset)
    except Exception:
        model = create_model(
            vocab_size=vocab.vocab_size,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout
        )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=args.output_dir
    )
    