

import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import D3PMCodeGenerator, CodeTokenizer, create_diffusion_model


class CodeGenerationDataset(Dataset):

    
    def __init__(
        self,
        split: str = 'train',
        max_code_len: int = 512,
        max_pcode_len: int = 256
    ):
        self.data_dir = Path(data_dir)
        self.code_tokenizer = code_tokenizer
        self.pcode_vocab = pcode_vocab
        self.split = split
        self.max_code_len = max_code_len
        self.max_pcode_len = max_pcode_len
        
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self) -> List[Dict]:

        samples = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
        
        # Load JSON files
        for json_file in split_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get P-Code operations
        pcode_ops = sample.get('pcode', sample.get('instructions', []))
        
        # Encode P-Code mnemonics
        pcode_tokens = []
        for op in pcode_ops[:self.max_pcode_len]:
            if isinstance(op, dict):
                mnemonic = op.get('mnemonic', 'UNKNOWN').upper()
            else:
                mnemonic = str(op).upper()
            
            token_id = self.pcode_vocab.get(mnemonic, self.pcode_vocab.get('<UNK>', 1))
            pcode_tokens.append(token_id)
        
        # Pad P-Code
        while len(pcode_tokens) < self.max_pcode_len:
            pcode_tokens.append(self.pcode_vocab.get('<PAD>', 0))
        
        pcode_tensor = torch.tensor(pcode_tokens[:self.max_pcode_len], dtype=torch.long)
        
        # Get target C code
        code = sample.get('code', sample.get('source', ''))
        
        # Encode C code
        code_ids = self.code_tokenizer.encode(
            code,
            max_length=self.max_code_len,
            add_special_tokens=True
        )
        code_tensor = torch.tensor(code_ids, dtype=torch.long)
        
        # Create masks
        pcode_mask = (pcode_tensor == self.pcode_vocab.get('<PAD>', 0))
        code_mask = (code_tensor == self.code_tokenizer.pad_token_id)
        
        return {
            'pcode_tokens': pcode_tensor,
            'pcode_mask': pcode_mask,
            'code_tokens': code_tensor,
            'code_mask': code_mask
        }


class DiffusionTrainer:

    
    def __init__(
        self,
        model: D3PMCodeGenerator,
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
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.98)
        )
        
        # Learning rate scheduler
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
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch + 1}/{self.config['epochs']}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pcode_tokens = batch['pcode_tokens'].to(self.device)
            pcode_mask = batch['pcode_mask'].to(self.device)
            code_tokens = batch['code_tokens'].to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                logits, loss = self.model(
                    code_tokens=code_tokens,
                    pcode_tokens=pcode_tokens,
                    pcode_mask=pcode_mask
                )
            
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
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            pcode_tokens = batch['pcode_tokens'].to(self.device)
            pcode_mask = batch['pcode_mask'].to(self.device)
            code_tokens = batch['code_tokens'].to(self.device)
            
            logits, loss = self.model(
                code_tokens=code_tokens,
                pcode_tokens=pcode_tokens,
                pcode_mask=pcode_mask
            )
            
            total_loss += loss.item()
            num_batches += 1
        
        return {'loss': total_loss / num_batches}
    
    def train(self) -> Dict[str, List[float]]:
        print(f"Training on {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.config['epochs']):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            # Log
            for k, v in train_metrics.items():
                self.history[f'train_{k}'].append(v)
            for k, v in val_metrics.items():
                self.history[f'val_{k}'].append(v)
            
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            
            # Checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model!")
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch + 1}.pth')
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        self.save_checkpoint('final_model.pth')
        
        with open(self.output_dir / 'history.json', 'w') as f:
            json.dump({k: [float(v) for v in vals] for k, vals in self.history.items()}, f)
        
        return dict(self.history)
    
    def save_checkpoint(self, filename: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': dict(self.history)
        }, self.output_dir / filename)


def create_pcode_vocab() -> Dict[str, int]:
    mnemonics = [
        '<PAD>', '<UNK>', '<START>', '<END>',
        'COPY', 'LOAD', 'STORE', 'PIECE', 'SUBPIECE',
        'INT_ADD', 'INT_SUB', 'INT_MULT', 'INT_DIV', 'INT_SDIV',
        'INT_REM', 'INT_SREM', 'INT_NEGATE',
        'INT_AND', 'INT_OR', 'INT_XOR', 'INT_NOT',
        'INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT',
        'INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS', 'INT_SLESS',
        'INT_LESSEQUAL', 'INT_SLESSEQUAL',
        'BOOL_AND', 'BOOL_OR', 'BOOL_XOR', 'BOOL_NEGATE',
        'FLOAT_ADD', 'FLOAT_SUB', 'FLOAT_MULT', 'FLOAT_DIV',
        'BRANCH', 'CBRANCH', 'BRANCHIND', 'CALL', 'CALLIND', 'RETURN',
        'PUSH', 'POP', 'MOV', 'LEA', 'NOP', 'JMP',
        'CMP', 'TEST', 'XOR', 'AND', 'OR', 'ADD', 'SUB', 'MUL', 'DIV',
    ]
    
    return {m: i for i, m in enumerate(mnemonics)}


def create_synthetic_data(output_dir: str, num_samples: int = 1000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # Template patterns
    patterns = [
        {
            'pcode': ['LOAD', 'INT_ADD', 'STORE', 'RETURN'],
            'code': 'int add(int a, int b) {\n    return a + b;\n}'
        },
        {
            'pcode': ['LOAD', 'INT_SUB', 'STORE', 'RETURN'],
            'code': 'int subtract(int a, int b) {\n    return a - b;\n}'
        },
        {
            'pcode': ['LOAD', 'INT_MULT', 'STORE', 'RETURN'],
            'code': 'int multiply(int a, int b) {\n    return a * b;\n}'
        },
        {
            'pcode': ['LOAD', 'CBRANCH', 'INT_ADD', 'BRANCH', 'INT_SUB', 'RETURN'],
            'code': 'int conditional(int a, int b, int cond) {\n    if (cond) {\n        return a + b;\n    }\n    return a - b;\n}'
        },
        {
            'pcode': ['LOAD', 'INT_LESS', 'CBRANCH', 'INT_ADD', 'BRANCH', 'RETURN'],
            'code': 'int loop_sum(int n) {\n    int sum = 0;\n    for (int i = 0; i < n; i++) {\n        sum += i;\n    }\n    return sum;\n}'
        },
    ]
    
    for i in range(num_samples):
        pattern = random.choice(patterns)
        
        # Add variations
        pcode = pattern['pcode'].copy()
        if random.random() < 0.3:
            pcode.insert(random.randint(0, len(pcode)), 'NOP')
        
        samples.append({
            'pcode': pcode,
            'code': pattern['code'],
            'function_name': f'func_{i:04d}'
        })
    
    # Split
    random.shuffle(samples)
    n_train = int(0.8 * len(samples))
    n_val = int(0.1 * len(samples))
    
    for split, split_samples in [
        ('train', samples[:n_train]),
        ('val', samples[n_train:n_train + n_val]),
        ('test', samples[n_train + n_val:])
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        with open(split_dir / 'data.json', 'w') as f:
            json.dump(split_samples, f)
    
    print(f"Created synthetic dataset: {n_train} train, {n_val} val, {len(samples) - n_train - n_val} test")
    
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, default='./data/diffusion')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/diffusion')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=10)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--create_synthetic', action='store_true')
    parser.add_argument('--num_synthetic', type=int, default=1000)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.create_synthetic:
        args.data_dir = str(create_synthetic_data(args.data_dir, args.num_synthetic))
    
    # Create tokenizers
    code_tokenizer = CodeTokenizer()
    pcode_vocab = create_pcode_vocab()
    
    # Save tokenizers
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    code_tokenizer.save(Path(args.output_dir) / 'code_tokenizer.json')
    with open(Path(args.output_dir) / 'pcode_vocab.json', 'w') as f:
        json.dump(pcode_vocab, f)
    
    # Create datasets
    train_dataset = CodeGenerationDataset(args.data_dir, code_tokenizer, pcode_vocab, 'train')
    val_dataset = CodeGenerationDataset(args.data_dir, code_tokenizer, pcode_vocab, 'val')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = create_diffusion_model(
        code_vocab_size=code_tokenizer.current_vocab_size,
        pcode_vocab_size=len(pcode_vocab),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    config = {
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'patience': args.patience
    }
    
    trainer = DiffusionTrainer(model, train_loader, val_loader, config, args.output_dir)
    trainer.train()
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
