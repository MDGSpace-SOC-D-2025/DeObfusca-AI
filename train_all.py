#!/usr/bin/env python3
"""
Automated Training Pipeline for DeObfusca-AI

This script trains all models in the correct order:
1. GNN (graph encoder)
2. LLM (fine-tune CodeLlama)
3. Diffusion (code refinement)
4. RL (strategy selection)

Usage:
    python3 train_all.py --data-dir ./training-data --output-dir ./models

Requirements:
    - Preprocessed dataset following DATASET_SPECIFICATION.md
    - At least 32GB RAM
    - GPU with 24GB+ VRAM (recommended)
    - ~200GB disk space for checkpoints
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset as HFDataset

# Import model definitions
sys.path.append('/Users/chayanaggarwal/DeObfusca-AI/ai-services')
from gnn_service.app import GNNSanitizer
from diffusion_service.app import DiffusionCodeGenerator
from rl_service.train_ppo import PPOAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """Central configuration for all training runs."""
    
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GNN config
        self.gnn_config = {
            'input_dim': 128,
            'hidden_dim': 256,
            'output_dim': 768,
            'num_layers': 6,
            'batch_size': 32,
            'epochs': 50,
            'lr': 1e-4,
            'weight_decay': 1e-5
        }
        
        # LLM config
        self.llm_config = {
            'model_name': 'codellama/CodeLlama-7b-hf',
            'batch_size': 4,
            'epochs': 3,
            'lr': 2e-5,
            'max_length': 2048,
            'gradient_accumulation_steps': 8
        }
        
        # Diffusion config
        self.diffusion_config = {
            'vocab_size': 50000,
            'd_model': 768,
            'num_timesteps': 1000,
            'batch_size': 16,
            'epochs': 50,
            'lr': 1e-4,
            'epsilon': 0.1,  # Adversarial perturbation
            'num_adv_steps': 5  # PGD steps
        }
        
        # RL config
        self.rl_config = {
            'state_dim': 128,
            'action_dim': 4,
            'batch_size': 32,
            'num_episodes': 10000,
            'lr': 3e-4,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'K_epochs': 4
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training on device: {self.device}")


# ============================================================================
# Dataset Classes
# ============================================================================

class GNNDataset(Dataset):
    """Dataset for GNN training."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / 'preprocessed' / 'gnn'
        
        # Load splits
        with open(Path(data_dir) / 'splits.json') as f:
            splits = json.load(f)
        self.sample_ids = splits[split]
        
        logger.info(f"Loaded {len(self.sample_ids)} samples for GNN {split}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load preprocessed data
        data_file = self.data_dir / f"{sample_id}.json"
        with open(data_file) as f:
            sample = json.load(f)
        
        # Convert to PyTorch Geometric format
        x = torch.tensor(sample['node_features'], dtype=torch.float)
        edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
        y = torch.tensor(sample['labels'], dtype=torch.long)
        
        # Optional edge attributes
        if 'edge_attr' in sample:
            edge_attr = torch.tensor(sample['edge_attr'], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
        return Data(x=x, edge_index=edge_index, y=y)


class LLMDataset(Dataset):
    """Dataset for LLM fine-tuning."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / 'preprocessed' / 'llm'
        
        with open(Path(data_dir) / 'splits.json') as f:
            splits = json.load(f)
        self.sample_ids = splits[split]
        
        logger.info(f"Loaded {len(self.sample_ids)} samples for LLM {split}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        data_file = self.data_dir / f"{sample_id}.json"
        with open(data_file) as f:
            return json.load(f)


class DiffusionDataset(Dataset):
    """Dataset for diffusion model training."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / 'preprocessed' / 'diffusion'
        
        with open(Path(data_dir) / 'splits.json') as f:
            splits = json.load(f)
        self.sample_ids = splits[split]
        
        logger.info(f"Loaded {len(self.sample_ids)} samples for Diffusion {split}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        data_file = self.data_dir / f"{sample_id}.json"
        with open(data_file) as f:
            sample = json.load(f)
        
        tokens = torch.tensor(sample['tokens'], dtype=torch.long)
        condition = torch.tensor(
            sample['condition']['assembly_embedding'],
            dtype=torch.float
        )
        
        return tokens, condition, tokens  # Use tokens as labels


class RLDataset(Dataset):
    """Dataset for RL training."""
    
    def __init__(self, data_dir, split='train'):
        self.data_dir = Path(data_dir) / 'preprocessed' / 'rl'
        
        with open(Path(data_dir) / 'splits.json') as f:
            splits = json.load(f)
        self.sample_ids = splits[split]
        
        # Load all episodes
        self.episodes = []
        for sample_id in self.sample_ids:
            data_file = self.data_dir / f"{sample_id}.json"
            with open(data_file) as f:
                self.episodes.append(json.load(f))
        
        logger.info(f"Loaded {len(self.episodes)} episodes for RL {split}")
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        return self.episodes[idx]


# ============================================================================
# Training Functions
# ============================================================================

def train_gnn(config):
    """Train Graph Neural Network."""
    logger.info("=" * 80)
    logger.info("STAGE 1: Training GNN")
    logger.info("=" * 80)
    
    # Load datasets
    train_dataset = GNNDataset(config.data_dir, 'train')
    val_dataset = GNNDataset(config.data_dir, 'validation')
    
    train_loader = GeoDataLoader(
        train_dataset,
        batch_size=config.gnn_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = GeoDataLoader(
        val_dataset,
        batch_size=config.gnn_config['batch_size'],
        num_workers=4
    )
    
    # Initialize model
    model = GNNSanitizer(
        input_dim=config.gnn_config['input_dim'],
        hidden_dim=config.gnn_config['hidden_dim'],
        output_dim=config.gnn_config['output_dim'],
        num_layers=config.gnn_config['num_layers']
    ).to(config.device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.gnn_config['lr'],
        weight_decay=config.gnn_config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.gnn_config['epochs']
    )
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(config.gnn_config['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.gnn_config["epochs"]}'):
            batch = batch.to(config.device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index)
            loss = criterion(out, batch.y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.y.size(0)
        
        train_acc = train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(config.device)
                out = model(batch.x, batch.edge_index)
                loss = criterion(out, batch.y)
                
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.y.size(0)
        
        val_acc = val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, "
                   f"Train Acc={train_acc:.4f}, Val Loss={val_loss/len(val_loader):.4f}, "
                   f"Val Acc={val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.output_dir / 'gnn_best.pth')
            logger.info(f"Saved best model with Val Acc={val_acc:.4f}")
        
        scheduler.step()
    
    logger.info(f"GNN training complete. Best Val Acc: {best_val_acc:.4f}")
    return model


def train_llm(config):
    """Fine-tune CodeLlama for decompilation."""
    logger.info("=" * 80)
    logger.info("STAGE 2: Fine-tuning LLM")
    logger.info("=" * 80)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.llm_config['model_name'])
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config.llm_config['model_name'],
        torch_dtype=torch.float16 if config.device.type == 'cuda' else torch.float32,
        device_map='auto'
    )
    
    # Load and prepare dataset
    train_dataset = LLMDataset(config.data_dir, 'train')
    val_dataset = LLMDataset(config.data_dir, 'validation')
    
    def format_sample(sample):
        """Format sample for instruction tuning."""
        prompt = f"""### Task: Decompile the following assembly to C code.

### Assembly:
{sample['assembly']}

### C Code:
"""
        full_text = prompt + sample['source_code']
        
        # Tokenize
        encodings = tokenizer(
            full_text,
            truncation=True,
            max_length=config.llm_config['max_length'],
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze()
        }
    
    # Convert to HuggingFace Dataset
    train_samples = [train_dataset[i] for i in range(len(train_dataset))]
    val_samples = [val_dataset[i] for i in range(len(val_dataset))]
    
    train_hf = HFDataset.from_list(train_samples)
    val_hf = HFDataset.from_list(val_samples)
    
    train_hf = train_hf.map(format_sample, remove_columns=train_hf.column_names)
    val_hf = val_hf.map(format_sample, remove_columns=val_hf.column_names)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir / 'llm_checkpoints'),
        num_train_epochs=config.llm_config['epochs'],
        per_device_train_batch_size=config.llm_config['batch_size'],
        per_device_eval_batch_size=config.llm_config['batch_size'],
        gradient_accumulation_steps=config.llm_config['gradient_accumulation_steps'],
        learning_rate=config.llm_config['lr'],
        warmup_steps=500,
        logging_steps=100,
        eval_steps=1000,
        save_steps=1000,
        save_total_limit=3,
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        fp16=config.device.type == 'cuda',
        report_to='none'
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        tokenizer=tokenizer
    )
    
    # Train
    logger.info("Starting LLM fine-tuning...")
    trainer.train()
    
    # Save final model
    trainer.save_model(str(config.output_dir / 'llm_final'))
    tokenizer.save_pretrained(str(config.output_dir / 'llm_final'))
    
    logger.info("LLM fine-tuning complete")
    return model, tokenizer


def train_diffusion(config):
    """Train diffusion model with adversarial training."""
    logger.info("=" * 80)
    logger.info("STAGE 3: Training Diffusion Model")
    logger.info("=" * 80)
    
    # Load datasets
    train_dataset = DiffusionDataset(config.data_dir, 'train')
    val_dataset = DiffusionDataset(config.data_dir, 'validation')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.diffusion_config['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.diffusion_config['batch_size'],
        num_workers=4
    )
    
    # Initialize model
    model = DiffusionCodeGenerator(
        vocab_size=config.diffusion_config['vocab_size'],
        d_model=config.diffusion_config['d_model'],
        num_timesteps=config.diffusion_config['num_timesteps']
    ).to(config.device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.diffusion_config['lr'],
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.diffusion_config['epochs']
    )
    
    # Import adversarial trainer
    from diffusion_service.train import AdversarialDiffusionTrainer
    adv_trainer = AdversarialDiffusionTrainer(
        model,
        device=config.device,
        epsilon=config.diffusion_config['epsilon'],
        num_adv_steps=config.diffusion_config['num_adv_steps']
    )
    
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(config.diffusion_config['epochs']):
        # Train with adversarial examples
        clean_loss, adv_loss = adv_trainer.train_epoch(train_loader, optimizer, epoch)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for tokens, condition, labels in val_loader:
                tokens = tokens.to(config.device)
                condition = condition.to(config.device)
                labels = labels.to(config.device)
                
                # Random timestep
                t = torch.randint(0, model.num_timesteps, (tokens.size(0),)).to(config.device)
                
                # Forward pass
                noise = torch.randn_like(tokens.float())
                noisy_tokens = tokens + noise
                pred_noise = model(noisy_tokens.long(), t, condition)
                
                loss = nn.MSELoss()(pred_noise, noise)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f"Epoch {epoch+1}: Clean Loss={clean_loss:.4f}, "
                   f"Adv Loss={adv_loss:.4f}, Val Loss={val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.output_dir / 'diffusion_best.pth')
            logger.info(f"Saved best model with Val Loss={val_loss:.4f}")
        
        # Evaluate robustness every 5 epochs
        if epoch % 5 == 0:
            robustness = adv_trainer.evaluate_robustness(val_loader)
            logger.info(f"Robustness metrics: {robustness}")
    
    logger.info("Diffusion training complete")
    return model


def train_rl(config):
    """Train RL agent for strategy selection."""
    logger.info("=" * 80)
    logger.info("STAGE 4: Training RL Agent")
    logger.info("=" * 80)
    
    # Load dataset
    train_dataset = RLDataset(config.data_dir, 'train')
    
    # Initialize PPO agent
    agent = PPOAgent(
        state_dim=config.rl_config['state_dim'],
        action_dim=config.rl_config['action_dim'],
        lr=config.rl_config['lr']
    )
    
    # Training loop
    episode_rewards = []
    
    for episode_idx in tqdm(range(len(train_dataset)), desc='Training RL'):
        episode = train_dataset[episode_idx]
        
        # Replay episode trajectory
        for step in episode['trajectory']:
            state = np.array(step['state'])
            action = step['action']
            reward = step['reward']
            next_state = np.array(step['next_state'])
            done = step['done']
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
        
        # Update agent
        agent.update()
        
        episode_rewards.append(episode['total_reward'])
        
        # Log progress
        if (episode_idx + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Episode {episode_idx+1}: Avg Reward (last 100)={avg_reward:.2f}")
    
    # Save trained agent
    torch.save(agent.policy.state_dict(), config.output_dir / 'rl_policy.pth')
    torch.save(agent.value.state_dict(), config.output_dir / 'rl_value.pth')
    
    logger.info("RL training complete")
    return agent


def validate_dataset(data_dir):
    """Validate dataset structure and format."""
    logger.info("Validating dataset...")
    
    data_dir = Path(data_dir)
    
    # Check directory structure
    required_dirs = [
        'binaries',
        'ground_truth',
        'preprocessed/gnn',
        'preprocessed/llm',
        'preprocessed/diffusion',
        'preprocessed/rl'
    ]
    
    for dir_name in required_dirs:
        dir_path = data_dir / dir_name
        if not dir_path.exists():
            logger.error(f"Missing directory: {dir_path}")
            return False
        logger.info(f"✓ Found {dir_name}")
    
    # Check metadata
    metadata_file = data_dir / 'metadata.json'
    if not metadata_file.exists():
        logger.error("Missing metadata.json")
        return False
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    logger.info(f"✓ Dataset has {metadata['num_samples']} samples")
    
    # Check splits
    splits_file = data_dir / 'splits.json'
    if not splits_file.exists():
        logger.error("Missing splits.json")
        return False
    
    with open(splits_file) as f:
        splits = json.load(f)
    logger.info(f"✓ Splits: Train={len(splits['train'])}, "
               f"Val={len(splits['validation'])}, Test={len(splits['test'])}")
    
    logger.info("Dataset validation passed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Train all DeObfusca-AI models')
    parser.add_argument('--data-dir', type=str, required=True,
                       help='Path to training data directory')
    parser.add_argument('--output-dir', type=str, default='./models',
                       help='Path to save trained models')
    parser.add_argument('--skip-gnn', action='store_true',
                       help='Skip GNN training')
    parser.add_argument('--skip-llm', action='store_true',
                       help='Skip LLM training')
    parser.add_argument('--skip-diffusion', action='store_true',
                       help='Skip diffusion training')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip RL training')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset without training')
    
    args = parser.parse_args()
    
    # Validate dataset
    if not validate_dataset(args.data_dir):
        logger.error("Dataset validation failed. Please check DATASET_SPECIFICATION.md")
        return 1
    
    if args.validate_only:
        logger.info("Validation complete. Exiting.")
        return 0
    
    # Initialize config
    config = TrainingConfig(args.data_dir, args.output_dir)
    
    start_time = time.time()
    
    # Train models in sequence
    try:
        if not args.skip_gnn:
            gnn_model = train_gnn(config)
        
        if not args.skip_llm:
            llm_model, tokenizer = train_llm(config)
        
        if not args.skip_diffusion:
            diffusion_model = train_diffusion(config)
        
        if not args.skip_rl:
            rl_agent = train_rl(config)
        
        elapsed = time.time() - start_time
        logger.info("=" * 80)
        logger.info(f"ALL TRAINING COMPLETE in {elapsed/3600:.2f} hours")
        logger.info(f"Models saved to: {config.output_dir}")
        logger.info("=" * 80)
        
        # Save training summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(args.data_dir),
            'output_dir': str(args.output_dir),
            'training_time_hours': elapsed / 3600,
            'device': str(config.device),
            'models_trained': {
                'gnn': not args.skip_gnn,
                'llm': not args.skip_llm,
                'diffusion': not args.skip_diffusion,
                'rl': not args.skip_rl
            }
        }
        
        with open(config.output_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
