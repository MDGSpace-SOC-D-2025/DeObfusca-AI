import subprocess
import sys
import os
import glob
import re
import math
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import softmax
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Auto-install PyG if missing (Kaggle env)
try:
    import torch_geometric
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric", "-f", "https://data.pyg.org/whl/torch-2.0.0+cu118.html"])

# Configuration
CONFIG = {
    'vocab_size': 500,  
    'embed_dim': 256,
    'num_layers': 6,
    'num_heads': 8,
    'edge_dim': 32,
    'dropout': 0.1,
    'lr': 3e-4,
    'batch_size': 8,
    'epochs': 20,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'data_path': '/kaggle/input', 
    'save_dir': './checkpoints',
    'max_nodes': 2000 
}

os.makedirs(CONFIG['save_dir'], exist_ok=True)

# -----------------------------------------------------------------------------
# Vocabulary & Parsing
# -----------------------------------------------------------------------------

class InstructionVocabulary:
    # Comprehensive mnemonic list for x86/ARM/PCode
    MNEMONICS = [
        '<PAD>', '<UNK>', '<START>', '<END>',
        'MOV', 'PUSH', 'POP', 'LEA', 'NOP', 'XCHG', 'IN', 'OUT',
        'ADD', 'SUB', 'MUL', 'DIV', 'INC', 'DEC', 'NEG', 'CMP', 'AND', 'OR', 'XOR', 'NOT', 'TEST',
        'SHL', 'SHR', 'SAR', 'ROL', 'ROR',
        'JMP', 'JE', 'JNE', 'JG', 'JGE', 'JL', 'JLE', 'JA', 'JB', 'JAE', 'JBE', 'CALL', 'RET',
        'INT', 'SYSCALL', 'LEAVE', 'ENTER',
        'CMOV', 'CMOVE', 'CMOVNE', 'SET', 'SETE', 'SETNE',
        'LDR', 'STR', 'BL', 'BX', 'SVC', # ARM
        'COPY', 'LOAD', 'STORE', 'BRANCH', 'CBRANCH' # PCode
    ]

    def __init__(self):
        self.m2i = {m: i for i, m in enumerate(self.MNEMONICS)}
        self.unk_idx = self.m2i['<UNK>']

    def encode(self, mnemonic):
        return self.m2i.get(mnemonic.upper().strip(), self.unk_idx)

class AssemblyParser:
    def __init__(self, vocab):
        self.vocab = vocab
        # Regex to grab mnemonic (usually 2nd token in standard objdump/IDA output)
        self.token_pattern = re.compile(r'[a-zA-Z0-9_]+')

    def parse(self, filepath, max_nodes):
        nodes, edge_src, edge_dst, labels = [], [], [], []
        
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                lines = f.readlines()
        except: return None

        # Filter to code section
        code_lines = [l for l in lines if '.text' in l or '.code' in l][:max_nodes]
        if not code_lines: return None

        for idx, line in enumerate(code_lines):
            parts = self.token_pattern.findall(line)
            if len(parts) < 2: continue

            # Heuristic: Mnemonic is often 2nd token (after address)
            # Adjust index based on specific ASM format (IDA vs Objdump)
            mnemonic = parts[1] if len(parts) > 1 else parts[0]
            
            vid = self.vocab.encode(mnemonic)
            nodes.append(vid)
            
            # Auto-labeling logic for training (Proxy task: NOP/INT/Zeroing)
            # In production, load external labels here
            is_junk = 1.0 if mnemonic.upper() in ['NOP', 'FNOP', 'INT3', 'INT'] else 0.0
            labels.append(is_junk)

            # Sequential flow edges
            if idx > 0:
                edge_src.append(idx - 1)
                edge_dst.append(idx)

        if not nodes: return None

        return Data(
            x=torch.tensor(nodes, dtype=torch.long),
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_type=torch.zeros(len(edge_src), dtype=torch.long), # Placeholder for CFG types
            positions=torch.arange(len(nodes)),
            y=torch.tensor(labels, dtype=torch.float)
        )

# -----------------------------------------------------------------------------
# Model Architecture
# -----------------------------------------------------------------------------

class EdgeAugmentedAttention(nn.Module):
    def __init__(self, dim, heads=8, edge_dim=32, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        
        q = self.q_proj(x).view(-1, self.heads, self.head_dim)
        k = self.k_proj(x).view(-1, self.heads, self.head_dim)
        v = self.v_proj(x).view(-1, self.heads, self.head_dim)
        
        # Calculate scores
        q_i = q[row]
        k_j = k[col]
        edge_bias = self.edge_proj(edge_attr).unsqueeze(-1)
        
        scores = (q_i * k_j).sum(dim=-1, keepdim=True) * self.scale
        scores = scores + edge_bias
        
        attn = softmax(scores, row, num_nodes=x.size(0))
        attn = self.dropout(attn)
        
        # Aggregation
        v_j = v[col]
        out = torch.zeros_like(v)
        # Scatter add implementation
        import torch_scatter
        out = torch_scatter.scatter(v_j * attn, row, dim=0, reduce='add', dim_size=x.size(0))
        
        out = out.view(-1, self.dim)
        return self.out_proj(out)

class GraphTransformerBlock(nn.Module):
    def __init__(self, dim, heads, edge_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EdgeAugmentedAttention(dim, heads, edge_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, edge_index, edge_attr):
        x = x + self.attn(self.norm1(x), edge_index, edge_attr)
        x = x + self.ff(self.norm2(x))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x, positions):
        # Clamp positions to avoid index error on very large graphs
        positions = positions.clamp(max=self.pe.size(0)-1)
        return x + self.pe[positions]

class JunkInstructionDetector(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, edge_dim, dropout=0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.edge_embed = nn.Embedding(4, edge_dim) # 4 edge types assumption
        self.pos_enc = PositionalEncoding(embed_dim)
        
        self.layers = nn.ModuleList([
            GraphTransformerBlock(embed_dim, num_heads, edge_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, x, edge_index, edge_type, positions, batch=None):
        h = self.embed(x)
        h = self.pos_enc(h, positions)
        edge_attr = self.edge_embed(edge_type)
        
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            
        return self.classifier(h).squeeze(-1)

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class MalwareDataset(Dataset):
    def __init__(self, root_dir, vocab, max_nodes=2000, limit=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, '**', '*.asm'), recursive=True))
        if limit: self.files = self.files[:limit]
        self.vocab = vocab
        self.parser = AssemblyParser(vocab)
        self.max_nodes = max_nodes
        print(f"Dataset loaded: {len(self.files)} files")

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        # On-the-fly parsing to save memory
        data = self.parser.parse(self.files[idx], self.max_nodes)
        if data is None:
            # Fallback for parsing errors
            return Data(
                x=torch.tensor([0, 1], dtype=torch.long),
                edge_index=torch.tensor([[0],[1]], dtype=torch.long),
                edge_type=torch.tensor([0], dtype=torch.long),
                positions=torch.arange(2),
                y=torch.tensor([0., 0.], dtype=torch.float)
            )
        return data

# -----------------------------------------------------------------------------
# Training Engine
# -----------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, crit, device):
    model.train()
    total_loss = 0
    preds, targets = [], []

    for batch in tqdm(loader, desc="Train"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        logits = model(batch.x, batch.edge_index, batch.edge_type, batch.positions)
        loss = crit(logits, batch.y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds.extend((probs > 0.5).long().cpu().numpy())
        targets.extend(batch.y.cpu().numpy())

    return total_loss / len(loader), f1_score(targets, preds, zero_division=0)

def validate(model, loader, crit, device):
    model.eval()
    total_loss = 0
    preds, targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.edge_type, batch.positions)
            loss = crit(logits, batch.y)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds.extend((probs > 0.5).long().cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
            
    return total_loss / len(loader), f1_score(targets, preds, zero_division=0)

def main():
    print(f"Initializing Junk Code Detector on {CONFIG['device']}")
    
    vocab = InstructionVocabulary()
    ds = MalwareDataset(CONFIG['data_path'], vocab, max_nodes=CONFIG['max_nodes'])
    
    if len(ds) == 0:
        print("Error: No .asm files found. Please attach a dataset.")
        return

    # Split
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2)
    
    # Model
    model = JunkInstructionDetector(
        vocab_size=len(vocab.MNEMONICS),
        embed_dim=CONFIG['embed_dim'],
        num_layers=CONFIG['num_layers'],
        num_heads=CONFIG['num_heads'],
        edge_dim=CONFIG['edge_dim']
    ).to(CONFIG['device'])
    
    # Class imbalance handling
    pos_weight = torch.tensor([5.0]).to(CONFIG['device'])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    
    best_f1 = 0
    history = {'t_loss': [], 'v_loss': [], 'f1': []}

    print("Starting training loop...")
    for epoch in range(CONFIG['epochs']):
        t_loss, t_f1 = train_epoch(model, train_loader, optimizer, criterion, CONFIG['device'])
        v_loss, v_f1 = validate(model, val_loader, criterion, CONFIG['device'])
        
        history['t_loss'].append(t_loss)
        history['v_loss'].append(v_loss)
        history['f1'].append(v_f1)
        
        print(f"Ep {epoch+1} | T_Loss: {t_loss:.4f} | V_Loss: {v_loss:.4f} | V_F1: {v_f1:.4f}")
        
        if v_f1 > best_f1:
            best_f1 = v_f1
            torch.save(model.state_dict(), f"{CONFIG['save_dir']}/best_model.pth")
            
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['t_loss'], label='Train')
    plt.plot(history['v_loss'], label='Val')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['f1'], label='F1 Score', color='green')
    plt.title('Validation F1')
    plt.savefig('training_stats.png')
    
    print(f"Done. Best F1: {best_f1:.4f}")

if __name__ == '__main__':
    main()