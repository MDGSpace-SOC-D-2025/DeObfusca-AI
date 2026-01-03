import os
import sys
import glob
import re
import math
import random
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


# 1. Configuration 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = '/kaggle/working/data/train'
CHECKPOINT_DIR = './checkpoints'

# Training parameters
HPARAMS = {
    'embed_dim': 256,
    'op_embed_dim': 32,
    'layers': 6,
    'heads': 8,
    'lr': 5e-5,
    'batch_size': 8,
    'epochs': 20,
    'obfuscate_p': 0.3, # 30% synthetic injection rate
    'gamma': 2.0,       # Focal Loss focusing param
    'alpha': 0.75,      # Focal Loss balance param
    'target_files': 1200 # Extract until we have this many files
}

# Dependency Check
def install_deps():
    pkgs = ['torch-geometric', 'py7zr']
    for p in pkgs:
        try:
            __import__(p.replace('-', '_'))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", p])

install_deps()
import py7zr 


# 2. Vocabulary & Model Architecture

class Vocab:
    """Maps x86 assembly mnemonics to integer IDs."""
    TOKENS = [
        '<PAD>', '<UNK>', '<S>', '</S>',
        'MOV', 'PUSH', 'POP', 'LEA', 'NOP', 'XCHG', 'IN', 'OUT',
        'ADD', 'SUB', 'MUL', 'DIV', 'INC', 'DEC', 'NEG', 'CMP', 'AND', 'OR', 'XOR', 'NOT', 'TEST',
        'SHL', 'SHR', 'SAR', 'ROL', 'ROR',
        'JMP', 'JE', 'JNE', 'JG', 'JGE', 'JL', 'JLE', 'JA', 'JB', 'JAE', 'JBE', 'CALL', 'RET',
        'INT', 'SYSCALL', 'LEAVE', 'ENTER', 'CMOV', 'CMOVE', 'CMOVNE', 'SET', 'SETE', 'SETNE',
        'LDR', 'STR', 'BL', 'BX', 'SVC', 'COPY', 'LOAD', 'STORE', 'BRANCH'
    ]
    
    def __init__(self):
        self.map = {t: i for i, t in enumerate(self.TOKENS)}
        self.unk = self.map['<UNK>']
        
    def get(self, token):
        return self.map.get(token.upper().strip(), self.unk)

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy negatives and focuses on hard positives (Junk).
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class EdgeAugmentedAttention(nn.Module):
    """
    Multi-head attention mechanism that injects edge attributes (Control Flow Type)
    into the attention scores as a bias term.
    """
    def __init__(self, dim, heads, edge_dim, drop=0.1):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index, edge_attr):
        r, c = edge_index
        
        q = self.q(x).view(-1, self.heads, self.head_dim)
        k = self.k(x).view(-1, self.heads, self.head_dim)
        v = self.v(x).view(-1, self.heads, self.head_dim)
        
        # Edge bias
        e_bias = self.edge_proj(edge_attr).unsqueeze(-1)
        score = (q[r] * k[c]).sum(dim=-1, keepdim=True) * self.scale
        score = score + e_bias
        
        # Softmax & Dropout
        attn = pyg_softmax(score, r, num_nodes=x.size(0))
        attn = self.drop(attn)
        
        # Aggregate
        out = torch.zeros_like(v)
        out.index_add_(0, r, v[c] * attn)
        
        return self.out(out.view(-1, self.dim))

class GraphTransformerBlock(nn.Module):
    def __init__(self, dim, heads, edge_dim=32, drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EdgeAugmentedAttention(dim, heads, edge_dim, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim*4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(dim*4, dim), nn.Dropout(drop)
        )

    def forward(self, x, edge_index, edge_attr):
        x = x + self.attn(self.norm1(x), edge_index, edge_attr)
        x = x + self.ff(self.norm2(x))
        return x

class GNN_Deobfuscator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, op_dim=32, layers=6, heads=8):
        super().__init__()
        
        # Embeddings
        self.emb_mnem = nn.Embedding(vocab_size, embed_dim)
        self.emb_op = nn.Embedding(5, op_dim) 
        self.emb_edge = nn.Embedding(4, 32)
        
        # Fusion
        self.fusion = nn.Linear(embed_dim + 2*op_dim, embed_dim)
        
        # Positional Encoding
        self.register_buffer('pe', self._gen_pe(embed_dim))
        
        # Encoder Stack
        self.layers = nn.ModuleList([
            GraphTransformerBlock(embed_dim, heads) for _ in range(layers)
        ])
        
        # Head
        self.head = nn.Linear(embed_dim, 1)

    def _gen_pe(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0)/dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, batch):
        mnem_emb = self.emb_mnem(batch.x_mnem)
        op1_emb = self.emb_op(batch.x_op1)
        op2_emb = self.emb_op(batch.x_op2)
        x = torch.cat([mnem_emb, op1_emb, op2_emb], dim=-1)
        x = self.fusion(x)
        
        pos = batch.pos.clamp(max=self.pe.size(0)-1)
        x = x + self.pe[pos]
        
        e_attr = self.emb_edge(batch.edge_attr)
        for layer in self.layers:
            x = layer(x, batch.edge_index, e_attr)
            
        return self.head(x).squeeze(-1)


# 3. Data Extraction Logic

def prepare_dataset(target_count):

    if not os.path.exists(DATA_ROOT):
        os.makedirs(DATA_ROOT)

    existing_files = glob.glob(os.path.join(DATA_ROOT, '**', '*.asm'), recursive=True)
    current_count = len(existing_files)
    
    print(f"[Data Check] Found {current_count} existing files.")
    
    if current_count >= target_count:
        print(" -> Target count met. Skipping extraction.")
        return

    needed = target_count - current_count
    print(f" -> Need {needed} more files. Locating archive...")

    archive_paths = glob.glob('/kaggle/input/**/train.7z', recursive=True)
    if not archive_paths:
        print("Error: Could not find 'train.7z' in /kaggle/input.")
        return
    
    archive_path = archive_paths[0]
    
    # Filter duplicates
    existing_basenames = {os.path.basename(f) for f in existing_files}
    
    try:
        with py7zr.SevenZipFile(archive_path, mode='r') as z:
            all_files = z.getnames()
            candidates = [f for f in all_files if f.endswith('.asm')]
            
            to_extract = []
            for f in candidates:
                if os.path.basename(f) not in existing_basenames:
                    to_extract.append(f)
                    if len(to_extract) >= needed:
                        break
            
            if to_extract:
                print(f" -> Extracting {len(to_extract)} files...")
                z.extract(path=DATA_ROOT, targets=to_extract)
                print(" -> Extraction complete.")
    except Exception as e:
        print(f"Extraction Error: {e}")


# 4. Data Generators

class OLLVMInjector:
    """
    Simulates Bogus Control Flow (Diamond Obfuscation).
    HARDENED: Now generates fake operand types to prevent metadata leaks.
    """
    def __init__(self, vocab):
        self.vocab = vocab
        self.branches = ['JZ', 'JNZ', 'JG', 'JL']

    def gen_diamond_graph(self, context_buffer):
        nodes, labels, edges = [], [], []
        
        
        def rand_op():
            r = random.random()
            if r < 0.6: return 1 
            if r < 0.9: return 3 
            return 2 

        
        pred = random.choice(context_buffer) if context_buffer else 'CMP'
        nodes.append((self.vocab.get(pred), 1, 4)) 
        labels.append(1.0) 

    
        nodes.append((self.vocab.get(random.choice(self.branches)), 3, 0))
        labels.append(1.0) 

       
        block_len = random.randint(2, 4)
        for _ in range(block_len):
            op_mnem = random.choice(context_buffer) if context_buffer and random.random() > 0.2 else 'NOP'
            
            
            if op_mnem == 'NOP':
                op1, op2 = 0, 0
            else:
                op1, op2 = rand_op(), rand_op()
                
            nodes.append((self.vocab.get(op_mnem), op1, op2))
            labels.append(1.0)

        # Wiring (Diamond Structure)
        edges.append((0, 1))      # Pred -> Jump
        edges.append((1, -1))     # Jump -> Real Target
        edges.append((1, 2))      # Jump -> Fake Block
        
        for k in range(2, 2 + block_len - 1):
            edges.append((k, k+1))
            
        edges.append((2 + block_len - 1, -1)) # Converge
        
        return nodes, labels, edges

class GraphParser:
    def __init__(self):
        self.vocab = Vocab()
        self.injector = OLLVMInjector(self.vocab)
        self.regex = re.compile(r'[a-zA-Z0-9_\[\]]+')

    def _get_op_type(self, token):
        # 0:None, 1:Reg, 2:Mem, 3:Imm, 4:Zero
        if not token: return 0
        if '[' in token: return 2
        if token.isdigit() or token.startswith('0x'):
            try:
                val = int(token, 0)
                return 4 if val == 0 else 3
            except: pass
        return 1

    def parse(self, path, max_nodes=2500):
        try:
            with open(path, 'r', encoding='latin-1') as f:
                lines = [l for l in f if '.text' in l or '.code' in l]
        except: return None
        
        lines = lines[:int(max_nodes * 0.7)]
        if not lines: return None

        node_feats, edge_src, edge_dst, labels = [], [], [], []
        last_real_idx = -1
        curr_idx = 0
        context = [] 

        for line in lines:
            tokens = self.regex.findall(line)
            if len(tokens) < 2: continue
            
            mnem_str = tokens[1] if len(tokens) > 1 else tokens[0]
            op1_str = tokens[2] if len(tokens) > 2 else None
            op2_str = tokens[3] if len(tokens) > 3 else None
            
            # --- Injection Logic ---
            if last_real_idx != -1 and random.random() < HPARAMS['obfuscate_p']:
                d_nodes, d_labels, d_edges = self.injector.gen_diamond_graph(context)
                base = curr_idx
                for n, l in zip(d_nodes, d_labels):
                    node_feats.append(n) 
                    labels.append(l)
                    curr_idx += 1
                
                edge_src.append(last_real_idx)
                edge_dst.append(base)
                for u, v in d_edges:
                    s = base + u
                    d = curr_idx if v == -1 else base + v
                    edge_src.append(s)
                    edge_dst.append(d)

            # --- Real Instruction ---
            op1_t = self._get_op_type(op1_str)
            op2_t = self._get_op_type(op2_str)
            node_feats.append((self.vocab.get(mnem_str), op1_t, op2_t))
            labels.append(0.0)
            
            context.append(mnem_str)
            if len(context) > 15: context.pop(0)

            if last_real_idx != -1 and len(node_feats) == (curr_idx + 1):
                edge_src.append(last_real_idx)
                edge_dst.append(curr_idx)
            
            last_real_idx = curr_idx
            curr_idx += 1

        if not node_feats: return None
        feats = torch.tensor(node_feats, dtype=torch.long)
        
        return Data(
            x_mnem=feats[:, 0],
            x_op1=feats[:, 1],
            x_op2=feats[:, 2],
            edge_index=torch.tensor([edge_src, edge_dst], dtype=torch.long),
            edge_attr=torch.zeros(len(edge_src), dtype=torch.long),
            pos=torch.arange(len(node_feats)),
            y=torch.tensor(labels, dtype=torch.float)
        )


# 5. Training Loop

class DatasetWrapper(Dataset):
    def __init__(self, root):
        self.files = sorted(glob.glob(os.path.join(root, '**', '*.asm'), recursive=True))
        if not self.files:
            self.files = sorted(glob.glob('/kaggle/working/**/*.asm', recursive=True))
        self.parser = GraphParser()
        print(f"Dataset Loaded: {len(self.files)} files")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        d = self.parser.parse(self.files[i])
        if d is None:
            return Data(x_mnem=torch.tensor([0]), x_op1=torch.tensor([0]), x_op2=torch.tensor([0]),
                       edge_index=torch.tensor([[0],[0]]), edge_attr=torch.tensor([0]),
                       pos=torch.tensor([0]), y=torch.tensor([0.]))
        return d

def run_training():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 1. Extraction
    prepare_dataset(HPARAMS['target_files'])
    
    ds = DatasetWrapper(DATA_ROOT)
    if len(ds) == 0: return

    # 2. Split
    train_len = int(len(ds) * 0.8)
    train_ds, val_ds = random_split(ds, [train_len, len(ds) - train_len])
    
    t_loader = DataLoader(train_ds, batch_size=HPARAMS['batch_size'], shuffle=True, num_workers=0)
    v_loader = DataLoader(val_ds, batch_size=HPARAMS['batch_size'], shuffle=False)
    
    # 3. Model
    model = GNN_Deobfuscator(vocab_size=len(Vocab.TOKENS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=HPARAMS['lr'], weight_decay=1e-2)
    crit = FocalLoss(alpha=HPARAMS['alpha'], gamma=HPARAMS['gamma'])
    
    history = {'t_loss': [], 'v_loss': [], 'f1': []}
    best_f1 = 0.0
    
    print(f"Starting training on {DEVICE}...")
    
    for ep in range(HPARAMS['epochs']):
        # Train
        model.train()
        losses = []
        for batch in tqdm(t_loader, desc=f"Ep {ep+1} Train", leave=False):
            batch = batch.to(DEVICE)
            opt.zero_grad()
            logits = model(batch)
            loss = crit(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        
        t_loss = np.mean(losses)
        
        # Validation
        model.eval()
        v_losses = []
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in v_loader:
                batch = batch.to(DEVICE)
                logits = model(batch)
                loss = crit(logits, batch.y)
                v_losses.append(loss.item())
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch.y.cpu().numpy())
                
        v_loss = np.mean(v_losses)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        history['t_loss'].append(t_loss)
        history['v_loss'].append(v_loss)
        history['f1'].append(f1)
        
        print(f"Ep {ep+1}: T_Loss {t_loss:.4f} | V_Loss {v_loss:.4f} | F1 {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best.pth")
            
    # 4. Visualization
    print("\nGenerating training graphs...")
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['t_loss'], label='Train Loss')
    plt.plot(history['v_loss'], label='Val Loss')
    plt.title('Loss Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['f1'], label='Val F1', color='green')
    plt.title('F1 Score Performance')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('training_metrics.png')
    print("Graphs saved to 'training_metrics.png'")

if __name__ == "__main__":
    run_training()