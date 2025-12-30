import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.utils import softmax as pyg_softmax

# ==============================================================================
# Vocabulary & Constants
# ==============================================================================
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

    def __len__(self):
        return len(self.TOKENS)

# ==============================================================================
# Custom Objectives
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Down-weights easy negatives (Real Code) and focuses on hard positives (Junk).
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

# ==============================================================================
# Graph Transformer Layers
# ==============================================================================
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
        
        # Projections
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.out = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x, edge_index, edge_attr):
        r, c = edge_index
        
        # Project Q, K, V
        q = self.q(x).view(-1, self.heads, self.head_dim)
        k = self.k(x).view(-1, self.heads, self.head_dim)
        v = self.v(x).view(-1, self.heads, self.head_dim)
        
        # Add edge bias to attention scores
        e_bias = self.edge_proj(edge_attr).unsqueeze(-1)
        score = (q[r] * k[c]).sum(dim=-1, keepdim=True) * self.scale
        score = score + e_bias
        
        # Softmax & Dropout
        attn = pyg_softmax(score, r, num_nodes=x.size(0))
        attn = self.drop(attn)
        
        # Aggregate (Pure PyTorch scatter add)
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
        # Pre-Norm configuration
        x = x + self.attn(self.norm1(x), edge_index, edge_attr)
        x = x + self.ff(self.norm2(x))
        return x

# ==============================================================================
# Main Architecture
# ==============================================================================
class GNN_Deobfuscator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, op_dim=32, layers=6, heads=8):
        super().__init__()
        
        # Embeddings: Mnemonic + Operand Types + Edge Types
        self.emb_mnem = nn.Embedding(vocab_size, embed_dim)
        self.emb_op = nn.Embedding(5, op_dim) 
        self.emb_edge = nn.Embedding(4, 32)
        
        # Fusion: Combines instruction components into one vector
        self.fusion = nn.Linear(embed_dim + 2*op_dim, embed_dim)
        
        # Positional Encodings
        self.register_buffer('pe', self._gen_pe(embed_dim))
        
        # Encoder Stack
        self.layers = nn.ModuleList([
            GraphTransformerBlock(embed_dim, heads) for _ in range(layers)
        ])
        
        # Classifier Head
        self.head = nn.Linear(embed_dim, 1)

    def _gen_pe(self, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0)/dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, batch):
        # 1. Fuse Features
        mnem_emb = self.emb_mnem(batch.x_mnem)
        op1_emb = self.emb_op(batch.x_op1)
        op2_emb = self.emb_op(batch.x_op2)
        x = torch.cat([mnem_emb, op1_emb, op2_emb], dim=-1)
        x = self.fusion(x)
        
        # 2. Add Positional Encodings
        pos = batch.pos.clamp(max=self.pe.size(0)-1)
        x = x + self.pe[pos]
        
        # 3. Process Graph
        e_attr = self.emb_edge(batch.edge_attr)
        for layer in self.layers:
            x = layer(x, batch.edge_index, e_attr)
            
        return self.head(x).squeeze(-1)