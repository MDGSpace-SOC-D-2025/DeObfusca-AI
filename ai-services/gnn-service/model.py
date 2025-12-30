

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops, softmax
import math
from typing import Optional, Tuple, List, Dict
import numpy as np


class EdgeAugmentedAttention(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        edge_dim: int = 32,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.head_dim = out_channels // heads
        assert out_channels % heads == 0, "out_channels must be divisible by heads"
        self.q_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.k_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.v_proj = nn.Linear(in_channels, out_channels, bias=bias)
        self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        self.out_proj = nn.Linear(out_channels, out_channels, bias=bias)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:

        # Add self-loops
        num_nodes = x.size(0)
        edge_index, edge_attr = self._add_self_loops(edge_index, edge_attr, num_nodes)
        
        # Project to Q, K, V
        query = self.q_proj(x).view(-1, self.heads, self.head_dim)
        key = self.k_proj(x).view(-1, self.heads, self.head_dim)
        value = self.v_proj(x).view(-1, self.heads, self.head_dim)
        
        # Project edge features to attention bias
        edge_bias = self.edge_proj(edge_attr)  # [num_edges, heads]
        
        # Message passing
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_bias=edge_bias
        )
        
        # Reshape and project
        out = out.view(-1, self.out_channels)
        out = self.out_proj(out)
        
        return out
    
    def _add_self_loops(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Create self-loop edges
        self_loop_index = torch.arange(num_nodes, device=edge_index.device)
        self_loop_index = torch.stack([self_loop_index, self_loop_index])
        
        # Concatenate
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        
        # Zero edge features for self-loops
        self_loop_attr = torch.zeros(num_nodes, self.edge_dim, device=edge_attr.device)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        return edge_index, edge_attr
    
    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        edge_bias: torch.Tensor,
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:

        # Compute attention scores
        attn = (query_i * key_j).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = attn + edge_bias  # Add edge bias
        
        # Softmax over neighbors
        attn = softmax(attn, index, ptr, size_i)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        # Weight values by attention
        out = value_j * attn.unsqueeze(-1)
        
        return out


class FeedForward(nn.Module):

    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphTransformerBlock(nn.Module):

    
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        edge_dim: int = 32,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EdgeAugmentedAttention(
            in_channels=dim,
            out_channels=dim,
            heads=heads,
            edge_dim=edge_dim,
            dropout=dropout
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, dim * ff_mult, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Pre-norm attention with residual
        x = x + self.attn(self.norm1(x), edge_index, edge_attr)
        
        # Pre-norm feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class PositionalEncoding(nn.Module):

    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return x + self.pe[positions]


class JunkInstructionDetector(nn.Module):

    
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        edge_dim: int = 32,
        num_edge_types: int = 8,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Instruction embedding
        self.instruction_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Edge type embedding
        self.edge_embed = nn.Embedding(num_edge_types, edge_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Input projection (for additional features)
        self.input_proj = nn.Linear(embed_dim + 64, embed_dim)  # 64 for additional features
        
        # Graph Transformer blocks
        self.layers = nn.ModuleList([
            GraphTransformerBlock(
                dim=embed_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(embed_dim)
        
        # Node-level classification head (per-instruction)
        self.node_classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Graph-level classification head (overall obfuscation score)
        self.graph_classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # Concat mean and max pool
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        positions: torch.Tensor,
        additional_features: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:

        # Embed instructions
        h = self.instruction_embed(x)  # [num_nodes, embed_dim]
        
        # Add positional encoding
        h = self.pos_encoding(h, positions)
        
        # Concatenate additional features if provided
        if additional_features is not None:
            h = torch.cat([h, additional_features], dim=-1)
            h = self.input_proj(h)
        
        # Embed edge types
        edge_attr = self.edge_embed(edge_type)  # [num_edges, edge_dim]
        
        # Apply Graph Transformer layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        
        # Final normalization
        h = self.final_norm(h)
        
        # Node-level classification
        node_logits = self.node_classifier(h)
        node_probs = F.softmax(node_logits, dim=-1)[:, 1]  # Probability of being junk
        
        # Graph-level classification (if batched)
        if batch is not None:
            # Global pooling
            mean_pool = global_mean_pool(h, batch)
            max_pool = global_max_pool(h, batch)
            graph_repr = torch.cat([mean_pool, max_pool], dim=-1)
            graph_score = self.graph_classifier(graph_repr).squeeze(-1)
        else:
            # Single graph
            mean_pool = h.mean(dim=0, keepdim=True)
            max_pool = h.max(dim=0, keepdim=True)[0]
            graph_repr = torch.cat([mean_pool, max_pool], dim=-1)
            graph_score = self.graph_classifier(graph_repr).squeeze(-1)
        
        return {
            'node_logits': node_logits,
            'node_probs': node_probs,
            'graph_score': graph_score,
            'embeddings': h
        }
    
    def predict(self, data: Data, threshold: float = 0.5) -> Dict[str, torch.Tensor]:

        self.eval()
        with torch.no_grad():
            output = self.forward(
                x=data.x,
                edge_index=data.edge_index,
                edge_type=data.edge_type,
                positions=data.positions,
                additional_features=data.get('additional_features'),
                batch=data.get('batch')
            )
            
            junk_mask = output['node_probs'] > threshold
            
            return {
                'junk_mask': junk_mask,
                'junk_probs': output['node_probs'],
                'graph_score': output['graph_score'],
                'num_junk': junk_mask.sum().item(),
                'junk_ratio': junk_mask.float().mean().item()
            }


class InstructionVocabulary:

    
    PCODE_MNEMONICS = [
        # Special tokens
        '<PAD>', '<UNK>', '<START>', '<END>',
        
        # Data operations
        'COPY', 'LOAD', 'STORE', 'PIECE', 'SUBPIECE',
        
        # Arithmetic
        'INT_ADD', 'INT_SUB', 'INT_MULT', 'INT_DIV', 'INT_SDIV',
        'INT_REM', 'INT_SREM', 'INT_NEGATE',
        
        # Logical
        'INT_AND', 'INT_OR', 'INT_XOR', 'INT_NOT',
        'INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT',
        
        # Comparison
        'INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS', 'INT_SLESS',
        'INT_LESSEQUAL', 'INT_SLESSEQUAL', 'INT_CARRY', 'INT_SCARRY',
        'INT_SBORROW',
        
        # Boolean
        'BOOL_AND', 'BOOL_OR', 'BOOL_XOR', 'BOOL_NEGATE',
        
        # Floating point
        'FLOAT_ADD', 'FLOAT_SUB', 'FLOAT_MULT', 'FLOAT_DIV',
        'FLOAT_NEG', 'FLOAT_ABS', 'FLOAT_SQRT',
        'FLOAT_CEIL', 'FLOAT_FLOOR', 'FLOAT_ROUND',
        'FLOAT_NAN', 'FLOAT_EQUAL', 'FLOAT_NOTEQUAL',
        'FLOAT_LESS', 'FLOAT_LESSEQUAL',
        'INT2FLOAT', 'FLOAT2FLOAT', 'TRUNC', 'FLOAT2INT',
        
        # Extensions
        'INT_ZEXT', 'INT_SEXT',
        
        # Control flow
        'BRANCH', 'CBRANCH', 'BRANCHIND', 'CALL', 'CALLIND',
        'RETURN', 'CALLOTHER',
        
        # Memory
        'CAST', 'PTRADD', 'PTRSUB',
        
        # Processor specific (common)
        'CPOOLREF', 'NEW', 'MULTIEQUAL', 'INDIRECT',
        
        # x86 specific (common)
        'PUSH', 'POP', 'MOV', 'LEA', 'NOP', 'JMP', 'JE', 'JNE',
        'JL', 'JLE', 'JG', 'JGE', 'JA', 'JB', 'JAE', 'JBE',
        'CMP', 'TEST', 'XOR', 'AND', 'OR', 'NOT',
        'ADD', 'SUB', 'MUL', 'DIV', 'IMUL', 'IDIV',
        'SHL', 'SHR', 'SAR', 'ROL', 'ROR',
        'INC', 'DEC', 'NEG',
        'CALL', 'RET', 'LEAVE', 'ENTER',
        'MOVZX', 'MOVSX', 'MOVSXD',
        'CMOV', 'CMOVE', 'CMOVNE', 'CMOVL', 'CMOVLE', 'CMOVG', 'CMOVGE',
        'SET', 'SETE', 'SETNE', 'SETL', 'SETLE', 'SETG', 'SETGE',
        
        # ARM specific (common)
        'LDR', 'STR', 'LDM', 'STM', 'BL', 'BX', 'BLX',
        'MRS', 'MSR', 'SVC', 'CPSID', 'CPSIE',
        
        # SSE/AVX (common)
        'MOVAPS', 'MOVUPS', 'MOVSS', 'MOVSD',
        'ADDPS', 'ADDSS', 'SUBPS', 'SUBSS',
        'MULPS', 'MULSS', 'DIVPS', 'DIVSS',
        'XORPS', 'ANDPS', 'ORPS',
        
        # Obfuscation patterns (for detection)
        'OPAQUE_PRED', 'DEAD_STORE', 'BOGUS_BRANCH', 'JUNK_COPY',
    ]
    
    def __init__(self):
        self.mnemonic_to_idx = {m: i for i, m in enumerate(self.PCODE_MNEMONICS)}
        self.idx_to_mnemonic = {i: m for i, m in enumerate(self.PCODE_MNEMONICS)}
        self.vocab_size = len(self.PCODE_MNEMONICS)
        
        self.pad_idx = self.mnemonic_to_idx['<PAD>']
        self.unk_idx = self.mnemonic_to_idx['<UNK>']
    
    def encode(self, mnemonic: str) -> int:
        mnemonic = mnemonic.upper().strip()
        return self.mnemonic_to_idx.get(mnemonic, self.unk_idx)

    def decode(self, idx: int) -> str:
        return self.idx_to_mnemonic.get(idx, '<UNK>')

    def encode_batch(self, mnemonics: List[str]) -> torch.Tensor:
        return torch.tensor([self.encode(m) for m in mnemonics])
    
    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'mnemonic_to_idx': self.mnemonic_to_idx,
                'idx_to_mnemonic': self.idx_to_mnemonic
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'InstructionVocabulary':
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.mnemonic_to_idx = data['mnemonic_to_idx']
        vocab.idx_to_mnemonic = data['idx_to_mnemonic']
        vocab.vocab_size = len(vocab.mnemonic_to_idx)
        return vocab


def create_model(
    vocab_size: int = 256,
    embed_dim: int = 256,
    num_layers: int = 6,
    num_heads: int = 8,
    edge_dim: int = 32,
    dropout: float = 0.1,
    pretrained_path: Optional[str] = None
) -> JunkInstructionDetector:

    model = JunkInstructionDetector(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        edge_dim=edge_dim,
        dropout=dropout
    )
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model


def create_kaggle_model(preset: str = 'medium', pretrained_path: Optional[str] = None) -> JunkInstructionDetector:

    if preset == 'small':
        return create_model(vocab_size=InstructionVocabulary().vocab_size, embed_dim=128, num_layers=3, num_heads=4, edge_dim=16, dropout=0.1, pretrained_path=pretrained_path)
    elif preset == 'large':
        return create_model(vocab_size=InstructionVocabulary().vocab_size, embed_dim=512, num_layers=8, num_heads=8, edge_dim=64, dropout=0.1, pretrained_path=pretrained_path)
    else:
        return create_model(vocab_size=InstructionVocabulary().vocab_size, embed_dim=256, num_layers=6, num_heads=8, edge_dim=32, dropout=0.1, pretrained_path=pretrained_path)
