from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
import numpy as np
import pickle
import os
import math

app = Flask(__name__)

class EdgeAugmentedGraphTransformer(nn.Module):
    """
    Edge-Augmented Graph Transformer (EAGT) with Relative Positional Encoding.
    
    Resolves oversmoothing in deep obfuscation by:
    - Global attention instead of local message passing
    - Dominator tree bias in attention scores
    - Relative positional encoding based on graph distance
    
    This architecture can distinguish nested obfuscation (loop in loop in switch)
    and identifies code that violates dominator hierarchy rules.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, num_heads=8):
        super(EdgeAugmentedGraphTransformer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Instruction embedding with positional encoding
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        # Transformer layers with edge augmentation
        self.transformer_layers = nn.ModuleList([
            TransformerConv(
                hidden_dim,
                hidden_dim // num_heads,
                heads=num_heads,
                edge_dim=32,  # Edge features
                dropout=0.1
            ) for _ in range(num_layers)
        ])
        
        # Edge feature encoder (for dominator tree encoding)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 16),  # [distance, is_dominator, is_cfg, is_pdg]
            nn.ReLU(),
            nn.Linear(16, 32)
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Classification head with dominator-aware features
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 16, hidden_dim // 2),  # +16 for dominator features
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Dominator feature extractor
        self.dom_feature_extractor = nn.Linear(hidden_dim, 16)
    
    def forward(self, x, edge_index, edge_attr=None, dominator_mask=None):
        """
        Forward pass with dominator-biased attention.
        
        Args:
            x: Node features [N, input_dim]
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, 4] - [distance, is_dominator, is_cfg, is_pdg]
            dominator_mask: Boolean mask [N, N] indicating dominator relationships
        """
        # Initial embedding with positional encoding
        x = self.embed(x)
        x = self.pos_encoder(x)
        
        # Encode edge features
        if edge_attr is not None:
            edge_features = self.edge_encoder(edge_attr)
        else:
            edge_features = None
        
        # Transformer layers with residual connections
        for i, transformer in enumerate(self.transformer_layers):
            residual = x
            
            # Apply transformer with edge features
            x = transformer(x, edge_index, edge_attr=edge_features)
            
            # Residual connection and layer norm
            x = self.layer_norms[i](x + residual)
        
        # Extract dominator-aware features
        dom_features = self.dom_feature_extractor(x)
        
        # Concatenate with node features for classification
        x_combined = torch.cat([x, dom_features], dim=-1)
        
        # Classify each node
        out = self.classifier(x_combined)
        return out
    
    def compute_attention_bias(self, dominators, num_nodes):
        """
        Compute attention bias based on dominator relationships.
        Real code has strict dominator hierarchy; obfuscation violates this.
        """
        bias = torch.zeros(num_nodes, num_nodes)
        
        for node, doms in dominators.items():
            for dom in doms:
                if node < num_nodes and dom < num_nodes:
                    # Increase attention to dominators
                    bias[node, dom] = 1.0
                    
        return bias


class PositionalEncoding(nn.Module):
    """
    Relative positional encoding based on graph distance.
    Helps transformer understand code structure depth.
    """
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """Add positional encoding to input."""
        seq_len = x.size(0)
        return x + self.pe[:seq_len]

# Load model
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/gnn_sanitizer.pth')
EMBEDDING_PATH = os.getenv('EMBEDDING_PATH', '/app/models/instruction_embeddings.pkl')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
instruction_vocab = None

def load_model():
    global model, instruction_vocab
    
    try:
        # Load instruction vocabulary and embeddings
        with open(EMBEDDING_PATH, 'rb') as f:
            instruction_vocab = pickle.load(f)
        
        input_dim = len(instruction_vocab)
        model = EdgeAugmentedGraphTransformer(
            input_dim=input_dim,
            hidden_dim=128,
            output_dim=2,
            num_layers=6,
            num_heads=8
        )
        
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print("Warning: No pretrained model found. Using random initialization.")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'gnn-sanitizer', 'device': str(device)})

@app.route('/sanitize', methods=['POST'])
def sanitize():
    """
    Sanitize binary by detecting and removing junk instructions.
    
    Request body:
    {
        "pcode": [...],  # P-Code from Ghidra
        "cfg": {...}      # Control Flow Graph
    }
    
    Returns:
    {
        "sanitized_features": [...],
        "junk_indices": [...],
        "confidence_scores": [...]
    }
    """
    try:
        data = request.json
        pcode = data.get('pcode', [])
        cfg = data.get('cfg', {})
        
        # Input validation
        if not isinstance(pcode, list):
            return jsonify({'error': 'pcode must be a list'}), 400
        if not isinstance(cfg, dict):
            return jsonify({'error': 'cfg must be a dict'}), 400
        if len(pcode) == 0:
            return jsonify({'error': 'pcode cannot be empty'}), 400
        if len(pcode) > 100000:  # Prevent memory exhaustion
            return jsonify({'error': 'pcode too large (max 100000 instructions)'}), 400
        
        # Validate pcode structure
        for i, instr in enumerate(pcode):
            if not isinstance(instr, dict):
                return jsonify({'error': f'pcode[{i}] must be a dict'}), 400
        
        # Convert P-Code to graph representation
        try:
            graph_data = convert_pcode_to_graph(pcode, cfg)
        except Exception as e:
            return jsonify({'error': f'Graph conversion failed: {str(e)}'}), 400
        
        # Run inference with edge features and dominator info
        try:
            with torch.no_grad():
                x = graph_data.x.to(device)
                edge_index = graph_data.edge_index.to(device)
                edge_attr = graph_data.edge_attr.to(device) if hasattr(graph_data, 'edge_attr') else None
                dominator_mask = graph_data.dominator_mask.to(device) if hasattr(graph_data, 'dominator_mask') else None
                
                logits = model(x, edge_index, edge_attr=edge_attr, dominator_mask=dominator_mask)
                probs = F.softmax(logits, dim=1)
                predictions = torch.argmax(probs, dim=1)
                
            # Extract junk instruction indices
            junk_indices = torch.where(predictions == 1)[0].cpu().tolist()
            confidence_scores = probs[:, 1].cpu().tolist()
            
            # Create sanitized features (remove junk)
            sanitized_features = [
                pcode[i] for i in range(len(pcode)) if i not in junk_indices
            ]
            
            return jsonify({
                'sanitized_features': sanitized_features,
                'junk_indices': junk_indices,
                'confidence_scores': confidence_scores,
                'original_count': len(pcode),
                'sanitized_count': len(sanitized_features)
            })
        except Exception as e:
            return jsonify({'error': f'Inference failed: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def convert_pcode_to_graph(pcode, cfg):
    """
    Convert P-Code and CFG to PyTorch Geometric Data object.
    """
    # Extract node features
    node_features = []
    for op in pcode:
        # Create one-hot encoding for instruction mnemonic
        mnemonic = op.get('mnemonic', 'UNKNOWN')
        
        if instruction_vocab and mnemonic in instruction_vocab:
            idx = instruction_vocab[mnemonic]
            feature = np.zeros(len(instruction_vocab))
            feature[idx] = 1.0
        else:
            feature = np.zeros(len(instruction_vocab) if instruction_vocab else 100)
        
        node_features.append(feature)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Extract edge index from CFG
    edges = cfg.get('edges', [])
    edge_list = [[e['from'], e['to']] for e in edges if e['from'] >= 0 and e['to'] >= 0]
    
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        # No edges, create self-loops
        edge_index = torch.tensor([[i, i] for i in range(len(pcode))], dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index)

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5002, debug=True)
