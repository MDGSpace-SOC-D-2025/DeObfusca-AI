

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


from model import JunkInstructionDetector, InstructionVocabulary, create_model

app = Flask(__name__)


@dataclass
class SanitizationResult:
    sanitized_features: List[Dict]
    removed_count: int
    total_count: int
    junk_indices: List[int]
    confidence_scores: List[float]
    obfuscation_score: float


class GNNSanitizer:
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.config = config or {
            'embed_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'edge_dim': 32,
            'dropout': 0.1,
            'threshold': 0.5
        }
        
        self.vocab = self._load_vocabulary(vocab_path)
        
        self.model = self._load_model(model_path)
        
        self.stats = {
            'total_processed': 0,
            'total_removed': 0,
            'patterns_detected': {}
        }
        
        print(f"GNN Sanitizer initialized on {self.device}")
    
    def _load_vocabulary(self, vocab_path: Optional[str]) -> InstructionVocabulary:
        if vocab_path and os.path.exists(vocab_path):
            return InstructionVocabulary.load(vocab_path)
        
        # Use default vocabulary
        return InstructionVocabulary()
    
    def _load_model(self, model_path: Optional[str]) -> JunkInstructionDetector:
        model = create_model(
            vocab_size=self.vocab.vocab_size,
            embed_dim=self.config['embed_dim'],
            num_layers=self.config['num_layers'],
            num_heads=self.config['num_heads'],
            edge_dim=self.config.get('edge_dim', 32),
            dropout=self.config['dropout']
        )
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded pretrained model from {model_path}")
        else:
            print("Warning: No pretrained model found. Using random initialization.")
            print("Run train_gnn.py to train the model first.")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def convert_pcode_to_graph(
        self,
        pcode_ops: List[Dict],
        cfg: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_nodes = len(pcode_ops)
        
        if num_nodes == 0:
            return (
                torch.zeros(0, dtype=torch.long),
                torch.zeros(2, 0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, dtype=torch.long),
                torch.zeros(0, 64)
            )
        
        mnemonics = [op.get('mnemonic', 'UNKNOWN').upper() for op in pcode_ops]
        x = torch.tensor([self.vocab.encode(m) for m in mnemonics], dtype=torch.long)
        
        edges = []
        edge_types = []
        
        for i in range(num_nodes - 1):
            edges.append((i, i + 1))
            edge_types.append(0)
        
        if cfg:
            for edge in cfg.get('edges', []):
                src = edge.get('from', -1)
                dst = edge.get('to', -1)
                flow_type = edge.get('flow_type', 'sequential')
                
                if 0 <= src < num_nodes and 0 <= dst < num_nodes:
                    if (src, dst) not in edges:
                        edges.append((src, dst))
                        if flow_type == 'branch':
                            edge_types.append(1)
                        elif flow_type == 'call':
                            edge_types.append(2)
                        else:
                            edge_types.append(0)
        
        definitions = {}
        for i, op in enumerate(pcode_ops):
            output = op.get('output')
            if output:
                var_id = output.get('offset', output.get('name', i))
                definitions[var_id] = i
            
            # Create def-use edges
            for inp in op.get('inputs', []):
                var_id = inp.get('offset', inp.get('name'))
                if var_id in definitions:
                    def_node = definitions[var_id]
                    if def_node != i and (def_node, i) not in edges:
                        edges.append((def_node, i))
                        edge_types.append(3)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_type = torch.tensor(edge_types, dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)
            edge_type = torch.zeros(0, dtype=torch.long)
        
        positions = torch.arange(num_nodes, dtype=torch.long)
        
        additional_features = self._extract_additional_features(pcode_ops, edges)
        
        return x, edge_index, edge_type, positions, additional_features
    
    def _extract_additional_features(
        self,
        pcode_ops: List[Dict],
        edges: List[Tuple[int, int]]
    ) -> torch.Tensor:
        num_nodes = len(pcode_ops)
        features = torch.zeros(num_nodes, 64)
        
        for i, op in enumerate(pcode_ops):
            # Feature 0-7: Instruction category one-hot
            mnemonic = op.get('mnemonic', '').upper()
            category = self._get_instruction_category(mnemonic)
            if category < 8:
                features[i, category] = 1.0
            
            # Feature 8: Number of inputs (normalized)
            num_inputs = len(op.get('inputs', []))
            features[i, 8] = min(num_inputs / 4.0, 1.0)
            
            # Feature 9: Has output
            features[i, 9] = 1.0 if op.get('output') else 0.0
            
            # Feature 10-11: Is first/last instruction
            features[i, 10] = 1.0 if i == 0 else 0.0
            features[i, 11] = 1.0 if i == num_nodes - 1 else 0.0
            
            # Feature 12-15: Instruction type flags
            features[i, 12] = 1.0 if 'BRANCH' in mnemonic or 'JMP' in mnemonic else 0.0
            features[i, 13] = 1.0 if 'CALL' in mnemonic else 0.0
            features[i, 14] = 1.0 if 'RETURN' in mnemonic or 'RET' in mnemonic else 0.0
            features[i, 15] = 1.0 if 'NOP' in mnemonic else 0.0
            
            # Feature 16-19: Memory operation flags
            features[i, 16] = 1.0 if 'LOAD' in mnemonic or 'LDR' in mnemonic else 0.0
            features[i, 17] = 1.0 if 'STORE' in mnemonic or 'STR' in mnemonic else 0.0
            features[i, 18] = 1.0 if 'PUSH' in mnemonic else 0.0
            features[i, 19] = 1.0 if 'POP' in mnemonic else 0.0
            
            # Feature 20-23: Arithmetic operation flags
            features[i, 20] = 1.0 if 'ADD' in mnemonic else 0.0
            features[i, 21] = 1.0 if 'SUB' in mnemonic else 0.0
            features[i, 22] = 1.0 if 'MUL' in mnemonic or 'MULT' in mnemonic else 0.0
            features[i, 23] = 1.0 if 'DIV' in mnemonic else 0.0
            
            # Feature 24-27: Logical operation flags
            features[i, 24] = 1.0 if 'AND' in mnemonic else 0.0
            features[i, 25] = 1.0 if 'OR' in mnemonic else 0.0
            features[i, 26] = 1.0 if 'XOR' in mnemonic else 0.0
            features[i, 27] = 1.0 if 'NOT' in mnemonic else 0.0
            
            # Feature 28-31: Comparison flags
            features[i, 28] = 1.0 if 'CMP' in mnemonic else 0.0
            features[i, 29] = 1.0 if 'TEST' in mnemonic else 0.0
            features[i, 30] = 1.0 if 'EQUAL' in mnemonic or 'EQ' in mnemonic else 0.0
            features[i, 31] = 1.0 if 'LESS' in mnemonic or 'LT' in mnemonic else 0.0
        
        # Graph-based features (computed from edges)
        in_degree = [0] * num_nodes
        out_degree = [0] * num_nodes
        
        for src, dst in edges:
            if src < num_nodes:
                out_degree[src] += 1
            if dst < num_nodes:
                in_degree[dst] += 1
        
        for i in range(num_nodes):
            # Feature 32-33: Degree features
            features[i, 32] = min(in_degree[i] / 4.0, 1.0)
            features[i, 33] = min(out_degree[i] / 4.0, 1.0)
            
            # Feature 34: Is potential dead end (no outgoing edges)
            features[i, 34] = 1.0 if out_degree[i] == 0 and i < num_nodes - 1 else 0.0
            
            # Feature 35: Is merge point (multiple incoming edges)
            features[i, 35] = 1.0 if in_degree[i] > 1 else 0.0
        
        return features
    
    def _get_instruction_category(self, mnemonic: str) -> int:
        if any(x in mnemonic for x in ['MOV', 'COPY', 'LOAD', 'STORE']):
            return 0  # Data movement
        elif any(x in mnemonic for x in ['ADD', 'SUB', 'MUL', 'DIV', 'NEG']):
            return 1  # Arithmetic
        elif any(x in mnemonic for x in ['AND', 'OR', 'XOR', 'NOT', 'SHL', 'SHR']):
            return 2  # Logical
        elif any(x in mnemonic for x in ['CMP', 'TEST', 'EQUAL', 'LESS']):
            return 3  # Comparison
        elif any(x in mnemonic for x in ['BRANCH', 'JMP', 'JE', 'JNE', 'JL', 'JG']):
            return 4  # Branch
        elif any(x in mnemonic for x in ['CALL', 'RETURN', 'RET']):
            return 5  # Call/Return
        elif any(x in mnemonic for x in ['PUSH', 'POP']):
            return 6  # Stack
        else:
            return 7  # Other
    
    @torch.no_grad()
    def sanitize(
        self,
        pcode_ops: List[Dict],
        cfg: Optional[Dict] = None,
        threshold: Optional[float] = None
    ) -> SanitizationResult:
        threshold = threshold or self.config.get('threshold', 0.5)
        
        if not pcode_ops:
            return SanitizationResult(
                sanitized_features=[],
                removed_count=0,
                total_count=0,
                junk_indices=[],
                confidence_scores=[],
                obfuscation_score=0.0
            )
        
        # Convert to graph
        x, edge_index, edge_type, positions, additional_features = self.convert_pcode_to_graph(
            pcode_ops, cfg
        )
        
        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        positions = positions.to(self.device)
        additional_features = additional_features.to(self.device)
        
        # Run inference
        output = self.model(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            positions=positions,
            additional_features=additional_features
        )
        
        # Get junk probabilities
        junk_probs = output['node_probs'].cpu().numpy()
        graph_score = output['graph_score'].cpu().item()
        
        # Identify junk instructions
        junk_mask = junk_probs > threshold
        junk_indices = np.where(junk_mask)[0].tolist()
        
        # Apply rule-based refinement (don't remove critical instructions)
        critical_types = {'CALL', 'RETURN', 'RET', 'CALLIND', 'BRANCHIND'}
        
        refined_junk_indices = []
        for idx in junk_indices:
            mnemonic = pcode_ops[idx].get('mnemonic', '').upper()
            if mnemonic not in critical_types:
                refined_junk_indices.append(idx)
        
        # Filter out junk
        sanitized = [
            op for i, op in enumerate(pcode_ops)
            if i not in refined_junk_indices
        ]
        
        # Update statistics
        self.stats['total_processed'] += len(pcode_ops)
        self.stats['total_removed'] += len(refined_junk_indices)
        
        return SanitizationResult(
            sanitized_features=sanitized,
            removed_count=len(refined_junk_indices),
            total_count=len(pcode_ops),
            junk_indices=refined_junk_indices,
            confidence_scores=junk_probs.tolist(),
            obfuscation_score=graph_score
        )
    
    @torch.no_grad()
    def analyze(
        self,
        pcode_ops: List[Dict],
        cfg: Optional[Dict] = None
    ) -> Dict:
        if not pcode_ops:
            return {
                'total_instructions': 0,
                'obfuscation_score': 0.0,
                'patterns': [],
                'instruction_scores': []
            }
        
        # Convert to graph
        x, edge_index, edge_type, positions, additional_features = self.convert_pcode_to_graph(
            pcode_ops, cfg
        )
        
        # Move to device
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        edge_type = edge_type.to(self.device)
        positions = positions.to(self.device)
        additional_features = additional_features.to(self.device)
        
        # Run inference
        output = self.model(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            positions=positions,
            additional_features=additional_features
        )
        
        # Extract scores
        junk_probs = output['node_probs'].cpu().numpy()
        graph_score = output['graph_score'].cpu().item()
        
        # Detect patterns
        patterns = self._detect_patterns(pcode_ops, junk_probs)
        
        # Create instruction-level analysis
        instruction_scores = []
        for i, (op, prob) in enumerate(zip(pcode_ops, junk_probs)):
            instruction_scores.append({
                'index': i,
                'mnemonic': op.get('mnemonic', 'UNKNOWN'),
                'address': op.get('address', f'0x{i:04x}'),
                'junk_probability': float(prob),
                'is_likely_junk': bool(prob > 0.5)
            })
        
        return {
            'total_instructions': len(pcode_ops),
            'obfuscation_score': float(graph_score),
            'junk_ratio': float(np.mean(junk_probs > 0.5)),
            'patterns': patterns,
            'instruction_scores': instruction_scores
        }
    
    def _detect_patterns(
        self,
        pcode_ops: List[Dict],
        junk_probs: np.ndarray
    ) -> List[Dict]:
        patterns = []
        
        # Pattern 1: NOP sequences
        nop_sequence = []
        for i, op in enumerate(pcode_ops):
            if op.get('mnemonic', '').upper() == 'NOP':
                nop_sequence.append(i)
            else:
                if len(nop_sequence) >= 3:
                    patterns.append({
                        'type': 'nop_sequence',
                        'start': nop_sequence[0],
                        'end': nop_sequence[-1],
                        'length': len(nop_sequence),
                        'description': f'NOP sled ({len(nop_sequence)} instructions)'
                    })
                nop_sequence = []
        
        # Pattern 2: Push-Pop cancellation
        for i in range(len(pcode_ops) - 1):
            mnem1 = pcode_ops[i].get('mnemonic', '').upper()
            mnem2 = pcode_ops[i + 1].get('mnemonic', '').upper()
            
            if mnem1 == 'PUSH' and mnem2 == 'POP':
                # Check if same register
                out1 = pcode_ops[i].get('inputs', [{}])[0].get('name', '')
                out2 = pcode_ops[i + 1].get('output', {}).get('name', '')
                
                if out1 == out2 or junk_probs[i] > 0.5:
                    patterns.append({
                        'type': 'push_pop_cancellation',
                        'start': i,
                        'end': i + 1,
                        'description': 'Push-Pop pair (likely obfuscation)'
                    })
        
        # Pattern 3: Redundant MOV sequences
        for i in range(len(pcode_ops) - 1):
            mnem1 = pcode_ops[i].get('mnemonic', '').upper()
            mnem2 = pcode_ops[i + 1].get('mnemonic', '').upper()
            
            if 'MOV' in mnem1 and 'MOV' in mnem2:
                out1 = pcode_ops[i].get('output', {}).get('name', '')
                inp2 = pcode_ops[i + 1].get('inputs', [{}])[0].get('name', '')
                out2 = pcode_ops[i + 1].get('output', {}).get('name', '')
                
                if out1 == inp2 or out1 == out2:
                    patterns.append({
                        'type': 'redundant_mov',
                        'start': i,
                        'end': i + 1,
                        'description': 'Redundant MOV sequence'
                    })
        
        # Pattern 4: Self-XOR (clearing register)
        for i, op in enumerate(pcode_ops):
            mnem = op.get('mnemonic', '').upper()
            if 'XOR' in mnem:
                inputs = op.get('inputs', [])
                if len(inputs) >= 2:
                    inp1 = inputs[0].get('name', '')
                    inp2 = inputs[1].get('name', '')
                    if inp1 == inp2:
                        # Could be legitimate zeroing, check context
                        if junk_probs[i] > 0.7:
                            patterns.append({
                                'type': 'self_xor',
                                'index': i,
                                'description': 'Self-XOR (potential obfuscation)'
                            })
        
        # Pattern 5: High junk probability clusters
        cluster_start = None
        for i, prob in enumerate(junk_probs):
            if prob > 0.7:
                if cluster_start is None:
                    cluster_start = i
            else:
                if cluster_start is not None and i - cluster_start >= 3:
                    patterns.append({
                        'type': 'junk_cluster',
                        'start': cluster_start,
                        'end': i - 1,
                        'length': i - cluster_start,
                        'avg_probability': float(np.mean(junk_probs[cluster_start:i])),
                        'description': f'High-confidence junk cluster ({i - cluster_start} instructions)'
                    })
                cluster_start = None
        
        return patterns


# Global sanitizer instance
sanitizer = None


def load_sanitizer():
    global sanitizer
    
    model_path = os.getenv('MODEL_PATH', '/app/models/gnn_sanitizer.pth')
    vocab_path = os.getenv('VOCAB_PATH', '/app/models/vocab.pkl')
    
    local_model = Path(__file__).parent / 'checkpoints' / 'best_model.pth'
    local_vocab = Path(__file__).parent / 'checkpoints' / 'vocab.pkl'
    
    if local_model.exists():
        model_path = str(local_model)
    if local_vocab.exists():
        vocab_path = str(local_vocab)
    
    config = {
        'embed_dim': int(os.getenv('EMBED_DIM', '256')),
        'num_layers': int(os.getenv('NUM_LAYERS', '6')),
        'num_heads': int(os.getenv('NUM_HEADS', '8')),
        'dropout': float(os.getenv('DROPOUT', '0.1')),
        'threshold': float(os.getenv('THRESHOLD', '0.5'))
    }
    
    try:
        sanitizer = GNNSanitizer(
            model_path=model_path,
            vocab_path=vocab_path,
            config=config
        )
        print("GNN Sanitizer loaded successfully")
    except Exception as e:
        print(f"Error loading sanitizer: {e}")
        sanitizer = GNNSanitizer(config=config)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'gnn-sanitizer',
        'model_loaded': sanitizer is not None,
        'device': str(sanitizer.device) if sanitizer else 'unknown',
        'stats': sanitizer.stats if sanitizer else {}
    })


@app.route('/sanitize', methods=['POST'])
def sanitize():
    try:
        data = request.json
        features = data.get('features', [])
        cfg = data.get('cfg')
        threshold = data.get('threshold')
        
        if not features:
            return jsonify({'error': 'features required'}), 400
        
        if not sanitizer:
            return jsonify({'error': 'Sanitizer not initialized'}), 503
        
        # Run sanitization
        result = sanitizer.sanitize(features, cfg, threshold)
        
        return jsonify({
            'sanitized_features': result.sanitized_features,
            'removed_count': result.removed_count,
            'total_count': result.total_count,
            'junk_ratio': result.removed_count / max(result.total_count, 1),
            'obfuscation_score': result.obfuscation_score,
            'junk_indices': result.junk_indices
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        features = data.get('features', [])
        cfg = data.get('cfg')
        
        if not features:
            return jsonify({'error': 'features required'}), 400
        
        if not sanitizer:
            return jsonify({'error': 'Sanitizer not initialized'}), 503
        
        # Run analysis
        analysis = sanitizer.analyze(features, cfg)
        
        return jsonify(analysis)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/batch-sanitize', methods=['POST'])
def batch_sanitize():
    try:
        data = request.json
        functions = data.get('functions', [])
        
        if not functions:
            return jsonify({'error': 'functions required'}), 400
        
        if not sanitizer:
            return jsonify({'error': 'Sanitizer not initialized'}), 503
        
        results = {}
        total_removed = 0
        total_processed = 0
        
        for func in functions:
            name = func.get('name', 'unnamed')
            features = func.get('features', [])
            cfg = func.get('cfg')
            
            result = sanitizer.sanitize(features, cfg)
            
            results[name] = {
                'sanitized_features': result.sanitized_features,
                'removed_count': result.removed_count,
                'obfuscation_score': result.obfuscation_score
            }
            
            total_removed += result.removed_count
            total_processed += result.total_count
        
        return jsonify({
            'results': results,
            'summary': {
                'total_functions': len(functions),
                'total_instructions': total_processed,
                'total_removed': total_removed,
                'overall_junk_ratio': total_removed / max(total_processed, 1)
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    if not sanitizer:
        return jsonify({'error': 'Sanitizer not initialized'}), 503
    
    return jsonify(sanitizer.stats)


if __name__ == '__main__':
    load_sanitizer()
    app.run(host='0.0.0.0', port=5001, debug=True)
