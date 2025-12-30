

from flask import Flask, request, jsonify
import networkx as nx
import torch
import torch.nn as nn
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib

app = Flask(__name__)

class NodeType(Enum):
    INSTRUCTION = "instruction"
    BASIC_BLOCK = "basic_block"
    FUNCTION = "function"
    VARIABLE = "variable"
    CONSTANT = "constant"


class EdgeType(Enum):
    CFG = "cfg"
    AST = "ast"
    PDG_DATA = "pdg_data"
    PDG_CONTROL = "pdg_control"
    CALL = "call"
    RETURN = "return"
    ALIAS = "alias"


@dataclass
class AbstractValue:
    is_constant: bool = False
    constant_value: Optional[int] = None
    is_tainted: bool = False
    taint_source: Optional[str] = None
    value_range: Tuple[Optional[int], Optional[int]] = (None, None)
    
    def __str__(self):
        if self.is_constant:
            return f"Const({self.constant_value})"
        elif self.value_range[0] is not None and self.value_range[1] is not None:
            return f"Range({self.value_range[0]}, {self.value_range[1]})"
        else:
            return "Top"


@dataclass
class SemanticPattern:
    """Detected semantic pattern/anomaly."""
    pattern_type: str
    node_ids: List[int]
    confidence: float
    description: str
    severity: str  # "low", "medium", "high"


# ============================================================================
# Graph Neural Network for Pattern Detection
# ============================================================================

class CPGEmbedding(nn.Module):
    # ...existing code...
    
    def __init__(self, node_features: int = 64, hidden_dim: int = 128, 
                 num_layers: int = 3, num_edge_types: int = 7):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        
        # Node type embedding
        self.node_type_emb = nn.Embedding(10, node_features)
        
        # Mnemonic embedding (hash-based)
        self.mnemonic_emb = nn.Embedding(256, node_features)
        
        # Edge type embedding
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = node_features * 2 if i == 0 else hidden_dim
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ))
        
        # Output head for pattern detection
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # 5 pattern types
        )
    
    def forward(self, node_types: torch.Tensor, mnemonics: torch.Tensor,
                edge_index: torch.Tensor, edge_types: torch.Tensor) -> torch.Tensor:
        # ...existing code...
        # Initial node features
        type_emb = self.node_type_emb(node_types)
        mnem_emb = self.mnemonic_emb(mnemonics)
        h = torch.cat([type_emb, mnem_emb], dim=-1)
        
        # Edge features
        edge_emb = self.edge_type_emb(edge_types)
        
        # Message passing
        for layer in self.layers:
            # Aggregate messages from neighbors
            messages = self._aggregate_messages(h, edge_index, edge_emb)
            h = layer(torch.cat([h if h.size(-1) == self.hidden_dim else h, messages], dim=-1))
        
        return h
    
    def _aggregate_messages(self, h: torch.Tensor, edge_index: torch.Tensor,
                           edge_emb: torch.Tensor) -> torch.Tensor:
        # ...existing code...
        num_nodes = h.size(0)
        messages = torch.zeros(num_nodes, self.hidden_dim, device=h.device)
        
        if edge_index.size(1) == 0:
            return messages
        
        src, dst = edge_index
        
        # Message = source embedding + edge embedding
        src_h = h[src] if h.size(-1) == self.hidden_dim else h[src, :self.hidden_dim]
        msg = src_h + edge_emb
        
        # Aggregate by destination
        messages.index_add_(0, dst, msg)
        
        # Normalize by degree
        degree = torch.zeros(num_nodes, device=h.device)
        degree.index_add_(0, dst, torch.ones(dst.size(0), device=h.device))
        degree = degree.clamp(min=1)
        
        messages = messages / degree.unsqueeze(-1)
        
        return messages
    
    def detect_patterns(self, h: torch.Tensor) -> torch.Tensor:
        # ...existing code...
        return self.pattern_head(h)


# ============================================================================
# Code Property Graph
# ============================================================================

class CodePropertyGraph:
    # ...existing code...
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.dominators: Dict[int, Set[int]] = {}
        self.post_dominators: Dict[int, Set[int]] = {}
        self.def_use_chains: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.abstract_values: Dict[int, Dict[str, AbstractValue]] = {}
        self.taint_map: Dict[int, Set[str]] = defaultdict(set)
        self.call_sites: List[Dict] = []
        
        # Pattern detector (loaded lazily)
        self._pattern_model = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @property
    def pattern_model(self):
        if self._pattern_model is None:
            self._pattern_model = CPGEmbedding().to(self._device)
            self._pattern_model.eval()
        return self._pattern_model
    
    def add_node(self, node_type: str, **attributes) -> int:
        # ...existing code...
        node_id = self.node_counter
        self.graph.add_node(node_id, node_type=node_type, **attributes)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, src: int, dst: int, edge_type: str, **attrs):
        # ...existing code...
        self.graph.add_edge(src, dst, edge_type=edge_type, **attrs)
    
    def build_from_pcode(self, pcode_ops: List[Dict], cfg: Dict) -> Dict:
        # ...existing code...
        # Step 1: Create instruction nodes
        instruction_nodes = self._create_instruction_nodes(pcode_ops)
        
        # Step 2: Add CFG edges
        self._add_cfg_edges(instruction_nodes, cfg)
        
        # Step 3: Build basic blocks
        basic_blocks = self._identify_basic_blocks(instruction_nodes)
        
        # Step 4: Compute dominators and post-dominators
        if instruction_nodes:
            self._compute_dominators(instruction_nodes)
            self._compute_post_dominators(instruction_nodes)
        
        # Step 5: Build def-use chains (data dependencies)
        self._build_def_use_chains(instruction_nodes, pcode_ops)
        
        # Step 6: Add control dependencies
        self._add_control_dependencies(instruction_nodes)
        
        # Step 7: Abstract interpretation
        self._abstract_interpret(instruction_nodes, pcode_ops)
        
        # Step 8: Taint analysis
        self._taint_analysis(instruction_nodes, pcode_ops)
        
        # Step 9: Detect semantic patterns
        patterns = self._detect_patterns(instruction_nodes, pcode_ops)
        
        # Step 10: Detect anomalies
        anomalies = self._detect_anomalies(instruction_nodes)
        
        return {
            'nodes': self._export_nodes(),
            'edges': self._export_edges(),
            'basic_blocks': basic_blocks,
            'dominators': {k: list(v) for k, v in self.dominators.items()},
            'post_dominators': {k: list(v) for k, v in self.post_dominators.items()},
            'def_use_chains': dict(self.def_use_chains),
            'abstract_values': {
                k: {var: str(val) for var, val in vals.items()}
                for k, vals in self.abstract_values.items()
            },
            'taint_map': {k: list(v) for k, v in self.taint_map.items()},
            'patterns': [
                {
                    'type': p.pattern_type,
                    'nodes': p.node_ids,
                    'confidence': p.confidence,
                    'description': p.description,
                    'severity': p.severity
                }
                for p in patterns
            ],
            'anomalies': anomalies
        }
    
    def _create_instruction_nodes(self, pcode_ops: List[Dict]) -> List[int]:
        # ...existing code...
        nodes = []
        for i, op in enumerate(pcode_ops):
            node_id = self.add_node(
                node_type='instruction',
                index=i,
                mnemonic=op.get('mnemonic', 'NOP'),
                address=op.get('address', f'0x{i:04x}'),
                inputs=op.get('inputs', []),
                output=op.get('output'),
                size=op.get('size', 4)
            )
            nodes.append(node_id)
        return nodes
    
    def _add_cfg_edges(self, nodes: List[int], cfg: Dict):
        # ...existing code...
        edges = cfg.get('edges', [])
        
        # Handle both list and dict edge formats
        for edge in edges:
            if isinstance(edge, dict):
                src_idx = edge.get('from', edge.get('source', -1))
                dst_idx = edge.get('to', edge.get('target', -1))
                flow_type = edge.get('flow_type', edge.get('type', 'sequential'))
            else:
                continue
            
            if 0 <= src_idx < len(nodes) and 0 <= dst_idx < len(nodes):
                self.add_edge(
                    nodes[src_idx],
                    nodes[dst_idx],
                    edge_type='cfg',
                    flow_type=flow_type
                )
        
        # Add implicit sequential edges if no edges provided
        if not edges:
            for i in range(len(nodes) - 1):
                mnemonic = self.graph.nodes[nodes[i]].get('mnemonic', '')
                if mnemonic not in ('RETURN', 'BRANCH'):
                    self.add_edge(nodes[i], nodes[i + 1], edge_type='cfg', flow_type='sequential')
    
    def _identify_basic_blocks(self, nodes: List[int]) -> List[Dict]:
        """Identify basic blocks from instruction nodes."""
        if not nodes:
            return []
        
        # Leaders: first instruction, targets of jumps, instructions after jumps
        leaders = {0}
        
        for node in nodes:
            mnemonic = self.graph.nodes[node].get('mnemonic', '')
            if mnemonic in ('BRANCH', 'CBRANCH', 'CALL', 'RETURN'):
                # Add next instruction as leader
                idx = self.graph.nodes[node].get('index', 0)
                if idx + 1 < len(nodes):
                    leaders.add(idx + 1)
                
                # Add branch targets as leaders
                for succ in self.graph.successors(node):
                    edge_data = self.graph.get_edge_data(node, succ)
                    if edge_data and any(e.get('edge_type') == 'cfg' for e in edge_data.values()):
                        succ_idx = self.graph.nodes[succ].get('index', 0)
                        leaders.add(succ_idx)
        
        # Create basic blocks
        sorted_leaders = sorted(leaders)
        basic_blocks = []
        
        for i, leader_idx in enumerate(sorted_leaders):
            # Find block end
            end_idx = sorted_leaders[i + 1] - 1 if i + 1 < len(sorted_leaders) else len(nodes) - 1
            
            block_nodes = [nodes[j] for j in range(leader_idx, end_idx + 1)]
            
            if block_nodes:
                block = {
                    'id': len(basic_blocks),
                    'nodes': block_nodes,
                    'entry': block_nodes[0],
                    'exit': block_nodes[-1],
                    'start_idx': leader_idx,
                    'end_idx': end_idx
                }
                basic_blocks.append(block)
        
        return basic_blocks
    
    def _compute_dominators(self, nodes: List[int]):
        """Compute dominator sets using iterative algorithm."""
        if not nodes:
            return
        
        entry = nodes[0]
        all_nodes = set(nodes)
        
        # Initialize
        self.dominators = {entry: {entry}}
        for node in nodes[1:]:
            self.dominators[node] = all_nodes.copy()
        
        # Iterate until fixpoint
        changed = True
        max_iter = 100
        iteration = 0
        
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            
            for node in nodes[1:]:
                preds = [p for p in self.graph.predecessors(node)
                        if any(e.get('edge_type') == 'cfg' 
                              for e in self.graph.get_edge_data(p, node).values())]
                
                if not preds:
                    continue
                
                # Intersect predecessor dominators
                new_dom = all_nodes.copy()
                for pred in preds:
                    if pred in self.dominators:
                        new_dom &= self.dominators[pred]
                
                new_dom.add(node)
                
                if new_dom != self.dominators[node]:
                    self.dominators[node] = new_dom
                    changed = True
    
    def _compute_post_dominators(self, nodes: List[int]):
        """Compute post-dominator sets (reverse dominators)."""
        if not nodes:
            return
        
        # Find exit nodes (nodes with no CFG successors)
        exit_nodes = []
        for node in nodes:
            cfg_succs = [s for s in self.graph.successors(node)
                        if any(e.get('edge_type') == 'cfg'
                              for e in self.graph.get_edge_data(node, s).values())]
            if not cfg_succs:
                exit_nodes.append(node)
        
        if not exit_nodes:
            exit_nodes = [nodes[-1]]
        
        all_nodes = set(nodes)
        
        # Initialize
        for exit_node in exit_nodes:
            self.post_dominators[exit_node] = {exit_node}
        for node in nodes:
            if node not in self.post_dominators:
                self.post_dominators[node] = all_nodes.copy()
        
        # Iterate until fixpoint (reverse CFG direction)
        changed = True
        max_iter = 100
        iteration = 0
        
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            
            for node in reversed(nodes):
                if node in exit_nodes:
                    continue
                
                # Successors become predecessors in post-dominator computation
                succs = [s for s in self.graph.successors(node)
                        if any(e.get('edge_type') == 'cfg'
                              for e in self.graph.get_edge_data(node, s).values())]
                
                if not succs:
                    continue
                
                new_pdom = all_nodes.copy()
                for succ in succs:
                    if succ in self.post_dominators:
                        new_pdom &= self.post_dominators[succ]
                
                new_pdom.add(node)
                
                if new_pdom != self.post_dominators.get(node, set()):
                    self.post_dominators[node] = new_pdom
                    changed = True
    
    def _build_def_use_chains(self, nodes: List[int], pcode_ops: List[Dict]):
        """Build def-use chains for data dependencies."""
        # Track variable definitions
        var_defs: Dict[str, List[int]] = defaultdict(list)
        
        for i, node in enumerate(nodes):
            if i >= len(pcode_ops):
                continue
            
            op = pcode_ops[i]
            
            # Track output (definition)
            output = op.get('output')
            if output:
                var_name = self._get_var_name(output)
                var_defs[var_name].append(node)
                
                # Create variable node
                var_node = self.add_node(
                    node_type='variable',
                    name=var_name,
                    definition_node=node
                )
                self.add_edge(node, var_node, edge_type='ast', relation='defines')
            
            # Track inputs (uses)
            for inp in op.get('inputs', []):
                var_name = self._get_var_name(inp)
                
                # Find reaching definition
                if var_name in var_defs and var_defs[var_name]:
                    def_node = var_defs[var_name][-1]
                    self.add_edge(def_node, node, edge_type='pdg_data', variable=var_name)
                    self.def_use_chains[var_name].append((def_node, node))
    
    def _get_var_name(self, operand: Any) -> str:
        """Extract variable name from operand."""
        if isinstance(operand, dict):
            if 'name' in operand:
                return operand['name']
            elif 'register' in operand:
                return f"reg_{operand['register']}"
            elif 'offset' in operand:
                return f"mem_{operand['offset']}"
        return f"var_{hash(str(operand)) % 10000}"
    
    def _add_control_dependencies(self, nodes: List[int]):
        """Add control dependencies based on post-dominators."""
        for node in nodes:
            mnemonic = self.graph.nodes[node].get('mnemonic', '')
            
            if mnemonic in ('CBRANCH', 'BRANCHIND'):
                # Find nodes control-dependent on this branch
                for other in nodes:
                    if other == node:
                        continue
                    
                    # Node B is control-dependent on A if:
                    # - There exists a path from A to B
                    # - B does not post-dominate A
                    
                    if other not in self.post_dominators.get(node, {node}):
                        # Check if there's a path
                        if nx.has_path(self.graph, node, other):
                            self.add_edge(node, other, edge_type='pdg_control')
    
    def _abstract_interpret(self, nodes: List[int], pcode_ops: List[Dict]):
        """Perform abstract interpretation to track values."""
        state: Dict[str, AbstractValue] = {}
        
        for i, node in enumerate(nodes):
            if i >= len(pcode_ops):
                continue
            
            op = pcode_ops[i]
            mnemonic = op.get('mnemonic', '')
            inputs = op.get('inputs', [])
            output = op.get('output')
            
            # Interpret operation
            if mnemonic == 'COPY':
                if inputs and output:
                    var_name = self._get_var_name(output)
                    input_name = self._get_var_name(inputs[0])
                    
                    if input_name in state:
                        state[var_name] = state[input_name]
                    elif isinstance(inputs[0], dict) and 'value' in inputs[0]:
                        state[var_name] = AbstractValue(
                            is_constant=True,
                            constant_value=inputs[0]['value']
                        )
            
            elif mnemonic in ('INT_ADD', 'INT_SUB', 'INT_MULT'):
                if len(inputs) >= 2 and output:
                    var_name = self._get_var_name(output)
                    
                    in1 = self._get_abstract_value(inputs[0], state)
                    in2 = self._get_abstract_value(inputs[1], state)
                    
                    if in1.is_constant and in2.is_constant:
                        if mnemonic == 'INT_ADD':
                            result = in1.constant_value + in2.constant_value
                        elif mnemonic == 'INT_SUB':
                            result = in1.constant_value - in2.constant_value
                        else:
                            result = in1.constant_value * in2.constant_value
                        
                        state[var_name] = AbstractValue(is_constant=True, constant_value=result)
                    else:
                        state[var_name] = AbstractValue()
            
            # Store state snapshot
            self.abstract_values[node] = state.copy()
    
    def _get_abstract_value(self, operand: Any, state: Dict[str, AbstractValue]) -> AbstractValue:
        """Get abstract value for operand."""
        if isinstance(operand, dict):
            if 'value' in operand:
                return AbstractValue(is_constant=True, constant_value=operand['value'])
            
            var_name = self._get_var_name(operand)
            if var_name in state:
                return state[var_name]
        
        return AbstractValue()
    
    def _taint_analysis(self, nodes: List[int], pcode_ops: List[Dict]):
        """Perform taint analysis for data flow tracking."""
        # Taint sources: LOAD from external, CALL returns
        tainted: Dict[str, str] = {}  # var_name -> taint_source
        
        for i, node in enumerate(nodes):
            if i >= len(pcode_ops):
                continue
            
            op = pcode_ops[i]
            mnemonic = op.get('mnemonic', '')
            output = op.get('output')
            inputs = op.get('inputs', [])
            
            # Check for taint sources
            if mnemonic in ('LOAD', 'CALL', 'CALLIND'):
                if output:
                    var_name = self._get_var_name(output)
                    tainted[var_name] = f"{mnemonic}@{node}"
                    self.taint_map[node].add(var_name)
            
            # Propagate taint
            elif output:
                var_name = self._get_var_name(output)
                
                for inp in inputs:
                    inp_name = self._get_var_name(inp)
                    if inp_name in tainted:
                        tainted[var_name] = tainted[inp_name]
                        self.taint_map[node].add(var_name)
                        break
    
    def _detect_patterns(self, nodes: List[int], pcode_ops: List[Dict]) -> List[SemanticPattern]:
        """Detect semantic patterns using heuristics and GNN."""
        patterns = []
        
        # Pattern 1: Dead assignments
        patterns.extend(self._detect_dead_assignments(nodes, pcode_ops))
        
        # Pattern 2: Opaque predicates
        patterns.extend(self._detect_opaque_predicates(nodes, pcode_ops))
        
        # Pattern 3: Instruction substitution
        patterns.extend(self._detect_instruction_substitution(nodes, pcode_ops))
        
        # Pattern 4: Control flow flattening
        patterns.extend(self._detect_cfg_flattening(nodes))
        
        # Pattern 5: Junk code insertion
        patterns.extend(self._detect_junk_code(nodes, pcode_ops))
        
        return patterns
    
    def _detect_dead_assignments(self, nodes: List[int], pcode_ops: List[Dict]) -> List[SemanticPattern]:
        """Detect assignments whose results are never used."""
        patterns = []
        
        # Build use map
        uses: Dict[str, List[int]] = defaultdict(list)
        defs: Dict[str, List[int]] = defaultdict(list)
        
        for i, node in enumerate(nodes):
            if i >= len(pcode_ops):
                continue
            
            op = pcode_ops[i]
            output = op.get('output')
            inputs = op.get('inputs', [])
            
            if output:
                var_name = self._get_var_name(output)
                defs[var_name].append(node)
            
            for inp in inputs:
                var_name = self._get_var_name(inp)
                uses[var_name].append(node)
        
        # Find definitions without uses
        dead_defs = []
        for var_name, def_nodes in defs.items():
            use_nodes = uses.get(var_name, [])
            
            for def_node in def_nodes:
                # Check if any use comes after this definition
                has_use = any(use > def_node for use in use_nodes)
                
                if not has_use:
                    mnemonic = self.graph.nodes[def_node].get('mnemonic', '')
                    # Exclude side-effect instructions
                    if mnemonic not in ('CALL', 'STORE', 'RETURN'):
                        dead_defs.append(def_node)
        
        if dead_defs:
            patterns.append(SemanticPattern(
                pattern_type='dead_assignment',
                node_ids=dead_defs,
                confidence=0.9,
                description=f"Found {len(dead_defs)} assignments whose values are never used",
                severity='medium'
            ))
        
        return patterns
    
    def _detect_opaque_predicates(self, nodes: List[int], pcode_ops: List[Dict]) -> List[SemanticPattern]:
        """Detect opaque predicates (conditions that always evaluate the same way)."""
        patterns = []
        opaque_nodes = []
        
        for i, node in enumerate(nodes):
            if i >= len(pcode_ops):
                continue
            
            op = pcode_ops[i]
            mnemonic = op.get('mnemonic', '')
            
            if mnemonic == 'CBRANCH':
                # Check if condition can be evaluated statically
                inputs = op.get('inputs', [])
                
                if inputs:
                    # Check abstract value
                    abs_val = self.abstract_values.get(node, {})
                    
                    # If the condition input is a constant, it's an opaque predicate
                    for inp in inputs:
                        var_name = self._get_var_name(inp)
                        if var_name in abs_val:
                            val = abs_val[var_name]
                            if val.is_constant:
                                opaque_nodes.append(node)
                                break
        
        if opaque_nodes:
            patterns.append(SemanticPattern(
                pattern_type='opaque_predicate',
                node_ids=opaque_nodes,
                confidence=0.85,
                description=f"Found {len(opaque_nodes)} potential opaque predicates",
                severity='high'
            ))
        
        return patterns
    
    def _detect_instruction_substitution(self, nodes: List[int], pcode_ops: List[Dict]) -> List[SemanticPattern]:
        """Detect instruction substitution obfuscation."""
        patterns = []
        
        # Look for complex sequences that compute simple operations
        suspicious_sequences = []
        
        for i in range(len(nodes) - 2):
            if i + 2 >= len(pcode_ops):
                continue
            
            ops = [pcode_ops[j] for j in range(i, i + 3)]
            mnemonics = [op.get('mnemonic', '') for op in ops]
            
            # Pattern: XOR XOR (identity obfuscation)
            if mnemonics == ['INT_XOR', 'INT_XOR', 'INT_XOR']:
                suspicious_sequences.append(nodes[i:i+3])
            
            # Pattern: NEG ADD (subtraction obfuscation)
            if mnemonics == ['INT_NEGATE', 'INT_ADD', 'INT_ADD']:
                suspicious_sequences.append(nodes[i:i+3])
            
            # Pattern: Multiplication by shift and add
            if mnemonics == ['INT_LEFT', 'INT_ADD', 'INT_ADD']:
                suspicious_sequences.append(nodes[i:i+3])
        
        if suspicious_sequences:
            all_nodes = [n for seq in suspicious_sequences for n in seq]
            patterns.append(SemanticPattern(
                pattern_type='instruction_substitution',
                node_ids=all_nodes,
                confidence=0.75,
                description=f"Found {len(suspicious_sequences)} instruction substitution patterns",
                severity='medium'
            ))
        
        return patterns
    
    def _detect_cfg_flattening(self, nodes: List[int]) -> List[SemanticPattern]:
        """Detect control flow flattening obfuscation."""
        patterns = []
        
        # CFG flattening typically results in:
        # - A dispatcher block with many outgoing edges
        # - Multiple blocks all returning to the same dispatcher
        
        # Count in-degree and out-degree
        in_degree = defaultdict(int)
        out_degree = defaultdict(int)
        
        for node in nodes:
            for succ in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, succ)
                if edge_data and any(e.get('edge_type') == 'cfg' for e in edge_data.values()):
                    out_degree[node] += 1
                    in_degree[succ] += 1
        
        # Look for dispatcher pattern
        dispatchers = [n for n in nodes if out_degree[n] > 5]  # Many outgoing edges
        
        for dispatcher in dispatchers:
            # Check if many nodes return to this dispatcher
            returning = [n for n in nodes 
                        if n != dispatcher and dispatcher in [
                            s for s in self.graph.successors(n)
                            if any(e.get('edge_type') == 'cfg' 
                                  for e in self.graph.get_edge_data(n, s).values())
                        ]]
            
            if len(returning) > 3:
                patterns.append(SemanticPattern(
                    pattern_type='cfg_flattening',
                    node_ids=[dispatcher] + returning,
                    confidence=0.8,
                    description=f"Dispatcher at {dispatcher} with {len(returning)} returning blocks",
                    severity='high'
                ))
        
        return patterns
    
    def _detect_junk_code(self, nodes: List[int], pcode_ops: List[Dict]) -> List[SemanticPattern]:
        """Detect junk code insertions."""
        patterns = []
        junk_candidates = []
        
        for i, node in enumerate(nodes):
            if i >= len(pcode_ops):
                continue
            
            op = pcode_ops[i]
            mnemonic = op.get('mnemonic', '')
            output = op.get('output')
            inputs = op.get('inputs', [])
            
            # Pattern 1: NOP-equivalent operations
            if mnemonic == 'INT_ADD':
                # x + 0 = x
                for inp in inputs:
                    if isinstance(inp, dict) and inp.get('value') == 0:
                        junk_candidates.append(node)
                        break
            
            elif mnemonic == 'INT_MULT':
                # x * 1 = x
                for inp in inputs:
                    if isinstance(inp, dict) and inp.get('value') == 1:
                        junk_candidates.append(node)
                        break
            
            elif mnemonic == 'INT_XOR':
                # x ^ 0 = x
                for inp in inputs:
                    if isinstance(inp, dict) and inp.get('value') == 0:
                        junk_candidates.append(node)
                        break
            
            elif mnemonic == 'COPY':
                # Copy to self
                if output and inputs:
                    if self._get_var_name(output) == self._get_var_name(inputs[0]):
                        junk_candidates.append(node)
        
        if junk_candidates:
            patterns.append(SemanticPattern(
                pattern_type='junk_code',
                node_ids=junk_candidates,
                confidence=0.9,
                description=f"Found {len(junk_candidates)} NOP-equivalent operations",
                severity='low'
            ))
        
        return patterns
    
    def _detect_anomalies(self, nodes: List[int]) -> Dict:
        """Detect structural anomalies."""
        return {
            'dead_code': self._find_dead_code(nodes),
            'unreachable': self._find_unreachable(nodes),
            'circular_dependencies': self._find_circular_deps()
        }
    
    def _find_dead_code(self, nodes: List[int]) -> List[int]:
        """Find dead code (output never used, no side effects)."""
        dead = []
        
        for node in nodes:
            # Check for outgoing PDG edges
            pdg_edges = [s for s in self.graph.successors(node)
                        if any(e.get('edge_type') == 'pdg_data'
                              for e in self.graph.get_edge_data(node, s).values())]
            
            if not pdg_edges:
                mnemonic = self.graph.nodes[node].get('mnemonic', '')
                if mnemonic not in ('CALL', 'STORE', 'RETURN', 'BRANCH', 'CBRANCH'):
                    dead.append(node)
        
        return dead
    
    def _find_unreachable(self, nodes: List[int]) -> List[int]:
        """Find unreachable code."""
        if not nodes:
            return []
        
        reachable = set()
        queue = [nodes[0]]
        
        while queue:
            node = queue.pop(0)
            if node in reachable:
                continue
            reachable.add(node)
            
            for succ in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, succ)
                if edge_data and any(e.get('edge_type') == 'cfg' for e in edge_data.values()):
                    if succ not in reachable:
                        queue.append(succ)
        
        return [n for n in nodes if n not in reachable]
    
    def _find_circular_deps(self) -> List[List[int]]:
        """Find circular data dependencies."""
        pdg = nx.DiGraph()
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'pdg_data':
                pdg.add_edge(u, v)
        
        cycles = [list(c) for c in nx.strongly_connected_components(pdg) if len(c) > 1]
        return cycles
    
    def _export_nodes(self) -> List[Dict]:
        """Export nodes with all attributes."""
        return [
            {'id': node, **dict(self.graph.nodes[node])}
            for node in self.graph.nodes()
        ]
    
    def _export_edges(self) -> List[Dict]:
        """Export edges with types."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'from': u,
                'to': v,
                'type': data.get('edge_type', 'unknown'),
                **{k: v for k, v in data.items() if k != 'edge_type'}
            })
        return edges
    
    def compute_slice(self, criterion_node: int, backward: bool = True) -> Set[int]:
        """
        Compute program slice for a given criterion node.
        
        Args:
            criterion_node: Node to slice from
            backward: If True, compute backward slice (dependencies)
                     If False, compute forward slice (dependents)
        
        Returns:
            Set of nodes in the slice
        """
        slice_nodes = {criterion_node}
        worklist = [criterion_node]
        
        while worklist:
            node = worklist.pop(0)
            
            # Get relevant edges based on direction
            if backward:
                # Follow PDG edges backward (predecessors define us)
                neighbors = self.graph.predecessors(node)
            else:
                # Follow PDG edges forward (successors use us)
                neighbors = self.graph.successors(node)
            
            for neighbor in neighbors:
                edge_data = self.graph.get_edge_data(
                    neighbor if backward else node,
                    node if backward else neighbor
                )
                
                if edge_data is None:
                    continue
                
                # Include if connected by PDG edge
                if any(e.get('edge_type') in ('pdg_data', 'pdg_control') 
                      for e in edge_data.values()):
                    if neighbor not in slice_nodes:
                        slice_nodes.add(neighbor)
                        worklist.append(neighbor)
        
        return slice_nodes


# ============================================================================
# Flask Routes
# ============================================================================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'cpg-builder-v2',
        'features': [
            'CFG construction',
            'Dominator analysis',
            'Def-use chains',
            'Control dependencies',
            'Abstract interpretation',
            'Taint analysis',
            'Pattern detection',
            'Program slicing'
        ]
    })


@app.route('/build-cpg', methods=['POST'])
def build_cpg():
    """
    Build enhanced Code Property Graph.
    
    Request:
    {
        "pcode": [...],
        "cfg": {"nodes": [...], "edges": [...]}
    }
    """
    try:
        data = request.json
        pcode = data.get('pcode', [])
        cfg = data.get('cfg', {})
        
        if not pcode:
            return jsonify({'error': 'pcode required'}), 400
        
        cpg = CodePropertyGraph()
        result = cpg.build_from_pcode(pcode, cfg)
        
        return jsonify({
            'cpg': result,
            'stats': {
                'total_nodes': len(result['nodes']),
                'total_edges': len(result['edges']),
                'basic_blocks': len(result['basic_blocks']),
                'variables': len(result['def_use_chains']),
                'patterns': len(result['patterns']),
                'dead_code': len(result['anomalies']['dead_code']),
                'unreachable': len(result['anomalies']['unreachable'])
            }
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/slice', methods=['POST'])
def compute_slice():
    """
    Compute program slice.
    
    Request:
    {
        "pcode": [...],
        "cfg": {...},
        "criterion_index": 10,
        "backward": true
    }
    """
    try:
        data = request.json
        pcode = data.get('pcode', [])
        cfg = data.get('cfg', {})
        criterion_index = data.get('criterion_index', 0)
        backward = data.get('backward', True)
        
        cpg = CodePropertyGraph()
        cpg.build_from_pcode(pcode, cfg)
        
        # Get node for criterion
        if criterion_index >= len(cpg.graph.nodes):
            return jsonify({'error': 'Invalid criterion_index'}), 400
        
        slice_nodes = cpg.compute_slice(criterion_index, backward=backward)
        
        return jsonify({
            'criterion': criterion_index,
            'direction': 'backward' if backward else 'forward',
            'slice_nodes': list(slice_nodes),
            'slice_size': len(slice_nodes)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect-patterns', methods=['POST'])
def detect_patterns():
    """
    Detect obfuscation patterns in code.
    
    Request:
    {
        "pcode": [...],
        "cfg": {...}
    }
    """
    try:
        data = request.json
        pcode = data.get('pcode', [])
        cfg = data.get('cfg', {})
        
        cpg = CodePropertyGraph()
        result = cpg.build_from_pcode(pcode, cfg)
        
        return jsonify({
            'patterns': result['patterns'],
            'anomalies': result['anomalies'],
            'summary': {
                'total_patterns': len(result['patterns']),
                'by_type': {
                    p['type']: sum(1 for x in result['patterns'] if x['type'] == p['type'])
                    for p in result['patterns']
                }
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Starting Enhanced CPG Builder Service")
    app.run(host='0.0.0.0', port=5005, debug=True)
