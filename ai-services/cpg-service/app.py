"""
Code Property Graph (CPG) Builder Service

Constructs a hypergraph that fuses:
- Control Flow Graph (CFG): Execution order
- Abstract Syntax Tree (AST): Structural hierarchy  
- Program Dependence Graph (PDG): Data flow (Def-Use chains)

This makes dead code and obfuscation mathematically obvious.
"""

from flask import Flask, request, jsonify
import networkx as nx
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

app = Flask(__name__)

class CodePropertyGraph:
    """
    Multi-layered graph representation combining CFG, AST, and PDG.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.node_counter = 0
        self.dominators = {}
        self.def_use_chains = defaultdict(list)
        
    def add_node(self, node_type: str, attributes: dict) -> int:
        """Add node to CPG with type and attributes."""
        node_id = self.node_counter
        self.graph.add_node(node_id, node_type=node_type, **attributes)
        self.node_counter += 1
        return node_id
    
    def add_edge(self, src: int, dst: int, edge_type: str, **attrs):
        """Add typed edge (CFG, AST, PDG)."""
        self.graph.add_edge(src, dst, edge_type=edge_type, **attrs)
    
    def build_from_pcode(self, pcode_ops: List[dict], cfg: dict) -> dict:
        """
        Build CPG from P-Code and CFG.
        
        Steps:
        1. Create nodes for each instruction
        2. Add CFG edges from control flow
        3. Compute dominators
        4. Build def-use chains for PDG
        5. Detect dead/unreachable code
        """
        # Step 1: Create instruction nodes
        instruction_nodes = []
        for i, op in enumerate(pcode_ops):
            node_id = self.add_node(
                node_type='instruction',
                mnemonic=op.get('mnemonic', 'UNKNOWN'),
                address=op.get('address', f'0x{i:04x}'),
                inputs=op.get('inputs', []),
                output=op.get('output')
            )
            instruction_nodes.append(node_id)
        
        # Step 2: Add CFG edges
        for edge in cfg.get('edges', []):
            if edge['from'] >= 0 and edge['to'] >= 0:
                if edge['from'] < len(instruction_nodes) and edge['to'] < len(instruction_nodes):
                    self.add_edge(
                        instruction_nodes[edge['from']],
                        instruction_nodes[edge['to']],
                        edge_type='cfg',
                        flow_type=edge.get('flow_type', 'sequential')
                    )
        
        # Step 3: Compute dominator tree
        self.compute_dominators(instruction_nodes)
        
        # Step 4: Build def-use chains (PDG)
        self.build_def_use_chains(instruction_nodes, pcode_ops)
        
        # Step 5: Detect anomalies
        dead_nodes = self.detect_dead_code(instruction_nodes)
        unreachable = self.detect_unreachable_code(instruction_nodes)
        circular_deps = self.detect_circular_dependencies()
        
        return {
            'nodes': self.export_nodes(),
            'edges': self.export_edges(),
            'dominators': self.dominators,
            'anomalies': {
                'dead_code': dead_nodes,
                'unreachable': unreachable,
                'circular_dependencies': circular_deps
            }
        }
    
    def compute_dominators(self, nodes: List[int]):
        """
        Compute dominator tree using iterative algorithm.
        A node D dominates N if every path to N goes through D.
        """
        if not nodes:
            return
        
        entry = nodes[0]
        all_nodes = set(nodes)
        
        # Initialize: entry dominates itself, all others dominated by everyone
        dom = {entry: {entry}}
        for node in nodes[1:]:
            dom[node] = all_nodes.copy()
        
        # Iterative refinement
        changed = True
        iterations = 0
        max_iterations = 100
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for node in nodes[1:]:
                # Find predecessors
                preds = list(self.graph.predecessors(node))
                if not preds:
                    continue
                
                # Intersection of predecessor dominators
                new_dom = all_nodes.copy()
                for pred in preds:
                    if pred in dom:
                        new_dom &= dom[pred]
                
                # Add self
                new_dom.add(node)
                
                if new_dom != dom[node]:
                    dom[node] = new_dom
                    changed = True
        
        self.dominators = {node: list(doms) for node, doms in dom.items()}
    
    def build_def_use_chains(self, nodes: List[int], pcode_ops: List[dict]):
        """
        Build Program Dependence Graph by tracking variable definitions and uses.
        """
        # Track where each variable is defined
        definitions = defaultdict(list)
        
        for i, node in enumerate(nodes):
            op = pcode_ops[i] if i < len(pcode_ops) else {}
            
            # Track output (definition)
            output = op.get('output')
            if output:
                var_name = f"v{output.get('offset', 0)}"
                definitions[var_name].append(node)
            
            # Track inputs (uses)
            for inp in op.get('inputs', []):
                var_name = f"v{inp.get('offset', 0)}"
                
                # Find most recent definition
                if var_name in definitions and definitions[var_name]:
                    def_node = definitions[var_name][-1]
                    self.add_edge(def_node, node, edge_type='pdg', dep_type='def-use')
                    self.def_use_chains[def_node].append(node)
    
    def detect_dead_code(self, nodes: List[int]) -> List[int]:
        """
        Detect dead code: nodes whose outputs are never used.
        """
        dead_nodes = []
        
        for node in nodes:
            # Check if this node has any outgoing PDG edges
            pdg_edges = [e for e in self.graph.edges(node, data=True) 
                        if e[2].get('edge_type') == 'pdg']
            
            # Check if output is used in any computation
            has_side_effect = self.graph.nodes[node].get('mnemonic') in [
                'CALL', 'STORE', 'RETURN', 'BRANCH'
            ]
            
            if not pdg_edges and not has_side_effect:
                dead_nodes.append(node)
        
        return dead_nodes
    
    def detect_unreachable_code(self, nodes: List[int]) -> List[int]:
        """
        Detect unreachable code: nodes not reachable from entry via CFG.
        """
        if not nodes:
            return []
        
        entry = nodes[0]
        reachable = set()
        
        # BFS from entry
        queue = [entry]
        visited = {entry}
        
        while queue:
            node = queue.pop(0)
            reachable.add(node)
            
            # Follow CFG edges
            for successor in self.graph.successors(node):
                edge_data = self.graph.get_edge_data(node, successor)
                if any(e.get('edge_type') == 'cfg' for e in edge_data.values()):
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)
        
        unreachable = [n for n in nodes if n not in reachable]
        return unreachable
    
    def detect_circular_dependencies(self) -> List[List[int]]:
        """
        Detect circular def-use chains (usually junk code).
        """
        # Build PDG subgraph
        pdg_subgraph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            if data.get('edge_type') == 'pdg':
                pdg_subgraph.add_edge(u, v)
        
        # Find strongly connected components with size > 1
        cycles = [list(component) for component in nx.strongly_connected_components(pdg_subgraph)
                 if len(component) > 1]
        
        return cycles
    
    def export_nodes(self) -> List[dict]:
        """Export nodes with all attributes."""
        return [
            {'id': node, **self.graph.nodes[node]}
            for node in self.graph.nodes()
        ]
    
    def export_edges(self) -> List[dict]:
        """Export edges with types."""
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'from': u,
                'to': v,
                'type': data.get('edge_type'),
                **{k: v for k, v in data.items() if k != 'edge_type'}
            })
        return edges


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'cpg-builder'})


@app.route('/build-cpg', methods=['POST'])
def build_cpg():
    """
    Build Code Property Graph from P-Code and CFG.
    
    Request:
    {
        "pcode": [...],
        "cfg": {"nodes": [...], "edges": [...]}
    }
    
    Response:
    {
        "cpg": {
            "nodes": [...],
            "edges": [...],
            "dominators": {...},
            "anomalies": {
                "dead_code": [...],
                "unreachable": [...],
                "circular_dependencies": [...]
            }
        }
    }
    """
    try:
        data = request.json
        pcode = data.get('pcode', [])
        cfg = data.get('cfg', {})
        
        if not pcode:
            return jsonify({'error': 'pcode required'}), 400
        
        # Build CPG
        cpg_builder = CodePropertyGraph()
        cpg_data = cpg_builder.build_from_pcode(pcode, cfg)
        
        return jsonify({
            'cpg': cpg_data,
            'stats': {
                'total_nodes': len(cpg_data['nodes']),
                'total_edges': len(cpg_data['edges']),
                'dead_code_nodes': len(cpg_data['anomalies']['dead_code']),
                'unreachable_nodes': len(cpg_data['anomalies']['unreachable']),
                'circular_deps': len(cpg_data['anomalies']['circular_dependencies'])
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
