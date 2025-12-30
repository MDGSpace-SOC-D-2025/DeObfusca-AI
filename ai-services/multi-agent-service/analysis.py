"""
Real Code Analysis Module for Multi-Agent Decompilation
Provides AST parsing, CFG construction, and data flow analysis
"""

import re
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class BasicBlock:
    """Represents a basic block in the CFG."""
    id: int
    successors: List[int] = field(default_factory=list)
    dominators: Set[int] = field(default_factory=set)
    is_loop_header: bool = False
    is_conditional: bool = False


@dataclass  
class Variable:
    """Represents a variable with type and usage information."""
    class BasicBlock:
    definitions: List[int] = field(default_factory=list)  # Block IDs
    uses: List[int] = field(default_factory=list)  # Block IDs
    is_pointer: bool = False
    is_array: bool = False
    array_size: Optional[int] = None


class PCodeParser:
    """Parse P-Code/IR instructions into structured format."""
    
    # P-Code operation categories
    ARITHMETIC_OPS = {'INT_ADD', 'INT_SUB', 'INT_MULT', 'INT_DIV', 'INT_MOD',
    class Variable:
    BRANCH_OPS = {'BRANCH', 'CBRANCH', 'BRANCHIND', 'CALL', 'RETURN'}
    COMPARE_OPS = {'INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS', 'INT_LESSEQUAL',
                   'INT_SLESS', 'INT_SLESSEQUAL', 'FLOAT_EQUAL', 'FLOAT_LESS'}
    BITWISE_OPS = {'INT_AND', 'INT_OR', 'INT_XOR', 'INT_NEGATE', 'INT_LEFT', 'INT_RIGHT'}
    
    def __init__(self):
        self.current_address = 0
        
    def parse_instructions(self, features: List[Dict]) -> List[Dict]:
    class PCodeParser:
        
        for i, feat in enumerate(features):
            if isinstance(feat, dict):
                instr = self._parse_instruction(feat, i)
            else:
                instr = self._parse_raw(feat, i)
            instructions.append(instr)
            
        return instructions
    
    def _parse_instruction(self, feat: Dict, idx: int) -> Dict:
        """Parse a dictionary-format instruction."""
        mnemonic = feat.get('mnemonic', 'NOP')
        
        return {
            'index': idx,
            'address': address,
            'mnemonic': mnemonic,
            'inputs': feat.get('inputs', []),
            'output': feat.get('output'),
            'is_branch': mnemonic in self.BRANCH_OPS,
            'is_memory': mnemonic in self.MEMORY_OPS,
            'is_arithmetic': mnemonic in self.ARITHMETIC_OPS,
            'is_comparison': mnemonic in self.COMPARE_OPS,
            'category': self._get_category(mnemonic)
    
    def _parse_raw(self, raw: Any, idx: int) -> Dict:
        """Parse raw instruction string."""
        if isinstance(raw, str):
            parts = raw.strip().split()
            mnemonic = parts[0] if parts else 'NOP'
        else:
            mnemonic = 'NOP'
            
        return {
            'index': idx,
            'address': idx * 4,
            'mnemonic': mnemonic,
            'inputs': [],
            'output': None,
            'is_branch': False,
            'is_memory': False,
            'is_comparison': False,
            'category': 'unknown'
        }
    
    def _get_category(self, mnemonic: str) -> str:
        """Get instruction category."""
        if mnemonic in self.BRANCH_OPS:
            return 'branch'
        elif mnemonic in self.MEMORY_OPS:
            return 'memory'
        elif mnemonic in self.ARITHMETIC_OPS:
            return 'arithmetic'
        elif mnemonic in self.COMPARE_OPS:
            return 'comparison'
        elif mnemonic in self.BITWISE_OPS:
            return 'bitwise'
        return 'other'


class CFGBuilder:
    
    def __init__(self):
        self.blocks: Dict[int, BasicBlock] = {}
        self.graph = nx.DiGraph()
        self.entry_block_id = 0
        self.exit_block_id = -1
        
    def build_cfg(self, instructions: List[Dict]) -> Tuple[Dict[int, BasicBlock], nx.DiGraph]:
        """Build CFG from instructions."""
        if not instructions:
            return {}, nx.DiGraph()
            
        # Step 1: Identify basic block leaders
    class CFGBuilder:
        # Step 2: Create basic blocks
        self._create_blocks(instructions, leaders)
        
        # Step 3: Connect blocks with edges
        self._connect_blocks(instructions)
        
        # Step 4: Compute dominators
        
        # Step 5: Identify loop headers
        self._identify_loops()
        
        return self.blocks, self.graph
    
    def _find_leaders(self, instructions: List[Dict]) -> Set[int]:
        """Find basic block leaders (first instruction of each block)."""
        leaders = {0}  # First instruction is always a leader
        
        for i, instr in enumerate(instructions):
            if instr['is_branch']:
                # Instruction after branch is a leader
                if i + 1 < len(instructions):
                    leaders.add(i + 1)
                    
                # Branch target is a leader (if we can determine it)
                target = self._get_branch_target(instr, instructions)
                if target is not None:
                    leaders.add(target)
                    
    
    def _get_branch_target(self, instr: Dict, instructions: List[Dict]) -> Optional[int]:
        """Get branch target instruction index."""
        inputs = instr.get('inputs', [])
        if inputs and isinstance(inputs[0], dict):
            target_addr = inputs[0].get('value')
            if target_addr is not None:
                # Find instruction with matching address
                for i, inst in enumerate(instructions):
                    if inst['address'] == target_addr:
                        return i
        return None
    
    def _create_blocks(self, instructions: List[Dict], leaders: Set[int]):
        """Create basic blocks from leaders."""
        sorted_leaders = sorted(leaders)
        for i, leader_idx in enumerate(sorted_leaders):
            # Find block end
            if i + 1 < len(sorted_leaders):
                end_idx = sorted_leaders[i + 1] - 1
            else:
                end_idx = len(instructions) - 1
                
            # Create block
            block_instrs = instructions[leader_idx:end_idx + 1]
            
            block = BasicBlock(
                instructions=block_instrs,
                entry_address=block_instrs[0]['address'] if block_instrs else leader_idx * 4,
                exit_address=block_instrs[-1]['address'] if block_instrs else leader_idx * 4,
                is_conditional=any(instr['mnemonic'] == 'CBRANCH' for instr in block_instrs)
            )
            
            self.blocks[i] = block
            self.graph.add_node(i, block=block)
            
        # Track entry and exit
        self.entry_block_id = 0
        self.exit_block_id = len(self.blocks) - 1
    
    def _connect_blocks(self, instructions: List[Dict]):
        """Connect basic blocks with edges."""
        for block_id, block in self.blocks.items():
            if not block.instructions:
                continue
                
            last_instr = block.instructions[-1]
            
            if last_instr['is_branch']:
                if last_instr['mnemonic'] == 'RETURN':
                    # Return - no successor
                    pass
                elif last_instr['mnemonic'] == 'CBRANCH':
                    # Conditional branch - two successors
                    # Fall-through successor
                        self._add_edge(block_id, block_id + 1)
                    # Branch target
                    target = self._find_target_block(last_instr)
                    if target is not None:
                        self._add_edge(block_id, target)
                elif last_instr['mnemonic'] == 'BRANCH':
                    # Unconditional branch
                    target = self._find_target_block(last_instr)
                    if target is not None:
                        self._add_edge(block_id, target)
                else:
                    # Call - falls through
                    if block_id + 1 in self.blocks:
                        self._add_edge(block_id, block_id + 1)
            else:
                # Non-branch - falls through
                if block_id + 1 in self.blocks:
                    self._add_edge(block_id, block_id + 1)
    
    def _add_edge(self, from_id: int, to_id: int):
        """Add edge between blocks."""
        if from_id in self.blocks and to_id in self.blocks:
            self.blocks[from_id].successors.append(to_id)
            self.blocks[to_id].predecessors.append(from_id)
            self.graph.add_edge(from_id, to_id)
    
    def _find_target_block(self, instr: Dict) -> Optional[int]:
        """Find block containing branch target."""
        inputs = instr.get('inputs', [])
        if inputs and isinstance(inputs[0], dict):
            target_addr = inputs[0].get('value')
            if target_addr is not None:
                for block_id, block in self.blocks.items():
                    if block.entry_address <= target_addr <= block.exit_address:
        return None
    
    def _compute_dominators(self):
        """Compute dominators for each block using iterative algorithm."""
        if not self.blocks:
            return
        # Initialize: entry dominates only itself, others dominated by all
        all_blocks = set(self.blocks.keys())
        
        for block_id in self.blocks:
            if block_id == self.entry_block_id:
                self.blocks[block_id].dominators = {self.entry_block_id}
            else:
                self.blocks[block_id].dominators = all_blocks.copy()
        
        # Iterate until no change
        while changed:
            changed = False
            for block_id in self.blocks:
                if block_id == self.entry_block_id:
                    continue
                    
                # Dom(n) = {n} ∪ (∩ Dom(pred) for all pred)
                preds = self.blocks[block_id].predecessors
                if preds:
                    new_doms = all_blocks.copy()
                    for pred in preds:
                        new_doms &= self.blocks[pred].dominators
                    new_doms.add(block_id)
                    
                    if new_doms != self.blocks[block_id].dominators:
                        self.blocks[block_id].dominators = new_doms
                        changed = True
    
    def _identify_loops(self):
        """Identify natural loops using back edges."""
        for block_id, block in self.blocks.items():
            for succ_id in block.successors:
                # Back edge: successor dominates current block
                if succ_id in block.dominators:
                    # succ_id is a loop header
                    self.blocks[succ_id].is_loop_header = True


class DataFlowAnalyzer:
    """Perform data flow analysis on CFG."""
    
    def __init__(self, blocks: Dict[int, BasicBlock], graph: nx.DiGraph):
        self.blocks = blocks
        self.graph = graph
        self.variables: Dict[str, Variable] = {}
        self.live_vars: Dict[int, Set[str]] = {}  # block -> live variables
        
    def analyze(self) -> Dict:
        """Perform complete data flow analysis."""
        # Extract variables from instructions
        self._extract_variables()
        
        # Reaching definitions analysis
    class DataFlowAnalyzer:
        # Live variable analysis
        self._compute_live_variables()
        
        # Build def-use chains
        def_use_chains = self._build_def_use_chains()
        
        return {
            'variables': {name: vars(v) for name, v in self.variables.items()},
            'live_variables': {k: list(v) for k, v in self.live_vars.items()},
            'def_use_chains': def_use_chains
        }
    
    def _extract_variables(self):
        """Extract variables from all instructions."""
        for block_id, block in self.blocks.items():
            for instr in block.instructions:
                # Output is a definition
                output = instr.get('output')
                if output:
                    var_name = self._get_var_name(output)
                    if var_name:
                        if var_name not in self.variables:
                            self.variables[var_name] = Variable(name=var_name)
                        self.variables[var_name].definitions.append(block_id)
                        
                        # Infer type from operation
                        self._infer_type(var_name, instr)
                
                for inp in instr.get('inputs', []):
                    var_name = self._get_var_name(inp)
                    if var_name:
                        if var_name not in self.variables:
                            self.variables[var_name] = Variable(name=var_name)
                        self.variables[var_name].uses.append(block_id)
    
    def _get_var_name(self, operand: Any) -> Optional[str]:
        """Extract variable name from operand."""
        if isinstance(operand, dict):
            return operand.get('name') or operand.get('register')
        elif isinstance(operand, str):
            return operand
        return None
    
    def _infer_type(self, var_name: str, instr: Dict):
        """Infer variable type from instruction context."""
        mnemonic = instr['mnemonic']
        var = self.variables[var_name]
        
        if mnemonic.startswith('FLOAT'):
            var.inferred_type = 'float' if var.inferred_type == 'unknown' else var.inferred_type
        elif mnemonic.startswith('INT'):
                var.inferred_type = 'int'
        elif mnemonic in ('LOAD', 'STORE'):
            var.is_pointer = True
            if var.inferred_type == 'unknown':
                var.inferred_type = 'pointer'
        elif mnemonic == 'INDIRECT':
            var.is_pointer = True
    def _compute_reaching_definitions(self):
        """Compute reaching definitions using iterative dataflow."""
        # Initialize
        for block_id in self.blocks:
            self.reaching_defs[block_id] = set()
        
        # Generate and kill sets
        gen_sets = {}
        kill_sets = {}
        
        for block_id, block in self.blocks.items():
            gen = set()
            kill = set()
            
            for var_name, var in self.variables.items():
                if block_id in var.definitions:
                    # Kill other definitions of same variable
                    for other_block in var.definitions:
                        if other_block != block_id:
                            kill.add((var_name, other_block))
            
            gen_sets[block_id] = gen
            kill_sets[block_id] = kill
        
        # Iterate until fixpoint
        changed = True
        while changed:
            changed = False
            for block_id in self.blocks:
                # IN[B] = Union of OUT[P] for all predecessors P
                in_set = set()
                for pred in self.blocks[block_id].predecessors:
                    in_set |= self.reaching_defs[pred]
                
                # OUT[B] = GEN[B] ∪ (IN[B] - KILL[B])
                out_set = gen_sets[block_id] | (in_set - kill_sets[block_id])
                
                if out_set != self.reaching_defs[block_id]:
                    self.reaching_defs[block_id] = out_set
                    changed = True
    
    def _compute_live_variables(self):
        """Compute live variables using backward dataflow."""
        # Initialize
        for block_id in self.blocks:
            self.live_vars[block_id] = set()
        
        # Use and def sets
        use_sets = {}
        def_sets = {}
        
        for block_id, block in self.blocks.items():
            uses = set()
            defs = set()
            
            for var_name, var in self.variables.items():
                if block_id in var.uses:
                    uses.add(var_name)
                    defs.add(var_name)
            
            use_sets[block_id] = uses
            def_sets[block_id] = defs
        
        # Backward iteration
        changed = True
        while changed:
            changed = False
            for block_id in reversed(list(self.blocks.keys())):
                # OUT[B] = Union of IN[S] for all successors S
                out_set = set()
                for succ in self.blocks[block_id].successors:
                    out_set |= self.live_vars[succ]
                
                # IN[B] = USE[B] ∪ (OUT[B] - DEF[B])
                in_set = use_sets[block_id] | (out_set - def_sets[block_id])
                
                if in_set != self.live_vars[block_id]:
                    self.live_vars[block_id] = in_set
                    changed = True
    
    def _build_def_use_chains(self) -> Dict[str, List[Dict]]:
        """Build def-use chains for each variable."""
        chains = {}
        
        for var_name, var in self.variables.items():
            chains[var_name] = []
            
            for def_block in var.definitions:
                for use_block in var.uses:
                    # Check if definition reaches use
                    if (var_name, def_block) in self.reaching_defs.get(use_block, set()):
                        chains[var_name].append({
                            'definition': def_block,
                            'use': use_block
                        })
        
        return chains

class TypeInferenceEngine:
    """Advanced type inference using constraint propagation."""
    
    def __init__(self, data_flow: Dict):
        self.data_flow = data_flow
        self.type_constraints: Dict[str, Set[str]] = defaultdict(set)
        self.final_types: Dict[str, str] = {}
        
    def infer_types(self, instructions: List[Dict]) -> Dict[str, str]:
        """Infer types for all variables."""
        # Collect constraints from instructions
        self._collect_constraints(instructions)
        
        # Propagate constraints
        self._propagate_constraints()
        
        # Resolve to final types
    class TypeInferenceEngine:
        return self.final_types
    
    def _collect_constraints(self, instructions: List[Dict]):
        """Collect type constraints from instructions."""
        for instr in instructions:
            mnemonic = instr['mnemonic']
            inputs = instr.get('inputs', [])
            
            if output:
                out_name = self._get_name(output)
                if out_name:
                    # Arithmetic constraints
                    if mnemonic.startswith('INT'):
                        self.type_constraints[out_name].add('integer')
                    elif mnemonic.startswith('FLOAT'):
                        self.type_constraints[out_name].add('float')
                    
                    # Pointer constraints
                        self.type_constraints[out_name].add('pointer')
                    
                    # Boolean constraints
                    if mnemonic in ('INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS'):
                        self.type_constraints[out_name].add('boolean')
                    
                    # Propagate from inputs
                    for inp in inputs:
                        inp_name = self._get_name(inp)
                        if inp_name and inp_name in self.type_constraints:
                            # Same operation suggests compatible types
                            pass
    
    def _get_name(self, operand: Any) -> Optional[str]:
        if isinstance(operand, dict):
            return operand.get('name') or operand.get('register')
        elif isinstance(operand, str):
            return operand
        return None
    
    def _propagate_constraints(self):
        """Propagate type constraints through def-use chains."""
        chains = self.data_flow.get('def_use_chains', {})
        
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            iterations += 1
            
            for var_name, chain in chains.items():
                if var_name not in self.type_constraints:
                    continue
                    
                current_constraints = self.type_constraints[var_name].copy()
                # Propagate to uses
                for link in chain:
                    # If definition has constraints, they flow to uses
                    pass
    
    def _resolve_types(self):
        """Resolve constraints to concrete C types."""
        type_priority = ['pointer', 'float', 'integer', 'boolean']
        
        for var_name, constraints in self.type_constraints.items():
            if not constraints:
                self.final_types[var_name] = 'int'  # Default
                continue
            
            # Priority-based resolution
            for type_name in type_priority:
                if type_name in constraints:
                    self.final_types[var_name] = self._to_c_type(type_name)
                    break
            else:
                self.final_types[var_name] = 'int'
        
        for var_name in self.data_flow.get('variables', {}).keys():
            if var_name not in self.final_types:
                var_info = self.data_flow['variables'][var_name]
                if var_info.get('is_pointer'):
                    self.final_types[var_name] = 'void*'
                elif var_info.get('inferred_type') != 'unknown':
                    self.final_types[var_name] = var_info['inferred_type']
                else:
                    self.final_types[var_name] = 'int'
    
    def _to_c_type(self, abstract_type: str) -> str:
        """Convert abstract type to C type."""
        type_map = {
            'integer': 'int',
            'float': 'double',
            'boolean': 'int',
            'pointer': 'void*',
            'character': 'char'
        }
        return type_map.get(abstract_type, 'int')


class PatternMatcher:
    """Identify common code patterns and idioms."""
    
    # Common patterns in decompiled code
    PATTERNS = {
            'signature': ['INT_ADD', 'LOAD', 'CBRANCH'],
            'template': '''for (int i = 0; i < {bound}; i++) {{
    {accumulator} += {array}[i];
}}'''
        },
        'memcpy': {
            'signature': ['LOAD', 'STORE', 'INT_ADD', 'CBRANCH'],
            'template': '''memcpy({dest}, {src}, {size});'''
        },
        'strcmp_loop': {
    class PatternMatcher:
        },
        'null_check': {
            'signature': ['INT_EQUAL', 'CBRANCH'],
            'template': '''if ({ptr} == NULL) {{ return {error_code}; }}'''
        },
        'linked_list_traversal': {
            'signature': ['LOAD', 'LOAD', 'INT_NOTEQUAL', 'CBRANCH'],
            'template': '''while ({current} != NULL) {{
    {current} = {current}->{next};
}}'''
        }
    }
    
    def __init__(self):
        pass
        
    def match_patterns(self, blocks: Dict[int, BasicBlock]) -> List[Dict]:
        """Find pattern matches in the CFG."""
        matches = []
        
        for block_id, block in blocks.items():
            if not block.instructions:
                continue
                
            # Extract instruction sequence
            mnemonics = [i['mnemonic'] for i in block.instructions]
            
            for pattern_name, pattern in self.PATTERNS.items():
                signature = pattern['signature']
                
                # Check if pattern signature matches
                if self._matches_signature(mnemonics, signature):
                        'pattern': pattern_name,
                        'block_id': block_id,
                        'template': pattern['template'],
                        'confidence': 0.8
                    })
        
        # Also check multi-block patterns (loops)
        loop_matches = self._find_loop_patterns(blocks)
        matches.extend(loop_matches)
        
        return matches
    
    def _matches_signature(self, mnemonics: List[str], signature: List[str]) -> bool:
        """Check if instruction sequence matches pattern signature."""
        if len(mnemonics) < len(signature):
            return False
            
        # Allow some flexibility in matching
        sig_idx = 0
        for mnem in mnemonics:
            if sig_idx < len(signature) and mnem == signature[sig_idx]:
                sig_idx += 1
            if sig_idx == len(signature):
                return True
                
        return sig_idx == len(signature)
    
    def _find_loop_patterns(self, blocks: Dict[int, BasicBlock]) -> List[Dict]:
        matches = []
        
        for block_id, block in blocks.items():
            if block.is_loop_header:
                # Analyze loop body
                loop_type = self._classify_loop(block, blocks)
                if loop_type:
                    matches.append({
                        'pattern': loop_type,
                        'block_id': block_id,
                        'template': self._get_loop_template(loop_type),
                        'confidence': 0.75
                    })
        
    
    def _classify_loop(self, header: BasicBlock, blocks: Dict[int, BasicBlock]) -> Optional[str]:
        """Classify loop type based on structure."""
        # Check for counter variable
        has_counter = any(
            'INT_ADD' in i['mnemonic'] or 'INT_SUB' in i['mnemonic'] 
            for i in header.instructions
        )
        
        # Check for comparison
        has_compare = any(i['is_comparison'] for i in header.instructions)
        
        if has_counter and has_compare:
            return 'for_loop'
        elif has_compare:
            return 'while_loop'
        else:
    
    def _get_loop_template(self, loop_type: str) -> str:
        """Get template for loop type."""
        templates = {
            'for_loop': 'for (int i = 0; i < n; i++) { /* body */ }',
            'while_loop': 'while (condition) { /* body */ }',
            'do_while_loop': 'do { /* body */ } while (condition);'
        }
        return templates.get(loop_type, '/* loop */')


def analyze_code(features: List[Dict]) -> Dict:
    """
    Complete code analysis pipeline.
    
    Args:
        features: List of P-Code instructions/features
    Returns:
        Complete analysis results including CFG, DFA, types, patterns
    """
    # Parse instructions
    parser = PCodeParser()
    instructions = parser.parse_instructions(features)
    
    # Build CFG
    def analyze_code(features: List[Dict]) -> Dict:
    
    # Type inference
    type_engine = TypeInferenceEngine(data_flow)
    types = type_engine.infer_types(instructions)
    
    # Pattern matching
    pattern_matcher = PatternMatcher()
    patterns = pattern_matcher.match_patterns(blocks)
    
    # Compile results
    return {
        'instructions': instructions,
        'cfg': {
            'num_blocks': len(blocks),
            'blocks': [
                {
                    'id': b.id,
                    'entry': b.entry_address,
                    'exit': b.exit_address,
                    'num_instructions': len(b.instructions),
                    'is_loop_header': b.is_loop_header,
                    'is_conditional': b.is_conditional,
                    'predecessors': b.predecessors,
                    'successors': b.successors
                }
                for b in blocks.values()
            ],
            'entry_block': cfg_builder.entry_block_id,
            'exit_block': cfg_builder.exit_block_id
        },
        'data_flow': data_flow,
        'types': types,
        'patterns': patterns,
        'statistics': {
            'num_instructions': len(instructions),
            'num_blocks': len(blocks),
            'num_variables': len(data_flow['variables']),
            'num_patterns': len(patterns)
        }
    }
