

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
from concurrent.futures import ThreadPoolExecutor
import functools
import hashlib

# Import real analysis module
from analysis import (
    analyze_code, CFGBuilder, DataFlowAnalyzer, 
    TypeInferenceEngine, PatternMatcher, PCodeParser
)

app = Flask(__name__)


# ============================================================================
# Neural Models for Agents
# ============================================================================

class AgentModel(nn.Module):
    # ...existing code...
    
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256, 
                 num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_head = nn.Linear(hidden_dim, hidden_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Token indices [batch, seq_len]
            
        Returns:
            features: Hidden representations [batch, seq_len, hidden]
            confidence: Confidence score [batch, 1]
        """
        # Embedding + positional encoding
        emb = self.embedding(x)
        seq_len = x.size(1)
        emb = emb + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        features = self.transformer(emb)
        
        # Pool and predict confidence
        pooled = features.mean(dim=1)
        output_features = self.output_head(pooled)
        confidence = torch.sigmoid(self.confidence_head(pooled))
        
        return features, confidence


# ============================================================================
# Agent Base Class with Real Analysis
# ============================================================================

@dataclass
class AgentProposal:
    agent_name: str
    specialty: str
    code: str
    confidence: float
    reasoning: str
    analysis_data: Dict
    critique_history: List[str] = None
    
    def __post_init__(self):
        if self.critique_history is None:
            self.critique_history = []


class BaseAgent(ABC):
    # ...existing code...
    
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural model
        self.model = AgentModel().to(self.device)
        self.model.eval()
        
        # Analysis cache
        self._cache = {}
    
    @abstractmethod
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        pass
    
    def _get_confidence(self, analysis: Dict) -> float:
        # ...existing code...
        # Base confidence from analysis completeness
        has_cfg = analysis.get('cfg', {}).get('num_blocks', 0) > 0
        has_types = len(analysis.get('types', {})) > 0
        has_patterns = len(analysis.get('patterns', [])) > 0
        
        base_confidence = 0.5
        if has_cfg:
            base_confidence += 0.15
        if has_types:
            base_confidence += 0.15
        if has_patterns:
            base_confidence += 0.1
            
        return min(base_confidence, 0.95)
    
    def critique(self, proposal: AgentProposal) -> Dict:
        # ...existing code...


class StructureAgent(BaseAgent):
    # ...existing code...
    
    def __init__(self):
        super().__init__("StructureExpert", "control_flow")
        
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        # ...existing code...
        # Get or compute analysis
        full_analysis = context.get('full_analysis')
        if full_analysis is None:
            full_analysis = analyze_code(features)
        
        cfg = full_analysis.get('cfg', {})
        blocks = cfg.get('blocks', [])
        
        reasoning = []
        code_parts = []
        
        # Analyze loop structures
        loop_headers = [b for b in blocks if b.get('is_loop_header')]
        if loop_headers:
            reasoning.append(f"Detected {len(loop_headers)} loop(s)")
            
        # Analyze conditionals
        conditionals = [b for b in blocks if b.get('is_conditional')]
        if conditionals:
            reasoning.append(f"Found {len(conditionals)} conditional branch(es)")
            
        # Analyze complexity
        num_blocks = len(blocks)
        if num_blocks > 10:
            reasoning.append(f"Complex function with {num_blocks} basic blocks")
        elif num_blocks > 5:
            reasoning.append(f"Moderate complexity: {num_blocks} blocks")
        else:
            reasoning.append(f"Simple function: {num_blocks} blocks")
        
        # Generate structured code based on CFG
        code = self._generate_structure_code(blocks, loop_headers, conditionals, full_analysis)
        
        confidence = self._get_confidence(full_analysis)
        if loop_headers:
            confidence += 0.05
        if conditionals:
            confidence += 0.05
            
        return AgentProposal(
            agent_name=self.name,
            specialty=self.specialty,
            code=code,
            confidence=min(confidence, 0.95),
            reasoning="; ".join(reasoning),
            analysis_data={'cfg': cfg}
        )
    
    def _generate_structure_code(self, blocks: List[Dict], loops: List[Dict], 
                                  conditionals: List[Dict], analysis: Dict) -> str:
        # ...existing code...
        patterns = analysis.get('patterns', [])
        types = analysis.get('types', {})
        
        code_lines = []
        code_lines.append("void function() {")
        
        # Declare inferred variables
        for var_name, var_type in types.items():
            if not var_name.startswith('temp'):  # Skip temporaries
                code_lines.append(f"    {var_type} {var_name};")
        
        if types:
            code_lines.append("")
        
        # Generate control flow
        if loops:
            for i, loop in enumerate(loops):
                # Check for matching pattern
                loop_pattern = next(
                    (p for p in patterns if p.get('pattern', '').endswith('_loop')), 
                    None
                )
                if loop_pattern:
                    code_lines.append(f"    // Loop {i+1}")
                    code_lines.append(f"    {loop_pattern.get('template', 'for (...) {}')}")
                else:
                    code_lines.append(f"    for (int i = 0; i < n; i++) {{")
                    code_lines.append(f"        // Block {loop.get('id', i)}")
                    code_lines.append(f"    }}")
        
        if conditionals:
            for i, cond in enumerate(conditionals):
                code_lines.append(f"    if (condition_{i}) {{")
                code_lines.append(f"        // Then branch (block {cond.get('id', i)})")
                code_lines.append(f"    }} else {{")
                code_lines.append(f"        // Else branch")
                code_lines.append(f"    }}")
        
        if not loops and not conditionals:
            code_lines.append("    // Sequential code")
            for i, block in enumerate(blocks[:3]):  # Show first 3 blocks
                code_lines.append(f"    // Block {block.get('id', i)}: {block.get('num_instructions', 0)} instructions")
        
        code_lines.append("}")
        
        return "\n".join(code_lines)
    
    def critique(self, proposal: AgentProposal) -> Dict:
        # ...existing code...
        code = proposal.code
        
        # Check for structural issues
        if code.count('{') != code.count('}'):
            return {
                'message': 'Unbalanced braces detected',
                'severity': 0.8,
                'suggestion': 'Review brace matching in control structures'
            }
        
        # Check for infinite loop risk
        if 'while' in code and 'break' not in code and 'return' not in code:
            return {
                'message': 'Potential infinite loop without exit',
                'severity': 0.6,
                'suggestion': 'Add break or return condition'
            }
            
        return {
            'message': 'Control flow structure looks valid',
            'severity': 0.1,
            'suggestion': 'No structural issues detected'
        }


class DataFlowAgent(BaseAgent):
    # ...existing code...
    
    def __init__(self):
        super().__init__("DataFlowExpert", "data_flow")
        
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        # ...existing code...
        full_analysis = context.get('full_analysis')
        if full_analysis is None:
            full_analysis = analyze_code(features)
        
        data_flow = full_analysis.get('data_flow', {})
        variables = data_flow.get('variables', {})
        def_use_chains = data_flow.get('def_use_chains', {})
        live_vars = data_flow.get('live_variables', {})
        
        reasoning = []
        
        # Analyze variable patterns
        if variables:
            num_vars = len(variables)
            reasoning.append(f"Identified {num_vars} variable(s)")
            
            # Find accumulators (multiple definitions in same block)
            accumulators = []
            for var_name, var_info in variables.items():
                defs = var_info.get('definitions', [])
                uses = var_info.get('uses', [])
                if len(defs) > 1 and len(uses) > len(defs):
                    accumulators.append(var_name)
            
            if accumulators:
                reasoning.append(f"Accumulator pattern: {', '.join(accumulators[:3])}")
        
        # Analyze def-use chains for dependencies
        if def_use_chains:
            num_chains = sum(len(c) for c in def_use_chains.values())
            reasoning.append(f"Found {num_chains} def-use chain(s)")
        
        # Check for dead variables
        dead_vars = []
        for var_name, var_info in variables.items():
            if not var_info.get('uses'):
                dead_vars.append(var_name)
        if dead_vars:
            reasoning.append(f"Warning: {len(dead_vars)} unused variable(s)")
        
        # Generate data-flow aware code
        code = self._generate_dataflow_code(variables, def_use_chains, full_analysis)
        
        confidence = self._get_confidence(full_analysis)
        if variables:
            confidence += 0.05
        if not dead_vars:
            confidence += 0.05
            
        return AgentProposal(
            agent_name=self.name,
            specialty=self.specialty,
            code=code,
            confidence=min(confidence, 0.95),
            reasoning="; ".join(reasoning),
            analysis_data={'data_flow': data_flow}
        )
    
    def _generate_dataflow_code(self, variables: Dict, def_use_chains: Dict, 
                                 analysis: Dict) -> str:
        types = analysis.get('types', {})
        patterns = analysis.get('patterns', [])
        
        code_lines = []
        code_lines.append("void function() {")
        
        # Variable declarations with inferred types
        var_decls = []
        for var_name, var_info in variables.items():
            var_type = types.get(var_name, 'int')
            is_pointer = var_info.get('is_pointer', False)
            
            if is_pointer and not var_type.endswith('*'):
                var_type += '*'
                
            var_decls.append(f"    {var_type} {var_name};")
        
        # Sort declarations by type for readability
        code_lines.extend(sorted(set(var_decls)))
        
        if var_decls:
            code_lines.append("")
        
        # Check for accumulator pattern
        accumulator_pattern = next(
            (p for p in patterns if p.get('pattern') == 'accumulator_loop'), 
            None
        )
        
        if accumulator_pattern:
            # Accumulator pattern
            code_lines.append("    int sum = 0;")
            code_lines.append("    for (int i = 0; i < n; i++) {")
            code_lines.append("        sum += array[i];")
            code_lines.append("    }")
            code_lines.append("    return sum;")
        else:
            # Data flow analysis
            for var_name, chains in list(def_use_chains.items())[:5]:
                if chains:
                    code_lines.append(f"    // {var_name}: defined in block {chains[0].get('definition', '?')}, used in block {chains[0].get('use', '?')}")
        
        code_lines.append("}")
        
        return "\n".join(code_lines)
    
    def critique(self, proposal: AgentProposal) -> Dict:
        code = proposal.code
        
        # Check for uninitialized variables
        decl_pattern = r'\b(int|char|float|double)\s+(\w+)\s*;'
        init_pattern = r'\b(int|char|float|double)\s+(\w+)\s*='
        
        import re
        decls = set(re.findall(decl_pattern, code))
        inits = set(re.findall(init_pattern, code))
        
        uninitialized = decls - inits
        if uninitialized:
            var_names = [v[1] for v in list(uninitialized)[:3]]
            return {
                'message': f'Potentially uninitialized: {", ".join(var_names)}',
                'severity': 0.6,
                'suggestion': 'Initialize variables at declaration'
            }
        
        # Check for unused variables (defined but never used)
        if 'malloc' in code and 'free' not in code:
            return {
                'message': 'Memory allocated without free',
                'severity': 0.7,
                'suggestion': 'Add free() call to prevent memory leak'
            }
            
        return {
            'message': 'Data flow looks valid',
            'severity': 0.1,
            'suggestion': 'No data flow issues detected'
        }


class TypeAgent(BaseAgent):
    def __init__(self):
        super().__init__("TypeExpert", "type_inference")
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        full_analysis = context.get('full_analysis')
        if full_analysis is None:
            full_analysis = analyze_code(features)
        types = full_analysis.get('types', {})
        variables = full_analysis.get('data_flow', {}).get('variables', {})
        reasoning = []
        type_counts = {}
        for var_name, var_type in types.items():
            type_counts[var_type] = type_counts.get(var_type, 0) + 1
        if type_counts:
            reasoning.append(f"Type distribution: {type_counts}")
        pointer_vars = [v for v, t in types.items() if '*' in t or t == 'pointer']
        if pointer_vars:
            reasoning.append(f"Pointer variables: {len(pointer_vars)}")
        float_vars = [v for v, t in types.items() if 'float' in t or 'double' in t]
        if float_vars:
            reasoning.append(f"Floating-point computation detected")
        code = self._generate_typed_code(types, variables, full_analysis)
        confidence = self._get_confidence(full_analysis)
        if types:
            confidence += 0.1
        return AgentProposal(
            agent_name=self.name,
            specialty=self.specialty,
            code=code,
            confidence=min(confidence, 0.95),
            reasoning="; ".join(reasoning),
            analysis_data={'types': types}
        )
    def _generate_typed_code(self, types: Dict, variables: Dict, analysis: Dict) -> str:
        patterns = analysis.get('patterns', [])
        code_lines = []
        code_lines.append("void function() {")
        by_type = {}
        for var_name, var_type in types.items():
            if var_type not in by_type:
                by_type[var_type] = []
            by_type[var_type].append(var_name)
        for var_type, var_names in sorted(by_type.items()):
            for var_name in var_names:
                if var_type == 'int':
                    code_lines.append(f"    {var_type} {var_name} = 0;")
                elif '*' in var_type:
                    code_lines.append(f"    {var_type} {var_name} = NULL;")
                elif var_type in ('float', 'double'):
                    code_lines.append(f"    {var_type} {var_name} = 0.0;")
                else:
                    code_lines.append(f"    {var_type} {var_name};")
        
        if not types:
            code_lines.append("    // No type information available")
        
        # Add type-specific patterns
        if any(t in ('float', 'double') for t in types.values()):
            code_lines.append("")
            code_lines.append("    // Floating-point computation")
            code_lines.append("    double result = 0.0;")
            code_lines.append("    for (int i = 0; i < n; i++) {")
            code_lines.append("        result += values[i] * weights[i];")
            code_lines.append("    }")
        
        code_lines.append("}")
        
        return "\n".join(code_lines)
    
    def critique(self, proposal: AgentProposal) -> Dict:
        """Critique from type perspective."""
        code = proposal.code
        
        # Check for implicit type conversions
        if 'int' in code and 'float' in code:
            if '(float)' not in code and '(int)' not in code:
                return {
                    'message': 'Possible implicit type conversion int<->float',
                    'severity': 0.4,
                    'suggestion': 'Add explicit type casts for clarity'
                }
        
        # Check for void* without cast
        if 'void*' in code and 'malloc' in code:
            if '(int*)' not in code and '(char*)' not in code:
                return {
                    'message': 'malloc returns void* without cast',
                    'severity': 0.3,
                    'suggestion': 'Add explicit cast after malloc'
                }
                
        return {
            'message': 'Type usage looks correct',
            'severity': 0.1,
            'suggestion': 'No type issues detected'
        }


class MemoryAgent(BaseAgent):
    """
    Agent specialized in memory access patterns and pointer arithmetic.
    """
    
    def __init__(self):
        super().__init__("MemoryExpert", "memory_access")
        
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        """Analyze memory access patterns."""
        full_analysis = context.get('full_analysis')
        if full_analysis is None:
            full_analysis = analyze_code(features)
        
        instructions = full_analysis.get('instructions', [])
        types = full_analysis.get('types', {})
        patterns = full_analysis.get('patterns', [])
        
        reasoning = []
        
        # Count memory operations
        memory_ops = [i for i in instructions if i.get('is_memory')]
        if memory_ops:
            loads = sum(1 for i in memory_ops if i['mnemonic'] == 'LOAD')
            stores = sum(1 for i in memory_ops if i['mnemonic'] == 'STORE')
            reasoning.append(f"Memory operations: {loads} loads, {stores} stores")
        
        # Identify pointer variables
        pointer_vars = [v for v, t in types.items() if 'pointer' in t or '*' in t]
        if pointer_vars:
            reasoning.append(f"Pointer variables: {', '.join(pointer_vars[:5])}")
        
        # Check for memcpy pattern
        memcpy_pattern = next((p for p in patterns if p.get('pattern') == 'memcpy'), None)
        if memcpy_pattern:
            reasoning.append("Detected memory copy pattern")
        
        # Check for array access pattern
        array_access = next((p for p in patterns if 'array' in p.get('pattern', '')), None)
        if array_access:
            reasoning.append("Detected array access pattern")
        
        # Generate memory-aware code
        code = self._generate_memory_code(memory_ops, pointer_vars, patterns, full_analysis)
        
        confidence = self._get_confidence(full_analysis)
        if memory_ops:
            confidence += 0.05
        if pointer_vars:
            confidence += 0.05
            
        return AgentProposal(
            agent_name=self.name,
            specialty=self.specialty,
            code=code,
            confidence=min(confidence, 0.95),
            reasoning="; ".join(reasoning),
            analysis_data={'memory_ops': len(memory_ops), 'pointer_vars': pointer_vars}
        )
    
    def _generate_memory_code(self, memory_ops: List[Dict], pointer_vars: List[str],
                               patterns: List[Dict], analysis: Dict) -> str:
        """Generate code with proper memory handling."""
        code_lines = []
        code_lines.append("// Memory-aware decompilation")
        code_lines.append("void function() {")
        
        # Declare pointer variables
        for ptr_var in pointer_vars[:5]:
            code_lines.append(f"    void* {ptr_var} = NULL;")
        
        if pointer_vars:
            code_lines.append("")
        
        # Check for patterns
        memcpy_pattern = next((p for p in patterns if p.get('pattern') == 'memcpy'), None)
        linked_list = next((p for p in patterns if 'linked_list' in p.get('pattern', '')), None)
        
        if memcpy_pattern:
            code_lines.append("    // Memory copy operation")
            code_lines.append("    memcpy(dest, src, size);")
        elif linked_list:
            code_lines.append("    // Linked list traversal")
            code_lines.append("    Node* current = head;")
            code_lines.append("    while (current != NULL) {")
            code_lines.append("        process(current->data);")
            code_lines.append("        current = current->next;")
            code_lines.append("    }")
        elif memory_ops:
            # Generic memory access
            loads = sum(1 for i in memory_ops if i['mnemonic'] == 'LOAD')
            stores = sum(1 for i in memory_ops if i['mnemonic'] == 'STORE')
            
            if loads > stores:
                code_lines.append("    // Read-heavy memory access")
                code_lines.append("    for (int i = 0; i < count; i++) {")
                code_lines.append("        value = buffer[i];  // Load")
                code_lines.append("        process(value);")
                code_lines.append("    }")
            else:
                code_lines.append("    // Write-heavy memory access")
                code_lines.append("    for (int i = 0; i < count; i++) {")
                code_lines.append("        buffer[i] = compute(i);  // Store")
                code_lines.append("    }")
        else:
            code_lines.append("    // No significant memory operations")
        
        code_lines.append("}")
        
        return "\n".join(code_lines)
    
    def critique(self, proposal: AgentProposal) -> Dict:
        """Critique from memory safety perspective."""
        code = proposal.code
        
        # Check for NULL pointer dereference
        if '*' in code and 'NULL' not in code and 'if' not in code:
            return {
                'message': 'Pointer dereference without NULL check',
                'severity': 0.7,
                'suggestion': 'Add NULL check before dereference'
            }
        
        # Check for array bounds
        if '[' in code and ']' in code:
            if 'i <' not in code and 'i <=' not in code:
                return {
                    'message': 'Array access without bounds check',
                    'severity': 0.6,
                    'suggestion': 'Ensure loop bounds prevent overflow'
                }
        
        # Check for memory leak
        if 'malloc' in code and 'free' not in code:
            return {
                'message': 'Memory allocated but not freed',
                'severity': 0.8,
                'suggestion': 'Add free() call before return'
            }
            
        return {
            'message': 'Memory handling looks safe',
            'severity': 0.1,
            'suggestion': 'No memory safety issues detected'
        }


class OptimizationAgent(BaseAgent):
    """
    Agent specialized in recognizing compiler optimizations and idioms.
    """
    
    def __init__(self):
        super().__init__("OptimizationExpert", "optimizations")
        
        # Common optimization patterns
        self.optimization_signatures = {
            'strength_reduction': {
                'patterns': [('INT_MULT', 'INT_LEFT'), ('INT_DIV', 'INT_RIGHT')],
                'description': 'Multiplication/division replaced by shifts'
            },
            'loop_unrolling': {
                'patterns': [('INT_ADD', 'INT_ADD', 'INT_ADD', 'INT_ADD')],
                'description': 'Loop body duplicated to reduce overhead'
            },
            'inlining': {
                'patterns': [('COPY', 'INT_ADD', 'COPY')],
                'description': 'Function call inlined'
            }
        }
        
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        """Recognize compiler optimizations."""
        full_analysis = context.get('full_analysis')
        if full_analysis is None:
            full_analysis = analyze_code(features)
        
        instructions = full_analysis.get('instructions', [])
        cfg = full_analysis.get('cfg', {})
        
        reasoning = []
        detected_opts = []
        
        # Check for optimization patterns
        mnemonics = [i['mnemonic'] for i in instructions]
        
        for opt_name, opt_info in self.optimization_signatures.items():
            for pattern in opt_info['patterns']:
                if self._check_pattern(mnemonics, pattern):
                    detected_opts.append(opt_name)
                    reasoning.append(opt_info['description'])
                    break
        
        # Check for loop-related optimizations
        blocks = cfg.get('blocks', [])
        loop_headers = [b for b in blocks if b.get('is_loop_header')]
        
        if loop_headers and len(instructions) > 50:
            reasoning.append("Complex loop structure - possible loop optimization")
        
        # Check for bitwise operations (often optimizations)
        bitwise = [i for i in instructions if i.get('category') == 'bitwise']
        if len(bitwise) > 5:
            reasoning.append(f"Heavy bitwise usage ({len(bitwise)} ops) - strength reduction likely")
        
        # Generate optimization-aware code
        code = self._generate_optimized_code(detected_opts, instructions, full_analysis)
        
        confidence = self._get_confidence(full_analysis)
        if detected_opts:
            confidence += 0.1
            
        return AgentProposal(
            agent_name=self.name,
            specialty=self.specialty,
            code=code,
            confidence=min(confidence, 0.95),
            reasoning="; ".join(reasoning) if reasoning else "No specific optimizations detected",
            analysis_data={'optimizations': detected_opts}
        )
    
    def _check_pattern(self, mnemonics: List[str], pattern: Tuple) -> bool:
        """Check if instruction sequence contains pattern."""
        pattern_list = list(pattern)
        
        for i in range(len(mnemonics) - len(pattern_list) + 1):
            if mnemonics[i:i+len(pattern_list)] == pattern_list:
                return True
        return False
    
    def _generate_optimized_code(self, detected_opts: List[str], 
                                  instructions: List[Dict], analysis: Dict) -> str:
        """Generate code explaining optimizations."""
        code_lines = []
        code_lines.append("// Optimization-aware decompilation")
        code_lines.append("void function() {")
        
        if 'strength_reduction' in detected_opts:
            code_lines.append("    // Strength reduction: x * 8 -> x << 3")
            code_lines.append("    int result = x << 3;  // Optimized from x * 8")
            code_lines.append("")
        
        if 'loop_unrolling' in detected_opts:
            code_lines.append("    // Loop unrolling detected")
            code_lines.append("    for (int i = 0; i < n; i += 4) {")
            code_lines.append("        process(array[i]);")
            code_lines.append("        process(array[i+1]);")
            code_lines.append("        process(array[i+2]);")
            code_lines.append("        process(array[i+3]);")
            code_lines.append("    }")
            code_lines.append("")
        
        if 'inlining' in detected_opts:
            code_lines.append("    // Inlined function call")
            code_lines.append("    // Original: result = helper(x);")
            code_lines.append("    int temp = x;")
            code_lines.append("    int result = temp + 1;  // Inlined helper body")
        
        if not detected_opts:
            code_lines.append("    // Standard code - no significant optimizations")
            code_lines.append("    int result = compute(input);")
            code_lines.append("    return result;")
        
        code_lines.append("}")
        
        return "\n".join(code_lines)
    
    def critique(self, proposal: AgentProposal) -> Dict:
        """Critique from optimization perspective."""
        code = proposal.code
        
        # Check for inefficient patterns
        if '* 2' in code or '/ 2' in code:
            return {
                'message': 'Multiplication/division by power of 2 could use shift',
                'severity': 0.3,
                'suggestion': 'Use << 1 or >> 1 for clarity about optimization'
            }
        
        # Check for redundant operations
        if code.count('for') > 2:
            return {
                'message': 'Multiple loops could potentially be fused',
                'severity': 0.4,
                'suggestion': 'Consider loop fusion for efficiency'
            }
            
        return {
            'message': 'Optimization opportunities properly captured',
            'severity': 0.1,
            'suggestion': 'No missed optimization patterns'
        }


# ============================================================================
# Multi-Agent System with Debate and Consensus
# ============================================================================

class MultiAgentSystem:
    """
    Orchestrates multiple specialized agents using debate and consensus.
    
    Process:
    1. Run full analysis once (shared by all agents)
    2. All agents independently analyze with real analysis data
    3. Agents debate and critique each other's solutions
    4. Consensus is reached through weighted voting
    5. Final code synthesizes best insights
    """
    
    def __init__(self):
        self.agents = [
            StructureAgent(),
            DataFlowAgent(),
            TypeAgent(),
            MemoryAgent(),
            OptimizationAgent()
        ]
        self.max_debate_rounds = 3
        self.cache = {}
        
    def decompile(self, features: List[Dict], context: Dict = None) -> Dict:
        """
        Multi-agent decompilation with real analysis and debate.
        """
        if context is None:
            context = {}
            
        # Run full analysis once (shared by all agents)
        full_analysis = analyze_code(features)
        context['full_analysis'] = full_analysis
        
        # Stage 1: Independent Analysis
        proposals = []
        for agent in self.agents:
            try:
                proposal = agent.analyze(features, context)
                proposals.append(proposal)
            except Exception as e:
                # Handle agent failure gracefully
                proposals.append(AgentProposal(
                    agent_name=agent.name,
                    specialty=agent.specialty,
                    code=f"// Agent error: {str(e)}",
                    confidence=0.0,
                    reasoning=f"Analysis failed: {str(e)}",
                    analysis_data={}
                ))
        
        # Stage 2: Debate
        debate_log = []
        for round_num in range(self.max_debate_rounds):
            critiques = self._conduct_debate_round(proposals, round_num)
            debate_log.extend(critiques)
            
            # Adjust confidences based on critiques
            proposals = self._apply_critiques(proposals, critiques)
        
        # Stage 3: Consensus
        final_code, consensus_confidence, method = self._reach_consensus(proposals)
        
        return {
            'final_code': final_code,
            'confidence': consensus_confidence,
            'agent_proposals': [
                {
                    'agent': p.agent_name,
                    'specialty': p.specialty,
                    'code': p.code,
                    'confidence': p.confidence,
                    'reasoning': p.reasoning
                }
                for p in proposals
            ],
            'debate_log': debate_log,
            'consensus_method': method,
            'analysis_summary': {
                'num_instructions': full_analysis.get('statistics', {}).get('num_instructions', 0),
                'num_blocks': full_analysis.get('statistics', {}).get('num_blocks', 0),
                'num_variables': full_analysis.get('statistics', {}).get('num_variables', 0),
                'patterns_detected': len(full_analysis.get('patterns', []))
            }
        }
    
    def _conduct_debate_round(self, proposals: List[AgentProposal], 
                               round_num: int) -> List[str]:
        """Agents critique each other's proposals."""
        critiques = []
        
        for i, critic in enumerate(self.agents):
            for j, proposal in enumerate(proposals):
                if i != j:  # Don't critique self
                    critique = critic.critique(proposal)
                    
                    if critique['severity'] > 0.3:
                        msg = (f"Round {round_num + 1}: {critic.name} → "
                               f"{proposal.agent_name}: {critique['message']} "
                               f"(severity: {critique['severity']:.2f})")
                        critiques.append(msg)
                        
                        # Record critique in proposal
                        proposal.critique_history.append(critique['message'])
        
        return critiques
    
    def _apply_critiques(self, proposals: List[AgentProposal], 
                         critiques: List[str]) -> List[AgentProposal]:
        """Adjust proposal confidences based on critiques."""
        # Count critiques per agent
        critique_counts = {p.agent_name: 0 for p in proposals}
        
        for critique in critiques:
            for agent_name in critique_counts.keys():
                if f"→ {agent_name}" in critique:
                    critique_counts[agent_name] += 1
        
        # Adjust confidences
        for proposal in proposals:
            num_critiques = critique_counts.get(proposal.agent_name, 0)
            
            if num_critiques > 3:
                proposal.confidence *= 0.85
            elif num_critiques == 0:
                proposal.confidence = min(0.98, proposal.confidence * 1.05)
        
        return proposals
    
    def _reach_consensus(self, proposals: List[AgentProposal]) -> Tuple[str, float, str]:
        """Reach consensus using weighted voting and synthesis."""
        if not proposals:
            return "// No consensus reached", 0.0, "none"
        
        # Sort by confidence
        sorted_proposals = sorted(proposals, key=lambda p: p.confidence, reverse=True)
        best = sorted_proposals[0]
        
        # Check for clear winner
        if len(sorted_proposals) > 1:
            gap = best.confidence - sorted_proposals[1].confidence
            
            if gap > 0.2:
                return best.code, best.confidence, 'clear_winner'
        
        # Otherwise, synthesize from top proposals
        top_n = min(3, len(sorted_proposals))
        top_proposals = sorted_proposals[:top_n]
        
        # Generate synthesized code
        synthesized = self._synthesize_code(top_proposals)
        
        avg_confidence = sum(p.confidence for p in top_proposals) / len(top_proposals)
        
        return synthesized, avg_confidence, 'weighted_synthesis'
    
    def _synthesize_code(self, proposals: List[AgentProposal]) -> str:
        """Synthesize final code from multiple proposals."""
        lines = []
        lines.append(f"// Multi-agent consensus ({len(proposals)} agents)")
        lines.append("// Synthesized from best proposals")
        lines.append("")
        
        # Find structure from StructureAgent
        structure_proposal = next(
            (p for p in proposals if p.specialty == 'control_flow'), 
            proposals[0]
        )
        lines.append(f"// Primary structure from {structure_proposal.agent_name}")
        lines.append(structure_proposal.code)
        
        lines.append("")
        lines.append("/* Agent Insights:")
        
        for proposal in proposals:
            if proposal.specialty != 'control_flow':
                lines.append(f" * {proposal.agent_name} ({proposal.confidence:.2f}): "
                            f"{proposal.reasoning[:80]}...")
        
        lines.append(" */")
        
        return "\n".join(lines)


# ============================================================================
# Flask Application
# ============================================================================

# Global system instance
multi_agent_system = MultiAgentSystem()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model': 'multi-agent-system-v2',
        'num_agents': len(multi_agent_system.agents),
        'agents': [
            {'name': a.name, 'specialty': a.specialty}
            for a in multi_agent_system.agents
        ],
        'features': [
            'Real CFG analysis',
            'Data flow analysis',
            'Type inference',
            'Pattern matching',
            'Multi-round debate'
        ]
    })


@app.route('/decompile', methods=['POST'])
def decompile():
    """
    Multi-agent decompilation endpoint.
    
    Request:
    {
        "features": [...],  // P-Code features
        "context": {...}    // Optional context
    }
    
    Returns:
    {
        "final_code": "...",
        "confidence": 0.85,
        "agent_proposals": [...],
        "debate_log": [...],
        "consensus_method": "weighted_synthesis"
    }
    """
    try:
        data = request.json
        features = data.get('features', data.get('code_fragment', []))
        context = data.get('context', {})
        
        if not features:
            return jsonify({'error': 'features required'}), 400
        
        # Handle string input (convert to list of instructions)
        if isinstance(features, str):
            features = [{'mnemonic': 'RAW', 'raw': features}]
        
        result = multi_agent_system.decompile(features, context)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/refine', methods=['POST'])
def refine():
    """
    Refine code based on feedback using multi-agent collaboration.
    
    Request:
    {
        "current_code": "...",
        "feedback": "Error: ...",
        "features": [...],
        "context": {...}
    }
    """
    try:
        data = request.json
        current_code = data.get('current_code', '')
        feedback = data.get('feedback', '')
        features = data.get('features', [])
        context = data.get('context', {})
        
        # Add feedback to context
        context['feedback'] = feedback
        context['previous_code'] = current_code
        
        # Re-run analysis with feedback
        result = multi_agent_system.decompile(features, context)
        
        return jsonify({
            'refined_code': result['final_code'],
            'confidence': result['confidence'],
            'agent_reports': result['agent_proposals'],
            'method': 'multi_agent_refinement'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Get detailed analysis without full decompilation.
    
    Request:
    {
        "features": [...]
    }
    
    Returns:
    {
        "cfg": {...},
        "data_flow": {...},
        "types": {...},
        "patterns": [...]
    }
    """
    try:
        data = request.json
        features = data.get('features', [])
        
        if not features:
            return jsonify({'error': 'features required'}), 400
        
        analysis = analyze_code(features)
        
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/agents', methods=['GET'])
def list_agents():
    """List all available agents and their capabilities."""
    return jsonify({
        'agents': [
            {
                'name': agent.name,
                'specialty': agent.specialty,
                'description': agent.__class__.__doc__.strip() if agent.__class__.__doc__ else 'No description'
            }
            for agent in multi_agent_system.agents
        ]
    })


if __name__ == '__main__':
    print("Starting Multi-Agent Decompilation System v2")
    print(f"Loaded {len(multi_agent_system.agents)} agents:")
    for agent in multi_agent_system.agents:
        print(f"  - {agent.name} ({agent.specialty})")
    
    app.run(host='0.0.0.0', port=5007, debug=True)
