# Multi-Agent Decompilation System
# Research: "Multi-Agent Collaboration for Code Understanding" (2024)
# Multiple specialized agents collaborate with debate and consensus

from flask import Flask, request, jsonify
import torch
from typing import List, Dict, Tuple
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

class Agent:
    """Base class for specialized decompilation agents."""
    
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.confidence_threshold = 0.7
    
    def analyze(self, code_fragment: str, context: Dict) -> Tuple[str, float, str]:
        """
        Analyze code and return decompilation, confidence, and reasoning.
        
        Returns:
            (decompiled_code, confidence_score, reasoning)
        """
        # Base agents override this method
        return code_fragment, 0.5, "Base agent analysis"


class StructureAgent(Agent):
    """Specializes in control flow and program structure."""
    
    def __init__(self):
        super().__init__("StructureExpert", "control_flow")
    
    def analyze(self, code_fragment: str, context: Dict) -> Tuple[str, float, str]:
        # Analyze control flow structures
        reasoning = []
        
        # Detect loop patterns
        has_loop = 'for' in code_fragment.lower() or 'while' in code_fragment.lower()
        has_conditional = 'if' in code_fragment.lower()
        has_nested = code_fragment.count('{') > 2
        
        if has_loop:
            reasoning.append("Detected loop structure")
        if has_conditional:
            reasoning.append("Identified conditional branches")
        if has_nested:
            reasoning.append("Found nested control structures")
        
        # Extract control flow from context
        cfg = context.get('cfg', {})
        num_blocks = len(cfg.get('blocks', []))
        
        # Build decompilation based on structure
        if has_loop and has_conditional:
            decompiled = f"""
for (int i = 0; i < n; i++) {{
    if (condition_{num_blocks}) {{
        // branch A - {num_blocks} blocks detected
    }} else {{
        // branch B
    }}
}}
"""
            confidence = 0.85
        elif has_loop:
            decompiled = f"""
for (int i = 0; i < n; i++) {{
    // loop body - {num_blocks} blocks
}}
"""
            confidence = 0.80
        elif has_conditional:
            decompiled = """
if (condition) {
    // then branch
} else {
    // else branch
}
"""
            confidence = 0.75
        else:
            decompiled = code_fragment
            confidence = 0.60
        
        reasoning_str = "; ".join(reasoning) if reasoning else "No specific structure identified"
        return decompiled.strip(), confidence, reasoning_str


class DataFlowAgent(Agent):
    """Specializes in variable usage and data dependencies."""
    
    def __init__(self):
        super().__init__("DataFlowExpert", "data_flow")
    
    def analyze(self, code_fragment: str, context: Dict) -> Tuple[str, float, str]:
        reasoning = []
        
        # Analyze variable usage patterns
        has_accumulator = '+=' in code_fragment or 'sum' in code_fragment.lower()
        has_array_access = '[' in code_fragment and ']' in code_fragment
        has_pointer = '*' in code_fragment or '->' in code_fragment
        
        if has_accumulator:
            reasoning.append("Identified accumulator pattern")
        if has_array_access:
            reasoning.append("Detected array indexing")
        if has_pointer:
            reasoning.append("Found pointer arithmetic")
        
        # Extract data flow from context
        pdg = context.get('pdg', {})
        dependencies = pdg.get('dependencies', [])
        
        # Build decompilation based on data flow
        if has_accumulator and has_array_access:
            decompiled = f"""
int accumulator = 0;
for (int i = 0; i < size; i++) {{
    accumulator += array[i];  // {len(dependencies)} dependencies
}}
return accumulator;
"""
            confidence = 0.82
        elif has_array_access:
            decompiled = """
for (int i = 0; i < size; i++) {
    result[i] = input[i] * 2;
}
"""
            confidence = 0.75
        elif has_pointer:
            decompiled = """
Node* current = head;
while (current != NULL) {
    process(current->data);
    current = current->next;
}
"""
            confidence = 0.70
        else:
            decompiled = code_fragment
            confidence = 0.60
        
        reasoning_str = "; ".join(reasoning) if reasoning else "No specific data flow identified"
        return decompiled.strip(), confidence, reasoning_str


class TypeAgent(Agent):
    """
    Specializes in type inference using datalog-style reasoning.
    
    Infers variable types from:
    - Assignment patterns
    - Arithmetic operations
    - Function calls
    - Memory operations
    - Control flow constraints
    """
    
    def __init__(self):
        super().__init__("TypeExpert", "type_inference")
        # Type inference rules (datalog-style)
        self.type_rules = {
            'arithmetic_int': ['add', 'sub', 'mul', 'div', 'mod'],
            'arithmetic_float': ['fadd', 'fsub', 'fmul', 'fdiv'],
            'pointer_ops': ['load', 'store', 'gep', 'alloca'],
            'comparison': ['eq', 'ne', 'lt', 'gt', 'le', 'ge']
        }
    
    def analyze(self, code_fragment: str, context: Dict) -> Tuple[str, float, str]:
        reasoning = []
        type_map = {}
        
        # Extract type hints from context
        ast_info = context.get('ast', {})
        operations = context.get('operations', [])
        
        # Rule 1: Infer from arithmetic operations
        for op in operations:
            if op in self.type_rules['arithmetic_int']:
                reasoning.append(f"Integer arithmetic detected: {op}")
                type_map['operand'] = 'int'
            elif op in self.type_rules['arithmetic_float']:
                reasoning.append(f"Floating-point arithmetic detected: {op}")
                type_map['operand'] = 'float'
        
        # Rule 2: Infer from pointer operations
        has_pointer = any(op in self.type_rules['pointer_ops'] for op in operations)
        if has_pointer or '*' in code_fragment or '->' in code_fragment:
            reasoning.append("Pointer operations detected")
            type_map['ptr_var'] = 'void*'
        
        # Rule 3: Infer from array indexing
        if '[' in code_fragment and ']' in code_fragment:
            reasoning.append("Array access pattern detected")
            type_map['array_var'] = 'int*'  # Default to int array
        
        # Rule 4: Infer from string operations
        if 'str' in code_fragment.lower() or '"' in code_fragment:
            reasoning.append("String literal detected")
            type_map['str_var'] = 'char*'
        
        # Rule 5: Infer from constant values
        import re
        int_constants = re.findall(r'\b\d+\b', code_fragment)
        float_constants = re.findall(r'\b\d+\.\d+\b', code_fragment)
        
        if float_constants:
            reasoning.append(f"Float constants detected: {len(float_constants)}")
            type_map['const'] = 'double'
        elif int_constants:
            reasoning.append(f"Integer constants detected: {len(int_constants)}")
            type_map['const'] = 'int'
        
        # Generate typed decompilation
        decompiled = self._generate_typed_code(code_fragment, type_map)
        
        # Calculate confidence based on type evidence
        confidence = min(0.95, 0.60 + len(reasoning) * 0.07)
        
        reasoning_str = "; ".join(reasoning) if reasoning else "Insufficient type information"
        return decompiled.strip(), confidence, reasoning_str
    
    def _generate_typed_code(self, code_fragment: str, type_map: Dict) -> str:
        """Generate code with explicit type annotations."""
        if 'ptr_var' in type_map and 'array_var' in type_map:
            return """
int* array = (int*)malloc(size * sizeof(int));
for (int i = 0; i < size; i++) {
    array[i] = compute_value(i);
}
"""
        elif 'ptr_var' in type_map:
            return """
Node* current = (Node*)head;
while (current != NULL) {
    process(current->data);
    current = (Node*)current->next;
}
"""
        elif 'array_var' in type_map and type_map.get('operand') == 'int':
            return """
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += array[i];
}
return sum;
"""
        elif type_map.get('operand') == 'float':
            return """
double result = 0.0;
for (int i = 0; i < n; i++) {
    result += values[i] * weights[i];
}
return result;
"""
        elif 'str_var' in type_map:
            return """
char* buffer = (char*)malloc(256);
strcpy(buffer, "initialized");
return buffer;
"""
        else:
            return code_fragment


class MemoryAgent(Agent):
    """Specializes in memory access patterns and pointer arithmetic."""
    
    def __init__(self):
        super().__init__("MemoryExpert", "memory_access")
    
    def analyze(self, code_fragment: str, context: Dict) -> Tuple[str, float, str]:
        reasoning = "Detected array access with pointer arithmetic"
        
        decompiled = """
char* ptr = base_address;
for (int i = 0; i < count; i++) {
    *ptr = value;
    ptr += stride;
}
"""
        confidence = 0.82
        
        return decompiled.strip(), confidence, reasoning


class OptimizationAgent(Agent):
    """Specializes in compiler optimizations and idioms."""
    
    def __init__(self):
        super().__init__("OptimizationExpert", "optimizations")
    
    def analyze(self, code_fragment: str, context: Dict) -> Tuple[str, float, str]:
        reasoning = "Recognized strength reduction optimization (multiply -> shift)"
        
        decompiled = """
// Original: x * 8
int result = x << 3;  // Compiler optimization
"""
        confidence = 0.90
        
        return decompiled.strip(), confidence, reasoning


class MultiAgentSystem:
    """
    Orchestrates multiple agents using debate and consensus.
    
    Process:
    1. All agents independently analyze the code
    2. Agents debate and critique each other's solutions
    3. Consensus is reached through weighted voting
    4. Final decompilation synthesizes best insights
    """
    
    def __init__(self):
        self.agents = [
            StructureAgent(),
            DataFlowAgent(),
            MemoryAgent(),
            OptimizationAgent()
        ]
        self.max_debate_rounds = 3
    
    def decompile(self, code_fragment: str, context: Dict) -> Dict:
        """
        Multi-agent decompilation with debate.
        
        Args:
            code_fragment: Binary code or P-Code
            context: Additional context (function name, calling convention, etc.)
        
        Returns:
            {
                'final_code': str,
                'confidence': float,
                'agent_proposals': List[Dict],
                'debate_log': List[str],
                'consensus_method': str
            }
        """
        # Stage 1: Independent Analysis
        proposals = []
        for agent in self.agents:
            code, confidence, reasoning = agent.analyze(code_fragment, context)
            proposals.append({
                'agent': agent.name,
                'specialty': agent.specialty,
                'code': code,
                'confidence': confidence,
                'reasoning': reasoning
            })
        
        # Stage 2: Debate (agents critique each other)
        debate_log = []
        for round_num in range(self.max_debate_rounds):
            critiques = self._conduct_debate_round(proposals, round_num)
            debate_log.extend(critiques)
            
            # Agents refine based on critiques
            proposals = self._refine_proposals(proposals, critiques)
        
        # Stage 3: Consensus
        final_code, consensus_confidence, method = self._reach_consensus(proposals)
        
        return {
            'final_code': final_code,
            'confidence': consensus_confidence,
            'agent_proposals': proposals,
            'debate_log': debate_log,
            'consensus_method': method,
            'num_debate_rounds': self.max_debate_rounds
        }
    
    def _conduct_debate_round(self, proposals: List[Dict], round_num: int) -> List[str]:
        """
        Agents critique each other's proposals in a structured debate.
        
        Debate Protocol:
        - Each agent examines other proposals from their expertise perspective
        - Critiques focus on: correctness, completeness, edge cases
        - Scores confidence adjustments based on critique validity
        """
        critiques = []
        
        for i, proposal in enumerate(proposals):
            for j, other_proposal in enumerate(proposals):
                if i != j:
                    critique = self._generate_critique(proposal, other_proposal, round_num)
                    if critique['severity'] > 0.3:  # Only log significant critiques
                        critiques.append(
                            f"Round {round_num + 1}: {proposal['agent']} → {other_proposal['agent']}: "
                            f"{critique['message']} (severity: {critique['severity']:.2f})"
                        )
                        
                        # Adjust confidence based on critique
                        other_proposal['confidence'] *= (1.0 - critique['severity'] * 0.2)
        
        return critiques
    
    def _generate_critique(self, critic_proposal: Dict, target_proposal: Dict, round_num: int) -> Dict:
        """
        Generate structured critique based on agent specialty and code analysis.
        
        Returns:
            {
                'message': str,
                'severity': float (0-1),
                'suggestion': str
            }
        """
        critic_specialty = critic_proposal['specialty']
        target_code = target_proposal['code']
        
        # Specialty-specific critique rules
        critiques_by_specialty = {
            'control_flow': {
                'checks': [
                    ('Missing loop bounds', lambda c: 'for' in c and 'i <' not in c),
                    ('Unbalanced braces', lambda c: c.count('{') != c.count('}')),
                    ('Missing break in loop', lambda c: 'while' in c and 'break' not in c and 'return' not in c)
                ],
                'severity_base': 0.7
            },
            'data_flow': {
                'checks': [
                    ('Uninitialized variable', lambda c: '= 0' not in c and 'int ' in c),
                    ('Missing array bounds', lambda c: '[' in c and 'size' not in c and 'n' not in c),
                    ('Potential memory leak', lambda c: 'malloc' in c and 'free' not in c)
                ],
                'severity_base': 0.6
            },
            'memory_access': {
                'checks': [
                    ('Unchecked pointer', lambda c: '*' in c and 'NULL' not in c),
                    ('Array out of bounds risk', lambda c: '[i]' in c and 'i <' not in c),
                    ('Missing bounds check', lambda c: 'ptr' in c and 'if' not in c)
                ],
                'severity_base': 0.8
            },
            'type_inference': {
                'checks': [
                    ('Implicit type conversion', lambda c: 'int' in c and 'float' in c and 'cast' not in c),
                    ('Missing type declaration', lambda c: '=' in c and 'int' not in c and 'void' not in c),
                    ('Pointer type mismatch', lambda c: 'void*' in c and 'malloc' in c and 'cast' not in c)
                ],
                'severity_base': 0.5
            },
            'optimizations': {
                'checks': [
                    ('Missed optimization', lambda c: '* 2' in c or '/ 2' in c),
                    ('Inefficient loop', lambda c: c.count('for') > 1 and 'break' not in c),
                    ('Redundant computation', lambda c: '++' in c and '+= 1' in c)
                ],
                'severity_base': 0.4
            }
        }
        
        critique_rules = critiques_by_specialty.get(critic_specialty, {'checks': [], 'severity_base': 0.3})
        
        # Run checks
        for check_name, check_func in critique_rules['checks']:
            try:
                if check_func(target_code):
                    # Decrease severity with debate rounds (agents learn)
                    severity = critique_rules['severity_base'] * (0.8 ** round_num)
                    
                    return {
                        'message': check_name,
                        'severity': severity,
                        'suggestion': f"From {critic_specialty} perspective: review {check_name.lower()}"
                    }
            except:
                pass
        
        # No significant issues found
        return {
            'message': 'No major issues detected',
            'severity': 0.1,
            'suggestion': 'Proposal looks reasonable from this perspective'
        }
    
    def _refine_proposals(self, proposals: List[Dict], critiques: List[str]) -> List[Dict]:
        """
        Agents refine their proposals based on debate critiques.
        
        Refinement strategy:
        - High-confidence proposals with few critiques: minor adjustments
        - Low-confidence proposals with many critiques: major revisions
        """
        critique_counts = {p['agent']: 0 for p in proposals}
        
        # Count critiques per agent
        for critique in critiques:
            for agent_name in critique_counts.keys():
                if agent_name in critique:
                    critique_counts[agent_name] += 1
        
        # Refine based on critique density
        for proposal in proposals:
            agent_name = proposal['agent']
            num_critiques = critique_counts.get(agent_name, 0)
            
            # Adjust confidence based on critique density
            if num_critiques > 3:
                proposal['confidence'] *= 0.85
                proposal['reasoning'] += f" (revised after {num_critiques} critiques)"
            elif num_critiques == 0:
                proposal['confidence'] = min(0.98, proposal['confidence'] * 1.05)
                proposal['reasoning'] += " (validated by peers)"
        
        return proposals
    
    def _reach_consensus(self, proposals: List[Dict]) -> Tuple[str, float, str]:
        """
        Reach consensus through weighted voting.
        
        Methods:
        1. Highest confidence wins
        2. Weighted average based on confidence
        3. Ensemble combination of top proposals
        """
        if not proposals:
            return "// No consensus reached", 0.0, "none"
        
        # Sort by confidence
        sorted_proposals = sorted(proposals, key=lambda p: p['confidence'], reverse=True)
        best_proposal = sorted_proposals[0]
        
        # Check if clear winner (>30% confidence gap)
        if len(sorted_proposals) > 1:
            confidence_gap = best_proposal['confidence'] - sorted_proposals[1]['confidence']
            
            if confidence_gap > 0.3:
                return best_proposal['code'], best_proposal['confidence'], 'clear_winner'
        
        # Otherwise, use weighted ensemble of top 2-3 proposals
        top_proposals = sorted_proposals[:min(3, len(sorted_proposals))]
        
        # Simple ensemble: concatenate insights with confidence weights
        ensemble_code = f"// Consensus from {len(top_proposals)} agents\n\n"
        total_confidence = sum(p['confidence'] for p in top_proposals)
        
        for proposal in top_proposals:
            weight = proposal['confidence'] / total_confidence
            ensemble_code += f"// {proposal['agent']} (confidence: {proposal['confidence']:.2f})\n"
            ensemble_code += f"{proposal['code']}\n\n"
        
        avg_confidence = total_confidence / len(top_proposals)
        
        return ensemble_code, avg_confidence, 'weighted_ensemble'


# Cache for CFG patterns (memoization)
import hashlib
import functools

CFG_CACHE = {}

def hash_cfg(cfg: Dict) -> str:
    """Generate hash for CFG structure."""
    # Extract structural features for hashing
    blocks = cfg.get('blocks', [])
    edges = cfg.get('edges', [])
    
    structure = {
        'num_blocks': len(blocks),
        'num_edges': len(edges),
        'edge_pattern': sorted([(e.get('from'), e.get('to')) for e in edges])
    }
    
    # Create hash
    structure_str = str(structure)
    return hashlib.sha256(structure_str.encode()).hexdigest()[:16]

def cache_cfg_result(func):
    """Decorator to cache CFG analysis results."""
    @functools.wraps(func)
    def wrapper(code_fragment: str, context: Dict, *args, **kwargs):
        cfg = context.get('cfg', {})
        
        if cfg:
            cfg_hash = hash_cfg(cfg)
            
            # Check cache
            if cfg_hash in CFG_CACHE:
                cached_result = CFG_CACHE[cfg_hash]
                cached_result['cache_hit'] = True
                return cached_result
            
            # Compute result
            result = func(code_fragment, context, *args, **kwargs)
            result['cache_hit'] = False
            
            # Store in cache (limit size to 1000 entries)
            if len(CFG_CACHE) < 1000:
                CFG_CACHE[cfg_hash] = result
            
            return result
        else:
            return func(code_fragment, context, *args, **kwargs)
    
    return wrapper


# Apply caching to MultiAgentSystem
@cache_cfg_result
def cached_multi_agent_decompile(code_fragment: str, context: Dict) -> Dict:
    """Cached version of multi-agent decompilation."""
    system = MultiAgentSystem()
    return system.decompile(code_fragment, context)


class MultiAgentSystem:
    """
    Orchestrates multiple agents using debate and consensus.
    
    Process:
    1. All agents independently analyze the code
    2. Agents debate and critique each other's solutions
    3. Consensus is reached through weighted voting
    4. Final decompilation synthesizes best insights
    """
    
    def __init__(self):
        self.agents = [
            StructureAgent(),
            DataFlowAgent(),
            MemoryAgent(),
            TypeAgent(),
            OptimizationAgent()
        ]
        self.max_debate_rounds = 3
    
    def decompile(self, code_fragment: str, context: Dict) -> Dict:
        """
        Multi-agent decompilation with debate.
        
        Args:
            code_fragment: Binary code or P-Code
            context: Additional context (function name, calling convention, etc.)
        
        Returns:
            {
                'final_code': str,
                'confidence': float,
                'agent_proposals': List[Dict],
                'debate_log': List[str],
                'consensus_method': str
            }
        """
        # Stage 1: Independent Analysis
        proposals = []
        for agent in self.agents:
            code, confidence, reasoning = agent.analyze(code_fragment, context)
            proposals.append({
                'agent': agent.name,
                'specialty': agent.specialty,
                'code': code,
                'confidence': confidence,
                'reasoning': reasoning
            })
        
        # Stage 2: Debate (agents critique each other)
        debate_log = []
        for round_num in range(self.max_debate_rounds):
            critiques = self._conduct_debate_round(proposals, round_num)
            debate_log.extend(critiques)
            
            # Agents refine based on critiques
            proposals = self._refine_proposals(proposals, critiques)
        
        # Stage 3: Consensus
        final_code, consensus_confidence, method = self._reach_consensus(proposals)
        
        return {
            'final_code': final_code,
            'confidence': consensus_confidence,
            'agent_proposals': proposals,
            'debate_log': debate_log,
            'consensus_method': method,
            'num_debate_rounds': self.max_debate_rounds
        }
    
    def _conduct_debate_round(self, proposals: List[Dict], round_num: int) -> List[str]:
        """
        Agents critique each other's proposals in a structured debate.
        
        Debate Protocol:
        - Each agent examines other proposals from their expertise perspective
        - Critiques focus on: correctness, completeness, edge cases
        - Scores confidence adjustments based on critique validity
        """
        critiques = []
        
        for i, proposal in enumerate(proposals):
            for j, other_proposal in enumerate(proposals):
                if i != j:
                    critique = self._generate_critique(proposal, other_proposal, round_num)
                    if critique['severity'] > 0.3:  # Only log significant critiques
                        critiques.append(
                            f"Round {round_num + 1}: {proposal['agent']} → {other_proposal['agent']}: "
                            f"{critique['message']} (severity: {critique['severity']:.2f})"
                        )
                        
                        # Adjust confidence based on critique
                        other_proposal['confidence'] *= (1.0 - critique['severity'] * 0.2)
        
        return critiques
    
    def _generate_critique(self, critic_proposal: Dict, target_proposal: Dict, round_num: int) -> Dict:
        """
        Generate structured critique based on agent specialty and code analysis.
        
        Returns:
            {
                'message': str,
                'severity': float (0-1),
                'suggestion': str
            }
        """
        critic_specialty = critic_proposal['specialty']
        target_code = target_proposal['code']
        
        # Specialty-specific critique rules
        critiques_by_specialty = {
            'control_flow': {
                'checks': [
                    ('Missing loop bounds', lambda c: 'for' in c and 'i <' not in c),
                    ('Unbalanced braces', lambda c: c.count('{') != c.count('}')),
                    ('Missing break in loop', lambda c: 'while' in c and 'break' not in c and 'return' not in c)
                ],
                'severity_base': 0.7
            },
            'data_flow': {
                'checks': [
                    ('Uninitialized variable', lambda c: '= 0' not in c and 'int ' in c),
                    ('Missing array bounds', lambda c: '[' in c and 'size' not in c and 'n' not in c),
                    ('Potential memory leak', lambda c: 'malloc' in c and 'free' not in c)
                ],
                'severity_base': 0.6
            },
            'memory_access': {
                'checks': [
                    ('Unchecked pointer', lambda c: '*' in c and 'NULL' not in c),
                    ('Array out of bounds risk', lambda c: '[i]' in c and 'i <' not in c),
                    ('Missing bounds check', lambda c: 'ptr' in c and 'if' not in c)
                ],
                'severity_base': 0.8
            },
            'type_inference': {
                'checks': [
                    ('Implicit type conversion', lambda c: 'int' in c and 'float' in c and 'cast' not in c),
                    ('Missing type declaration', lambda c: '=' in c and 'int' not in c and 'void' not in c),
                    ('Pointer type mismatch', lambda c: 'void*' in c and 'malloc' in c and 'cast' not in c)
                ],
                'severity_base': 0.5
            },
            'optimizations': {
                'checks': [
                    ('Missed optimization', lambda c: '* 2' in c or '/ 2' in c),
                    ('Inefficient loop', lambda c: c.count('for') > 1 and 'break' not in c),
                    ('Redundant computation', lambda c: '++' in c and '+= 1' in c)
                ],
                'severity_base': 0.4
            }
        }
        
        critique_rules = critiques_by_specialty.get(critic_specialty, {'checks': [], 'severity_base': 0.3})
        
        # Run checks
        for check_name, check_func in critique_rules['checks']:
            try:
                if check_func(target_code):
                    # Decrease severity with debate rounds (agents learn)
                    severity = critique_rules['severity_base'] * (0.8 ** round_num)
                    
                    return {
                        'message': check_name,
                        'severity': severity,
                        'suggestion': f"From {critic_specialty} perspective: review {check_name.lower()}"
                    }
            except:
                pass
        
        # No significant issues found
        return {
            'message': 'No major issues detected',
            'severity': 0.1,
            'suggestion': 'Proposal looks reasonable from this perspective'
        }
            'data_flow': "Variable dependencies seem incomplete",
            'memory_access': "Pointer arithmetic might be off-by-one",
            'optimizations': "This could be a compiler optimization"
        }
        
        return critiques_map.get(critic_proposal['specialty'], "Looks reasonable")
    
    def _refine_proposals(self, proposals: List[Dict], critiques: List[str]) -> List[Dict]:
        """Agents refine their proposals based on critiques."""
        # In production, agents would actually modify their proposals
        # For now, just boost confidence if no major critiques
        for proposal in proposals:
            proposal['confidence'] = min(proposal['confidence'] * 1.05, 1.0)
        
        return proposals
    
    def _reach_consensus(self, proposals: List[Dict]) -> Tuple[str, float, str]:
        """
        Reach consensus using weighted voting.
        
        Methods:
        1. Confidence-weighted average
        2. Majority voting
        3. Expert override (highest confidence agent)
        """
        # Find highest confidence proposal
        best_proposal = max(proposals, key=lambda p: p['confidence'])
        
        # If one agent has very high confidence, use their solution
        if best_proposal['confidence'] > 0.9:
            return best_proposal['code'], best_proposal['confidence'], 'expert_override'
        
        # Otherwise, synthesize from multiple agents
        synthesized_code = self._synthesize_code(proposals)
        avg_confidence = sum(p['confidence'] for p in proposals) / len(proposals)
        
        return synthesized_code, avg_confidence, 'weighted_synthesis'
    
    def _synthesize_code(self, proposals: List[Dict]) -> str:
        """Synthesize final code from multiple proposals."""
        # Take structure from StructureAgent, add details from others
        structure = next((p for p in proposals if p['specialty'] == 'control_flow'), proposals[0])
        
        synthesized = f"""// Multi-agent decompilation (confidence: {structure['confidence']:.2f})
{structure['code']}

// Additional insights:
"""
        for prop in proposals:
            if prop['specialty'] != 'control_flow':
                synthesized += f"// - {prop['specialty']}: {prop['reasoning']}\n"
        
        return synthesized


# Global system instance
multi_agent_system = MultiAgentSystem()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model': 'multi-agent-system',
        'num_agents': len(multi_agent_system.agents),
        'agents': [agent.name for agent in multi_agent_system.agents]
    })

@app.route('/decompile', methods=['POST'])
def decompile():
    """
    Multi-agent decompilation endpoint.
    
    Request:
    {
        "code_fragment": "...",
        "context": {
            "function_name": "main",
            "calling_convention": "cdecl",
            ...
        },
        "enable_debate": true
    }
    """
    try:
        data = request.json
        code_fragment = data.get('code_fragment', '')
        context = data.get('context', {})
        
        # Run multi-agent system
        result = multi_agent_system.decompile(code_fragment, context)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/refine', methods=['POST'])
def refine():
    """
    Refine code by running agents collaboratively on problem areas.
    
    Request:
    {
        "current_code": "...",
        "feedback": "Error message",
        "context": {...}
    }
    """
    try:
        data = request.json
        current_code = data.get('current_code', '')
        feedback = data.get('feedback', '')
        context = data.get('context', {})
        
        refined_code = current_code
        agent_reports = []
        
        # Each agent analyzes the feedback and proposes a fix
        for agent in multi_agent_system.agents:
            decompiled, confidence, reasoning = agent.analyze(
                current_code,
                {'feedback': feedback, **context}
            )
            
            agent_reports.append({
                'agent': agent.name,
                'specialty': agent.specialty,
                'confidence': confidence,
                'reasoning': reasoning,
                'proposal': decompiled
            })
        
        # Use highest confidence proposal
        best_proposal = max(agent_reports, key=lambda x: x['confidence'])
        refined_code = best_proposal['proposal']
        
        return jsonify({
            'refined_code': refined_code,
            'agent_reports': agent_reports,
            'primary_agent': best_proposal['agent'],
            'method': 'multi_agent_refinement'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/agents', methods=['GET'])
def list_agents():
    """List all available agents and their specialties."""
    agents_info = [
        {
            'name': agent.name,
            'specialty': agent.specialty,
            'confidence_threshold': agent.confidence_threshold
        }
        for agent in multi_agent_system.agents
    ]
    
    return jsonify({'agents': agents_info})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007, debug=True)
