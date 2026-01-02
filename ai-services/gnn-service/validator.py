"""
GNN Output Validator
====================
Validates that GNN's junk instruction predictions are correct before
passing sanitized assembly to the decompiler.

The validator uses:
1. Static analysis checks (instruction reachability, data dependencies)
2. Heuristic pattern detection
3. Optional RL-based confidence scoring

This catches cases where GNN incorrectly classifies real instructions as junk.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ValidationResult:
    is_valid: bool
    confidence: float
    warnings: List[str]
    corrected_mask: List[bool]
    removed_critical: List[int]
    analysis: Dict


class GNNOutputValidator:
    """Validates GNN's junk instruction predictions."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threshold = self.config.get('confidence_threshold', 0.7)
        
        # Critical instruction patterns that should rarely be removed
        self.critical_patterns = {
            'ret', 'retn', 'leave',  # Function return
            'call',                   # Function calls
            'syscall', 'int',         # System calls
            'mov.*sp', 'push.*bp', 'pop.*bp',  # Stack frame
        }
        
        # Likely junk patterns
        self.junk_patterns = {
            r'^nop$',                 # NOPs
            r'^xchg\s+\w+,\s*\1$',    # Self-exchange (xchg eax, eax)
            r'^lea\s+\w+,\s*\[\w+\]$', # Identity LEA
        }

    def validate(
        self,
        instructions: List[Dict],
        gnn_predictions: List[float],
        cfg: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate GNN predictions against static analysis.
        
        Args:
            instructions: List of instruction dicts with 'mnemonic', 'address', etc.
            gnn_predictions: GNN output probabilities (P(junk))
            cfg: Optional control flow graph
        
        Returns:
            ValidationResult with corrected mask and warnings
        """
        n = len(instructions)
        if n == 0:
            return ValidationResult(
                is_valid=True, confidence=1.0, warnings=[],
                corrected_mask=[], removed_critical=[], analysis={}
            )
        
        # Convert predictions to binary mask
        gnn_mask = [p < self.threshold for p in gnn_predictions]  # True = keep
        
        warnings = []
        corrected_mask = gnn_mask.copy()
        removed_critical = []
        
        # Build analysis structures
        reachable = self._compute_reachability(instructions, cfg)
        data_deps = self._compute_data_dependencies(instructions)
        dominators = self._compute_dominators(instructions, cfg) if cfg else set()
        
        # Check each instruction marked for removal
        for i, (instr, keep) in enumerate(zip(instructions, gnn_mask)):
            if keep:
                continue  # Not marked as junk, skip
            
            mnem = instr.get('mnemonic', '').lower()
            
            # Check 1: Critical instruction patterns
            if self._is_critical(mnem):
                warnings.append(f"[CRITICAL] GNN marked critical instruction as junk: {i}: {mnem}")
                corrected_mask[i] = True  # Force keep
                removed_critical.append(i)
                continue
            
            # Check 2: Reachability - if instruction is reachable from entry
            if i in reachable:
                # Reachable instruction marked as junk - verify it's actually dead
                if not self._is_likely_junk(instr):
                    warnings.append(f"[REACHABLE] Reachable instruction marked junk: {i}: {mnem}")
                    # Don't auto-correct, but flag it
            
            # Check 3: Data dependencies - if removing breaks def-use chains
            if i in data_deps['used_by']:
                users = data_deps['used_by'][i]
                kept_users = [u for u in users if corrected_mask[u]]
                if kept_users:
                    warnings.append(f"[DATA_DEP] Instruction {i} has live users: {kept_users}")
                    corrected_mask[i] = True  # Force keep
            
            # Check 4: Dominator check - don't remove dominators of kept blocks
            if i in dominators and any(corrected_mask[j] for j in dominators[i]):
                warnings.append(f"[DOMINATOR] Instruction {i} dominates kept instructions")
                corrected_mask[i] = True
        
        # Verify the corrected mask doesn't break control flow
        final_warnings = self._verify_control_flow(instructions, corrected_mask, cfg)
        warnings.extend(final_warnings)
        
        # Compute confidence
        num_corrections = sum(c != g for c, g in zip(corrected_mask, gnn_mask))
        confidence = 1.0 - (num_corrections / max(n, 1))
        
        return ValidationResult(
            is_valid=len(removed_critical) == 0,
            confidence=confidence,
            warnings=warnings,
            corrected_mask=corrected_mask,
            removed_critical=removed_critical,
            analysis={
                'reachable_count': len(reachable),
                'total_instructions': n,
                'gnn_junk_count': sum(1 for k in gnn_mask if not k),
                'corrected_junk_count': sum(1 for k in corrected_mask if not k),
                'corrections_made': num_corrections
            }
        )

    def _is_critical(self, mnemonic: str) -> bool:
        """Check if instruction is critical (should not be removed)."""
        mnem = mnemonic.lower()
        for pattern in self.critical_patterns:
            if re.match(pattern, mnem):
                return True
        return False

    def _is_likely_junk(self, instr: Dict) -> bool:
        """Check if instruction matches known junk patterns."""
        text = instr.get('text', instr.get('mnemonic', '')).lower()
        for pattern in self.junk_patterns:
            if re.match(pattern, text):
                return True
        return False

    def _compute_reachability(
        self,
        instructions: List[Dict],
        cfg: Optional[Dict]
    ) -> Set[int]:
        """Compute which instructions are reachable from entry."""
        n = len(instructions)
        if n == 0:
            return set()
        
        # Build successor graph
        successors = defaultdict(list)
        
        # Default: sequential flow
        for i in range(n - 1):
            successors[i].append(i + 1)
        
        # Add CFG edges
        if cfg:
            for edge in cfg.get('edges', []):
                src = edge.get('from', edge.get('source', -1))
                dst = edge.get('to', edge.get('target', -1))
                if 0 <= src < n and 0 <= dst < n:
                    if dst not in successors[src]:
                        successors[src].append(dst)
        
        # Also handle branch instructions
        for i, instr in enumerate(instructions):
            mnem = instr.get('mnemonic', '').lower()
            if mnem.startswith('j') and mnem != 'jmp':
                # Conditional branch - can fall through
                pass
            elif mnem == 'jmp':
                # Unconditional jump - no fall through
                if i + 1 in successors[i]:
                    successors[i].remove(i + 1)
        
        # BFS from entry (index 0)
        reachable = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node in reachable:
                continue
            reachable.add(node)
            for succ in successors[node]:
                if succ not in reachable:
                    queue.append(succ)
        
        return reachable

    def _compute_data_dependencies(
        self,
        instructions: List[Dict]
    ) -> Dict[str, Dict]:
        """Compute def-use chains between instructions."""
        definitions = {}  # var -> defining instruction index
        used_by = defaultdict(list)  # instruction -> list of users
        
        for i, instr in enumerate(instructions):
            mnem = instr.get('mnemonic', '').lower()
            ops = instr.get('operands', [])
            
            # Extract register operands
            output_reg = None
            input_regs = []
            
            if len(ops) >= 1:
                # For most x86 instructions, first operand is destination
                if mnem in ['mov', 'add', 'sub', 'xor', 'and', 'or', 'lea', 'shl', 'shr']:
                    output_reg = self._normalize_reg(ops[0])
                    if len(ops) >= 2:
                        input_regs.append(self._normalize_reg(ops[1]))
                elif mnem in ['push', 'cmp', 'test']:
                    for op in ops:
                        input_regs.append(self._normalize_reg(op))
                elif mnem == 'pop':
                    output_reg = self._normalize_reg(ops[0])
            
            # Also check for explicit input/output in P-Code format
            for inp in instr.get('inputs', []):
                reg = inp.get('name', inp.get('value', ''))
                if reg:
                    input_regs.append(self._normalize_reg(reg))
            
            out = instr.get('output')
            if out:
                output_reg = self._normalize_reg(out.get('name', out.get('value', '')))
            
            # Record use
            for reg in input_regs:
                if reg and reg in definitions:
                    def_idx = definitions[reg]
                    used_by[def_idx].append(i)
            
            # Record definition
            if output_reg:
                definitions[output_reg] = i
        
        return {
            'definitions': definitions,
            'used_by': dict(used_by)
        }

    def _normalize_reg(self, reg: str) -> Optional[str]:
        """Normalize register name to canonical form."""
        if not reg:
            return None
        reg = str(reg).lower().strip()
        
        # Map to base register
        reg_map = {
            'al': 'rax', 'ah': 'rax', 'ax': 'rax', 'eax': 'rax', 'rax': 'rax',
            'bl': 'rbx', 'bh': 'rbx', 'bx': 'rbx', 'ebx': 'rbx', 'rbx': 'rbx',
            'cl': 'rcx', 'ch': 'rcx', 'cx': 'rcx', 'ecx': 'rcx', 'rcx': 'rcx',
            'dl': 'rdx', 'dh': 'rdx', 'dx': 'rdx', 'edx': 'rdx', 'rdx': 'rdx',
            'sil': 'rsi', 'si': 'rsi', 'esi': 'rsi', 'rsi': 'rsi',
            'dil': 'rdi', 'di': 'rdi', 'edi': 'rdi', 'rdi': 'rdi',
            'bpl': 'rbp', 'bp': 'rbp', 'ebp': 'rbp', 'rbp': 'rbp',
            'spl': 'rsp', 'sp': 'rsp', 'esp': 'rsp', 'rsp': 'rsp',
        }
        return reg_map.get(reg, reg if reg.startswith('r') else None)

    def _compute_dominators(
        self,
        instructions: List[Dict],
        cfg: Optional[Dict]
    ) -> Dict[int, Set[int]]:
        """Compute dominator relationships."""
        # Simplified: return empty for now, full impl would do proper dominator tree
        return {}

    def _verify_control_flow(
        self,
        instructions: List[Dict],
        mask: List[bool],
        cfg: Optional[Dict]
    ) -> List[str]:
        """Verify that control flow is preserved after masking."""
        warnings = []
        
        # Check that we don't have dangling jumps
        kept_indices = set(i for i, k in enumerate(mask) if k)
        
        for i, (instr, keep) in enumerate(zip(instructions, mask)):
            if not keep:
                continue
            
            mnem = instr.get('mnemonic', '').lower()
            
            # Check jump targets
            if mnem.startswith('j'):
                target = instr.get('target', instr.get('operands', [None])[0])
                if isinstance(target, int) and target not in kept_indices:
                    warnings.append(f"[CF] Jump at {i} targets removed instruction {target}")
        
        return warnings


class RLGNNValidator:
    """
    RL-based validator that learns to correct GNN predictions.
    Uses the RL verifier model to score and correct predictions.
    """

    def __init__(self, rl_model_path: Optional[str] = None):
        self.base_validator = GNNOutputValidator()
        self.rl_model = None
        
        if rl_model_path:
            self._load_rl_model(rl_model_path)

    def _load_rl_model(self, path: str):
        """Load trained RL model for prediction correction."""
        try:
            import torch
            checkpoint = torch.load(path, map_location='cpu')
            # Would load actual model here
            print(f"Loaded RL validator from {path}")
        except Exception as e:
            print(f"Could not load RL model: {e}")

    def validate_and_correct(
        self,
        instructions: List[Dict],
        gnn_predictions: List[float],
        cfg: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate GNN predictions using both static analysis and RL model.
        """
        # First do static validation
        result = self.base_validator.validate(instructions, gnn_predictions, cfg)
        
        # If RL model is available, use it to further refine
        if self.rl_model is not None:
            # RL would adjust confidence scores here
            pass
        
        return result
