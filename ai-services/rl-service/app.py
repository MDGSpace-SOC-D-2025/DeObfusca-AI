

from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import z3
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from pycparser import c_parser, c_ast, parse_file
    PYCPARSER_AVAILABLE = True
except ImportError:
    PYCPARSER_AVAILABLE = False
    print("Warning: pycparser not available, using pattern-based parsing")


try:
    from train_rl import (
        PPOTrainer, CodeStateEncoder, PolicyNetwork,
        DecompilationEnvironment, RewardShaper
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Warning: RL components not available")

app = Flask(__name__)



class NeuralSymbolicVerifier:
    
    def __init__(self):
        self.solver = z3.Solver()
        self.timeout_ms = 5000  # 5 second timeout
        self.solver.set("timeout", self.timeout_ms)
    
    def reset(self):
        self.solver.reset()
    
    def symbolic_execute(self, source_code: str, inputs: List[int]) -> Dict:
        self.reset()
        
        # Create symbolic variables
        symbolic_vars = {}
        for i, val in enumerate(inputs):
            symbolic_vars[f'input_{i}'] = z3.Int(f'input_{i}')
            self.solver.add(symbolic_vars[f'input_{i}'] == val)
        
        # Use full AST parser if available
        if PYCPARSER_AVAILABLE:
            return self._symbolic_execute_ast(source_code, symbolic_vars)
        else:
            return self._symbolic_execute_pattern(source_code, symbolic_vars)
    
    def _symbolic_execute_ast(self, source_code: str, symbolic_vars: Dict) -> Dict:
        try:
            parser = c_parser.CParser()
            
            # Wrap in minimal C structure if needed
            if 'int main' not in source_code and 'void ' not in source_code:
                wrapped_code = f"int main() {{ {source_code} return 0; }}"
            else:
                wrapped_code = source_code
            
            ast = parser.parse(wrapped_code)
            
            # Walk AST and build constraints
            local_vars = {}
            self._visit_ast_node(ast, local_vars, symbolic_vars)
            
            check_result = self.solver.check()
            
            return {
                'constraints': str(self.solver),
                'satisfiable': check_result == z3.sat,
                'model': str(self.solver.model()) if check_result == z3.sat else None,
                'variables': list(local_vars.keys()),
                'method': 'ast',
                'status': str(check_result)
            }
        except Exception as e:
            print(f"AST parsing failed: {e}, using pattern-based fallback")
            return self._symbolic_execute_pattern(source_code, symbolic_vars)
    
    def _visit_ast_node(self, node, local_vars: Dict, symbolic_vars: Dict):
        if node is None:
            return
            
        if isinstance(node, c_ast.Decl):
            var_name = node.name
            if var_name:
                local_vars[var_name] = z3.Int(var_name)
                
                if node.init:
                    if isinstance(node.init, c_ast.Constant):
                        try:
                            value = int(node.init.value)
                            self.solver.add(local_vars[var_name] == value)
                        except:
                            pass
        
        elif isinstance(node, c_ast.Assignment):
            if isinstance(node.lvalue, c_ast.ID):
                lhs = node.lvalue.name
                if lhs in local_vars:
                    rhs_expr = self._ast_to_z3(node.rvalue, local_vars, symbolic_vars)
                    if rhs_expr is not None:
                        # Create new version of variable for SSA-like handling
                        new_var = z3.Int(f'{lhs}_new')
                        self.solver.add(new_var == rhs_expr)
                        local_vars[lhs] = new_var
        
        elif isinstance(node, c_ast.If):
            cond_expr = self._ast_to_z3(node.cond, local_vars, symbolic_vars)
            if cond_expr is not None:
                # Add condition as constraint
                self.solver.add(cond_expr)
        
        elif isinstance(node, c_ast.For):
            # Handle for loop symbolically
            if node.init:
                self._visit_ast_node(node.init, local_vars, symbolic_vars)
            if node.cond:
                cond_expr = self._ast_to_z3(node.cond, local_vars, symbolic_vars)
                if cond_expr is not None:
                    self.solver.add(cond_expr)
        
        elif isinstance(node, c_ast.While):
            cond_expr = self._ast_to_z3(node.cond, local_vars, symbolic_vars)
            if cond_expr is not None:
                self.solver.add(cond_expr)
        
        elif isinstance(node, c_ast.Return):
            if node.expr:
                ret_expr = self._ast_to_z3(node.expr, local_vars, symbolic_vars)
                if ret_expr is not None:
                    ret_var = z3.Int('_return_')
                    self.solver.add(ret_var == ret_expr)
                    local_vars['_return_'] = ret_var
        
        # Recursively visit children
        for child_name, child in node.children():
            self._visit_ast_node(child, local_vars, symbolic_vars)
    
    def _ast_to_z3(self, node, local_vars: Dict, symbolic_vars: Dict):
        if node is None:
            return None
            
        if isinstance(node, c_ast.ID):
            name = node.name
            if name in local_vars:
                return local_vars[name]
            elif name in symbolic_vars:
                return symbolic_vars[name]
            return None
        
        elif isinstance(node, c_ast.Constant):
            try:
                if '.' in node.value:
                    return float(node.value)
                return int(node.value)
            except:
                return None
        
        elif isinstance(node, c_ast.BinaryOp):
            left = self._ast_to_z3(node.left, local_vars, symbolic_vars)
            right = self._ast_to_z3(node.right, local_vars, symbolic_vars)
            
            if left is None or right is None:
                return None
            
            op_map = {
                '+': lambda l, r: l + r,
                '-': lambda l, r: l - r,
                '*': lambda l, r: l * r,
                '/': lambda l, r: l / r,
                '%': lambda l, r: l % r,
                '<': lambda l, r: l < r,
                '>': lambda l, r: l > r,
                '<=': lambda l, r: l <= r,
                '>=': lambda l, r: l >= r,
                '==': lambda l, r: l == r,
                '!=': lambda l, r: l != r,
                '&&': lambda l, r: z3.And(l, r),
                '||': lambda l, r: z3.Or(l, r),
                '&': lambda l, r: l & r,
                '|': lambda l, r: l | r,
                '^': lambda l, r: l ^ r,
                '<<': lambda l, r: l << r,
                '>>': lambda l, r: l >> r,
            }
            
            if node.op in op_map:
                try:
                    return op_map[node.op](left, right)
                except:
                    return None
            return None
        
        elif isinstance(node, c_ast.UnaryOp):
            operand = self._ast_to_z3(node.expr, local_vars, symbolic_vars)
            if operand is None:
                return None
            
            if node.op == '-':
                return -operand
            elif node.op == '!':
                return z3.Not(operand)
            elif node.op == '~':
                return ~operand
            return None
        
        elif isinstance(node, c_ast.TernaryOp):
            cond = self._ast_to_z3(node.cond, local_vars, symbolic_vars)
            iftrue = self._ast_to_z3(node.iftrue, local_vars, symbolic_vars)
            iffalse = self._ast_to_z3(node.iffalse, local_vars, symbolic_vars)
            
            if cond is not None and iftrue is not None and iffalse is not None:
                return z3.If(cond, iftrue, iffalse)
            return None
        
        return None
    
    def _symbolic_execute_pattern(self, source_code: str, symbolic_vars: Dict) -> Dict:
        import re
        
        lines = source_code.strip().split('\n')
        local_vars = {}
        
        for line in lines:
            line = line.strip()
            
            # Variable declaration with initialization: int x = 5;
            decl_match = re.match(r'(int|long|short|char)\s+(\w+)\s*=\s*(.+?)\s*;', line)
            if decl_match:
                var_type, var_name, value_str = decl_match.groups()
                local_vars[var_name] = z3.Int(var_name)
                
                try:
                    value = int(value_str)
                    self.solver.add(local_vars[var_name] == value)
                except:
                    # Expression - try to parse
                    expr = self._parse_expression(value_str, local_vars, symbolic_vars)
                    if expr is not None:
                        self.solver.add(local_vars[var_name] == expr)
                continue
            
            # Plain declaration: int x;
            plain_decl = re.match(r'(int|long|short|char)\s+(\w+)\s*;', line)
            if plain_decl:
                var_type, var_name = plain_decl.groups()
                local_vars[var_name] = z3.Int(var_name)
                continue
            
            # Assignment: x = expr;
            assign_match = re.match(r'(\w+)\s*=\s*(.+?)\s*;', line)
            if assign_match:
                var_name, expr_str = assign_match.groups()
                if var_name in local_vars:
                    expr = self._parse_expression(expr_str, local_vars, symbolic_vars)
                    if expr is not None:
                        self.solver.add(local_vars[var_name] == expr)
                continue
            
            # Conditional: if (x > 10)
            if_match = re.match(r'if\s*\(\s*(.+?)\s*\)', line)
            if if_match:
                cond_str = if_match.group(1)
                cond = self._parse_condition(cond_str, local_vars, symbolic_vars)
                if cond is not None:
                    self.solver.add(cond)
                continue
        
        check_result = self.solver.check()
        
        return {
            'constraints': str(self.solver),
            'satisfiable': check_result == z3.sat,
            'model': str(self.solver.model()) if check_result == z3.sat else None,
            'variables': list(local_vars.keys()),
            'method': 'pattern',
            'status': str(check_result)
        }
    
    def _parse_expression(self, expr_str: str, local_vars: Dict, symbolic_vars: Dict):
        import re
        
        expr_str = expr_str.strip()
        
        # Handle simple values
        try:
            return int(expr_str)
        except:
            pass
        
        # Check if it's a variable
        if expr_str in local_vars:
            return local_vars[expr_str]
        if expr_str in symbolic_vars:
            return symbolic_vars[expr_str]
        
        # Parse binary operations
        for op, z3_op in [('+', lambda a, b: a + b), ('-', lambda a, b: a - b),
                          ('*', lambda a, b: a * b), ('/', lambda a, b: a / b)]:
            if op in expr_str:
                parts = expr_str.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expression(parts[0], local_vars, symbolic_vars)
                    right = self._parse_expression(parts[1], local_vars, symbolic_vars)
                    if left is not None and right is not None:
                        return z3_op(left, right)
        
        return None
    
    def _parse_condition(self, cond_str: str, local_vars: Dict, symbolic_vars: Dict):
        import re
        
        for op, z3_op in [('>=', lambda a, b: a >= b), ('<=', lambda a, b: a <= b),
                          ('==', lambda a, b: a == b), ('!=', lambda a, b: a != b),
                          ('>', lambda a, b: a > b), ('<', lambda a, b: a < b)]:
            if op in cond_str:
                parts = cond_str.split(op, 1)
                if len(parts) == 2:
                    left = self._parse_expression(parts[0].strip(), local_vars, symbolic_vars)
                    right = self._parse_expression(parts[1].strip(), local_vars, symbolic_vars)
                    if left is not None and right is not None:
                        return z3_op(left, right)
        
        return None
    
    def prove_equivalence(
        self,
        binary_outputs: List[int],
        decompiled_outputs: List[int],
        inputs: List[int]
    ) -> Dict:
        self.reset()
        
        # Quick check: if outputs don't match, not equivalent
        if binary_outputs != decompiled_outputs:
            mismatches = []
            for i, (b, d) in enumerate(zip(binary_outputs, decompiled_outputs)):
                if b != d:
                    mismatches.append({
                        'index': i,
                        'binary': b,
                        'decompiled': d,
                        'input': inputs[i] if i < len(inputs) else None
                    })
            
            return {
                'equivalent': False,
                'reason': f'Output mismatch at {len(mismatches)} positions',
                'counterexample': mismatches[0] if mismatches else None,
                'mismatches': mismatches[:5]  # First 5 mismatches
            }
        
        # Create symbolic inputs
        symbolic_inputs = [z3.Int(f'input_{i}') for i in range(len(inputs))]
        
        # Create symbolic outputs
        binary_out = z3.Int('binary_output')
        decompiled_out = z3.Int('decompiled_output')
        
        # Add constraints from observed behavior
        for i, (inp, b_out, d_out) in enumerate(zip(inputs, binary_outputs, decompiled_outputs)):
            # When input is inp, outputs are b_out and d_out
            self.solver.add(z3.Implies(
                symbolic_inputs[0] == inp,
                z3.And(binary_out == b_out, decompiled_out == d_out)
            ))
        
        # Try to find counterexample
        self.solver.push()
        self.solver.add(binary_out != decompiled_out)
        
        result = self.solver.check()
        self.solver.pop()
        
        if result == z3.sat:
            model = self.solver.model()
            counterexample = {}
            
            for i, inp in enumerate(symbolic_inputs):
                val = model[inp]
                counterexample[f'input_{i}'] = val.as_long() if val else None
            
            counterexample['binary_output'] = model[binary_out].as_long() if model[binary_out] else None
            counterexample['decompiled_output'] = model[decompiled_out].as_long() if model[decompiled_out] else None
            
            return {
                'equivalent': False,
                'reason': 'Z3 found counterexample',
                'counterexample': counterexample
            }
        elif result == z3.unsat:
            return {
                'equivalent': True,
                'reason': 'No counterexample exists (proven equivalent)',
                'counterexample': None
            }
        else:
            return {
                'equivalent': None,
                'reason': f'Z3 returned {result} (timeout or unknown)',
                'counterexample': None
            }



class ConfidenceCalibrator:
    
    def __init__(self, method: str = 'temperature'):
        self.method = method
        self.temperature = 1.5
        self.calibration_data = []
        self.bins = 10
    
    def calibrate_reward(self, raw_reward: float, confidence: float) -> float:
        scaled_confidence = confidence ** (1.0 / self.temperature)
        calibrated_reward = raw_reward * scaled_confidence
        calibrated_reward = 11.0 / (1.0 + np.exp(-0.5 * (calibrated_reward - 5.5)))
        return float(calibrated_reward)
    
    def update_calibration(self, predicted_confidence: float, actual_success: bool):
        self.calibration_data.append({
            'confidence': predicted_confidence,
            'success': actual_success
        })
        
        if len(self.calibration_data) >= 100:
            self._recompute_temperature()
    
    def _recompute_temperature(self):
        if not self.calibration_data:
            return
            
        confidences = [d['confidence'] for d in self.calibration_data]
        successes = [float(d['success']) for d in self.calibration_data]
        
        best_temp = 1.0
        best_error = float('inf')
        
        for temp in np.linspace(0.5, 3.0, 50):
            scaled_confs = [c ** (1.0 / temp) for c in confidences]
            error = np.mean([(c - s) ** 2 for c, s in zip(scaled_confs, successes)])
            
            if error < best_error:
                best_error = error
                best_temp = temp
        
        self.temperature = best_temp
    
    def get_stats(self) -> Dict:
        if not self.calibration_data:
            return {'error': 'No calibration data'}
        
        confidences = [d['confidence'] for d in self.calibration_data]
        successes = [d['success'] for d in self.calibration_data]
        
        return {
            'temperature': self.temperature,
            'num_samples': len(self.calibration_data),
            'avg_confidence': float(np.mean(confidences)),
            'avg_success_rate': float(np.mean(successes))
        }



def compile_source(source_code: str, timeout: int = 30) -> Dict:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(source_code)
            source_path = f.name
        
        binary_path = source_path.replace('.c', '.out')
        
        result = subprocess.run(
            ['gcc', '-O0', '-o', binary_path, source_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        os.unlink(source_path)
        
        if result.returncode == 0:
            return {
                'success': True,
                'binary_path': binary_path,
                'errors': []
            }
        else:
            return {
                'success': False,
                'binary_path': None,
                'errors': [result.stderr]
            }
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'binary_path': None, 'errors': ['Compilation timeout']}
    except Exception as e:
        return {'success': False, 'binary_path': None, 'errors': [str(e)]}


def run_binary(binary_path: str, input_data: str, timeout: int = 5) -> Optional[str]:
    try:
        result = subprocess.run(
            [binary_path],
            input=input_data,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout
    except:
        return None


def run_and_compare(decompiled_binary: str, original_binary: str, 
                    num_tests: int = 10) -> Dict:
    import random
    
    # Generate test inputs
    test_inputs = [str(random.randint(-1000, 1000)) for _ in range(num_tests)]
    
    binary_outputs = []
    decompiled_outputs = []
    matches = 0
    
    for test_input in test_inputs:
        decomp_output = run_binary(decompiled_binary, test_input)
        orig_output = run_binary(original_binary, test_input)
        
        try:
            binary_outputs.append(int(orig_output) if orig_output else 0)
            decompiled_outputs.append(int(decomp_output) if decomp_output else 0)
        except:
            binary_outputs.append(0)
            decompiled_outputs.append(0)
        
        if decomp_output == orig_output:
            matches += 1
    
    similarity = matches / num_tests
    
    return {
        'match': similarity >= 0.9,
        'similarity': similarity,
        'binary_outputs': binary_outputs,
        'decompiled_outputs': decompiled_outputs,
        'inputs': test_inputs
    }


def calculate_reward(
    compilation_success: bool,
    execution_match: bool,
    symbolic_equivalent: bool = False
) -> float:
    if not compilation_success:
        return -1.0
    
    reward = 0.5  # Base reward for compilation
    
    if execution_match:
        reward += 10.0
    
    if symbolic_equivalent:
        reward += 5.0
    
    return reward


def generate_feedback(
    compilation_success: bool,
    execution_match: bool,
    symbolic_result: Optional[Dict],
    test_inputs: List,
    binary_outputs: List,
    decompiled_outputs: List
) -> str:
    if not compilation_success:
        return "The code failed to compile. Fix syntax errors."
    
    if execution_match and (not symbolic_result or symbolic_result.get('equivalent')):
        return "Perfect! The decompiled code is functionally equivalent."
    
    feedback_parts = []
    
    if not execution_match and test_inputs:
        for i, (inp, expected, actual) in enumerate(zip(test_inputs, binary_outputs, decompiled_outputs)):
            if expected != actual:
                feedback_parts.append(
                    f"Mismatch at input={inp}: expected {expected}, got {actual}."
                )
                break
    
    if symbolic_result and not symbolic_result.get('equivalent'):
        if symbolic_result.get('counterexample'):
            ce = symbolic_result['counterexample']
            feedback_parts.append(f"Symbolic counterexample: {ce}")
        else:
            feedback_parts.append(f"Symbolic verification: {symbolic_result.get('reason')}")
    
    return " ".join(feedback_parts) if feedback_parts else "Review the logic."




symbolic_verifier = NeuralSymbolicVerifier()
confidence_calibrator = ConfidenceCalibrator(method='temperature')

# RL Agent (if available)
rl_agent = None
if RL_AVAILABLE:
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        state_encoder = CodeStateEncoder()
        policy = PolicyNetwork()
        rl_agent = {
            'encoder': state_encoder.to(device),
            'policy': policy.to(device),
            'device': device
        }
        
        # Try to load checkpoint
        checkpoint_path = os.getenv('RL_CHECKPOINT', './checkpoints/final_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            state_encoder.load_state_dict(checkpoint['state_encoder'])
            policy.load_state_dict(checkpoint['policy'])
            print(f"Loaded RL checkpoint from {checkpoint_path}")
    except Exception as e:
        print(f"Failed to initialize RL agent: {e}")
        rl_agent = None




@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'neural-symbolic-verification',
        'pycparser_available': PYCPARSER_AVAILABLE,
        'rl_available': RL_AVAILABLE and rl_agent is not None,
        'calibration': confidence_calibrator.get_stats()
    })


@app.route('/verify', methods=['POST'])
def verify():
    try:
        data = request.json
        source_code = data.get('source_code', '')
        original_binary = data.get('original_binary_path')
        use_symbolic = data.get('use_symbolic', False)
        
        # Input validation
        if not source_code:
            return jsonify({'error': 'source_code required'}), 400
        if len(source_code) > 1000000:
            return jsonify({'error': 'source_code too large (max 1MB)'}), 400
        
        # Step 1: Compile
        compile_result = compile_source(source_code)
        
        if not compile_result['success']:
            return jsonify({
                'compilation_success': False,
                'execution_match': False,
                'symbolic_equivalent': False,
                'reward': -1.0,
                'errors': compile_result['errors'],
                'feedback': f"Compilation failed: {compile_result['errors'][0]}"
            })
        
        # Step 2: Execute and compare
        execution_match = False
        binary_outputs = []
        decompiled_outputs = []
        test_inputs = []
        
        if original_binary and os.path.exists(original_binary):
            exec_result = run_and_compare(
                compile_result['binary_path'],
                original_binary
            )
            execution_match = exec_result['match']
            binary_outputs = exec_result.get('binary_outputs', [])
            decompiled_outputs = exec_result.get('decompiled_outputs', [])
            test_inputs = exec_result.get('inputs', [])
        
        # Step 3: Symbolic verification
        symbolic_equivalent = False
        symbolic_result = None
        
        if use_symbolic and binary_outputs and decompiled_outputs:
            symbolic_result = symbolic_verifier.prove_equivalence(
                binary_outputs,
                decompiled_outputs,
                [int(i) for i in test_inputs]
            )
            symbolic_equivalent = symbolic_result.get('equivalent', False) or False
        
        # Step 4: Generate feedback
        feedback = generate_feedback(
            compilation_success=True,
            execution_match=execution_match,
            symbolic_result=symbolic_result,
            test_inputs=test_inputs,
            binary_outputs=binary_outputs,
            decompiled_outputs=decompiled_outputs
        )
        
        # Step 5: Calculate reward
        raw_reward = calculate_reward(
            compilation_success=True,
            execution_match=execution_match,
            symbolic_equivalent=symbolic_equivalent
        )
        
        # Step 6: Calibrate
        confidence = 0.8 if execution_match else 0.5
        calibrated_reward = confidence_calibrator.calibrate_reward(raw_reward, confidence)
        
        # Step 7: Update calibration
        verification_success = execution_match and (symbolic_equivalent or not use_symbolic)
        confidence_calibrator.update_calibration(confidence, verification_success)
        
        # Clean up binary
        try:
            os.unlink(compile_result['binary_path'])
        except:
            pass
        
        return jsonify({
            'compilation_success': True,
            'execution_match': execution_match,
            'symbolic_equivalent': symbolic_equivalent,
            'reward': float(calibrated_reward),
            'raw_reward': float(raw_reward),
            'confidence': confidence,
            'errors': [],
            'feedback': feedback,
            'symbolic_details': symbolic_result
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/symbolic-execute', methods=['POST'])
def symbolic_execute():
    try:
        data = request.json
        source_code = data.get('source_code', '')
        inputs = data.get('inputs', [])
        
        if not source_code:
            return jsonify({'error': 'source_code required'}), 400
        
        result = symbolic_verifier.symbolic_execute(source_code, inputs)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/rl-refine', methods=['POST'])
def rl_refine():
    try:
        if not rl_agent:
            return jsonify({'error': 'RL agent not available'}), 503
        
        data = request.json
        source_code = data.get('source_code', '')
        features = data.get('features', [])
        
        # Encode state
        tokens = []
        for feat in features[:256]:
            if isinstance(feat, dict):
                mnemonic = feat.get('mnemonic', 'NOP')
                tokens.append(hash(mnemonic) % 1024)
            else:
                tokens.append(hash(str(feat)) % 1024)
        
        for char in source_code[:256]:
            tokens.append(ord(char) % 1024)
        
        while len(tokens) < 512:
            tokens.append(0)
        
        state = torch.tensor(tokens[:512], dtype=torch.long).unsqueeze(0)
        state = state.to(rl_agent['device'])
        
        # Get action
        with torch.no_grad():
            state_emb = rl_agent['encoder'](state)
            action_logits, value = rl_agent['policy'](state_emb)
            action = action_logits.argmax(dim=-1).item()
            confidence = torch.softmax(action_logits, dim=-1).max().item()
        
        action_name = PolicyNetwork.ACTIONS[action]
        
        return jsonify({
            'suggested_action': action_name,
            'action_index': action,
            'confidence': float(confidence),
            'value_estimate': float(value.item()),
            'all_actions': PolicyNetwork.ACTIONS
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/calibration', methods=['GET'])
def get_calibration():
    return jsonify(confidence_calibrator.get_stats())


if __name__ == '__main__':
    print("Starting Neural-Symbolic Verification Service")
    print(f"  pycparser: {'available' if PYCPARSER_AVAILABLE else 'not available'}")
    print(f"  RL agent: {'available' if rl_agent else 'not available'}")
    
    app.run(host='0.0.0.0', port=5004, debug=True)
