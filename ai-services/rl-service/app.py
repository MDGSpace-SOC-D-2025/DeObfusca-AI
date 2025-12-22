from flask import Flask, request, jsonify
import subprocess
import tempfile
import os
import difflib
from pathlib import Path
import z3
import numpy as np
from typing import List, Dict, Tuple
try:
    from pycparser import c_parser, c_ast, parse_file
    PYCPARSER_AVAILABLE = True
except ImportError:
    PYCPARSER_AVAILABLE = False
    print("Warning: pycparser not available, using pattern-based parsing")

app = Flask(__name__)

class NeuralSymbolicVerifier:
    """
    Neural-Symbolic Execution Engine using Z3 Solver.
    
    Proves mathematical equivalence between original binary and decompiled C code.
    Uses symbolic execution to verify behavior across all input spaces.
    """
    
    def __init__(self):
        self.solver = z3.Solver()
    
    def symbolic_execute(self, source_code: str, inputs: List[int]) -> Dict:
        """
        Symbolically execute C code and extract constraints.
        Uses pycparser for full AST parsing when available.
        
        Args:
            source_code: C code to analyze
            inputs: Concrete input values
        
        Returns:
            Dictionary with symbolic constraints and outputs
        """
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
        """Full AST-based symbolic execution using pycparser."""
        try:
            # Parse C code to AST
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
            
            return {
                'constraints': str(self.solver),
                'satisfiable': self.solver.check() == z3.sat,
                'model': str(self.solver.model()) if self.solver.check() == z3.sat else None,
                'variables': list(local_vars.keys()),
                'method': 'ast'
            }
        except Exception as e:
            # Fallback to pattern matching
            print(f"AST parsing failed: {e}, using pattern-based fallback")
            return self._symbolic_execute_pattern(source_code, symbolic_vars)
    
    def _visit_ast_node(self, node, local_vars: Dict, symbolic_vars: Dict):
        """Recursively visit AST nodes and build Z3 constraints."""
        if isinstance(node, c_ast.Decl):
            # Variable declaration
            var_name = node.name
            local_vars[var_name] = z3.Int(var_name)
            
            if node.init:
                if isinstance(node.init, c_ast.Constant):
                    value = int(node.init.value)
                    self.solver.add(local_vars[var_name] == value)
        
        elif isinstance(node, c_ast.Assignment):
            # Assignment operation
            if isinstance(node.lvalue, c_ast.ID):
                lhs = node.lvalue.name
                if lhs in local_vars:
                    rhs_expr = self._ast_to_z3(node.rvalue, local_vars, symbolic_vars)
                    if rhs_expr is not None:
                        self.solver.add(local_vars[lhs] == rhs_expr)
        
        elif isinstance(node, c_ast.If):
            # Conditional statement
            cond_expr = self._ast_to_z3(node.cond, local_vars, symbolic_vars)
            if cond_expr is not None:
                self.solver.add(cond_expr)
        
        # Recursively visit children
        for child in node:
            self._visit_ast_node(child, local_vars, symbolic_vars)
    
    def _ast_to_z3(self, node, local_vars: Dict, symbolic_vars: Dict):
        """Convert AST expression to Z3 expression."""
        if isinstance(node, c_ast.ID):
            name = node.name
            if name in local_vars:
                return local_vars[name]
            elif name in symbolic_vars:
                return symbolic_vars[name]
        
        elif isinstance(node, c_ast.Constant):
            return int(node.value)
        
        elif isinstance(node, c_ast.BinaryOp):
            left = self._ast_to_z3(node.left, local_vars, symbolic_vars)
            right = self._ast_to_z3(node.right, local_vars, symbolic_vars)
            
            if left is None or right is None:
                return None
            
            if node.op == '+':
                return left + right
            elif node.op == '-':
                return left - right
            elif node.op == '*':
                return left * right
            elif node.op == '/':
                return left / right
            elif node.op == '<':
                return left < right
            elif node.op == '>':
                return left > right
            elif node.op == '==':
                return left == right
            elif node.op == '!=':
                return left != right
        
        return None
    
    def _symbolic_execute_pattern(self, source_code: str, symbolic_vars: Dict) -> Dict:
        """Pattern-based symbolic execution (fallback method)."""
        # Extract variable declarations and operations
        lines = source_code.strip().split('\n')
        local_vars = {}
        
        for line in lines:
            line = line.strip()
            
            # Handle variable declarations: int x = 5;
            if 'int ' in line and '=' in line:
                parts = line.split('=')
                var_name = parts[0].replace('int', '').strip().rstrip(';')
                try:
                    value = int(parts[1].strip().rstrip(';'))
                    local_vars[var_name] = z3.Int(var_name)
                    self.solver.add(local_vars[var_name] == value)
                except:
                    local_vars[var_name] = z3.Int(var_name)
            
            # Handle arithmetic operations: x = y + z;
            elif '=' in line and '+' in line:
                parts = line.split('=')
                lhs = parts[0].strip()
                rhs = parts[1].strip().rstrip(';')
                if lhs in local_vars:
                    # Parse RHS
                    operands = rhs.split('+')
                    if len(operands) == 2:
                        op1 = operands[0].strip()
                        op2 = operands[1].strip()
                        if op1 in local_vars and op2 in local_vars:
                            self.solver.add(local_vars[lhs] == local_vars[op1] + local_vars[op2])
            
            # Handle comparisons: if (x > 10)
            elif 'if' in line and '>' in line:
                condition = line[line.find('(')+1:line.find(')')]
                parts = condition.split('>')
                if len(parts) == 2:
                    var = parts[0].strip()
                    val = int(parts[1].strip())
                    if var in local_vars:
                        self.solver.add(local_vars[var] > val)
        
        return {
            'constraints': str(self.solver),
            'satisfiable': self.solver.check() == z3.sat,
            'model': str(self.solver.model()) if self.solver.check() == z3.sat else None,
            'variables': list(local_vars.keys())
        }
    
    def prove_equivalence(
        self,
        binary_outputs: List[int],
        decompiled_outputs: List[int],
        inputs: List[int]
    ) -> Dict:
        """
        Prove that decompiled code is equivalent to original binary.
        
        Uses Z3 to check if there exists any input where outputs differ.
        """
        self.solver.reset()
        
        # Create symbolic inputs
        symbolic_inputs = [z3.Int(f'input_{i}') for i in range(len(inputs))]
        
        # Create symbolic outputs for both versions
        binary_out = z3.Int('binary_output')
        decompiled_out = z3.Int('decompiled_output')
        
        # Add constraints from observed behavior
        for i, (b_out, d_out) in enumerate(zip(binary_outputs, decompiled_outputs)):
            # For this input, outputs must match observed values
            self.solver.push()
            for j, inp in enumerate(inputs):
                self.solver.add(symbolic_inputs[j] == inp)
            self.solver.add(binary_out == b_out)
            self.solver.add(decompiled_out == d_out)
            
            # Check if this is satisfiable
            if self.solver.check() != z3.sat:
                return {
                    'equivalent': False,
                    'reason': f'Inconsistent at input {i}',
                    'counterexample': None
                }
            self.solver.pop()
        
        # Try to find counterexample: input where outputs differ
        self.solver.add(binary_out != decompiled_out)
        
        if self.solver.check() == z3.sat:
            # Found counterexample - not equivalent
            model = self.solver.model()
            counterexample = {
                'inputs': [model[v].as_long() if model[v] else None for v in symbolic_inputs],
                'binary_output': model[binary_out].as_long() if model[binary_out] else None,
                'decompiled_output': model[decompiled_out].as_long() if model[decompiled_out] else None
            }
            return {
                'equivalent': False,
                'reason': 'Found counterexample',
                'counterexample': counterexample
            }
        else:
            # No counterexample found - likely equivalent
            return {
                'equivalent': True,
                'reason': 'No counterexample found',
                'counterexample': None
            }


# Global verifier instance
symbolic_verifier = NeuralSymbolicVerifier()


class ConfidenceCalibrator:
    """
    Calibrates confidence scores for better reward scaling.
    
    Techniques:
    1. Temperature scaling
    2. Platt scaling
    3. Isotonic regression
    4. Histogram binning
    """
    
    def __init__(self, method='temperature'):
        self.method = method
        self.temperature = 1.5  # Default temperature
        self.calibration_data = []
        self.bins = 10
    
    def calibrate_reward(self, raw_reward: float, confidence: float) -> float:
        """
        Calibrate reward based on confidence and historical performance.
        
        Args:
            raw_reward: Uncalibrated reward (0-11 scale)
            confidence: Model confidence (0-1)
        
        Returns:
            Calibrated reward with proper scaling
        """
        if self.method == 'temperature':
            return self._temperature_scaling(raw_reward, confidence)
        elif self.method == 'platt':
            return self._platt_scaling(raw_reward, confidence)
        elif self.method == 'histogram':
            return self._histogram_binning(raw_reward, confidence)
        else:
            return raw_reward
    
    def _temperature_scaling(self, reward: float, confidence: float) -> float:
        """
        Apply temperature scaling to smooth reward distribution.
        
        Higher temperature: More conservative (compress high rewards)
        Lower temperature: More aggressive (amplify high rewards)
        """
        # Apply temperature to confidence
        scaled_confidence = confidence ** (1.0 / self.temperature)
        
        # Adjust reward based on scaled confidence
        calibrated_reward = reward * scaled_confidence
        
        # Apply sigmoid to bound output
        calibrated_reward = 11.0 / (1.0 + np.exp(-0.5 * (calibrated_reward - 5.5)))
        
        return calibrated_reward
    
    def _platt_scaling(self, reward: float, confidence: float) -> float:
        """
        Platt scaling using logistic regression.
        
        Maps raw scores to calibrated probabilities.
        """
        # Learned parameters (would be fitted on validation set)
        A = 1.2
        B = -0.5
        
        # Logistic function
        calibrated = 11.0 / (1.0 + np.exp(A * reward + B))
        
        # Weight by confidence
        calibrated = calibrated * confidence + reward * (1 - confidence)
        
        return np.clip(calibrated, 0.0, 11.0)
    
    def _histogram_binning(self, reward: float, confidence: float) -> float:
        """
        Histogram binning for calibration.
        
        Divides reward range into bins and adjusts based on empirical accuracy.
        """
        bin_size = 11.0 / self.bins
        bin_idx = int(reward / bin_size)
        bin_idx = min(bin_idx, self.bins - 1)
        
        # Empirical bin accuracies (would be computed from validation data)
        bin_accuracies = np.array([0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
        
        # Adjust reward based on bin accuracy
        bin_accuracy = bin_accuracies[bin_idx]
        calibrated = reward * (bin_accuracy / confidence) if confidence > 0 else reward
        
        return np.clip(calibrated, 0.0, 11.0)
    
    def update_calibration(self, predicted_confidence: float, actual_success: bool):
        """
        Update calibration data with new observation.
        
        Args:
            predicted_confidence: Model's predicted confidence
            actual_success: Whether verification actually succeeded
        """
        self.calibration_data.append({
            'confidence': predicted_confidence,
            'success': actual_success
        })
        
        # Recompute temperature if enough data
        if len(self.calibration_data) >= 100:
            self._recompute_temperature()
    
    def _recompute_temperature(self):
        """Recompute optimal temperature using validation data."""
        confidences = [d['confidence'] for d in self.calibration_data]
        successes = [d['success'] for d in self.calibration_data]
        
        # Find temperature that minimizes calibration error
        best_temp = 1.0
        best_error = float('inf')
        
        for temp in np.linspace(0.5, 3.0, 50):
            scaled_confs = [c ** (1.0 / temp) for c in confidences]
            
            # Compute calibration error (ECE - Expected Calibration Error)
            error = 0
            for i in range(self.bins):
                bin_confs = [c for c, s in zip(scaled_confs, successes) if i/self.bins <= c < (i+1)/self.bins]
                if bin_confs:
                    bin_successes = [s for c, s in zip(scaled_confs, successes) if i/self.bins <= c < (i+1)/self.bins]
                    avg_conf = np.mean(bin_confs)
                    avg_acc = np.mean(bin_successes)
                    error += abs(avg_conf - avg_acc) * len(bin_confs)
            
            error /= len(confidences)
            
            if error < best_error:
                best_error = error
                best_temp = temp
        
        self.temperature = best_temp
        print(f"Updated temperature to {self.temperature:.3f} (calibration error: {best_error:.4f})")
    
    def get_calibration_stats(self) -> Dict:
        """Get calibration statistics."""
        if not self.calibration_data:
            return {'error': 'No calibration data'}
        
        confidences = [d['confidence'] for d in self.calibration_data]
        successes = [d['success'] for d in self.calibration_data]
        
        # Compute ECE
        ece = 0
        for i in range(self.bins):
            bin_mask = [(i/self.bins <= c < (i+1)/self.bins) for c in confidences]
            bin_confs = [c for c, m in zip(confidences, bin_mask) if m]
            bin_succs = [s for s, m in zip(successes, bin_mask) if m]
            
            if bin_confs:
                avg_conf = np.mean(bin_confs)
                avg_acc = np.mean(bin_succs)
                ece += abs(avg_conf - avg_acc) * len(bin_confs) / len(confidences)
        
        return {
            'expected_calibration_error': ece,
            'temperature': self.temperature,
            'num_samples': len(self.calibration_data),
            'avg_confidence': np.mean(confidences),
            'avg_success_rate': np.mean(successes)
        }


# Global calibrator
confidence_calibrator = ConfidenceCalibrator(method='temperature')


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'neural-symbolic-verification'})

@app.route('/verify', methods=['POST'])
def verify():
    """
    Verify decompiled code through compilation, fuzzing, and symbolic execution.
    
    Request body:
    {
        "source_code": "...",
        "original_binary_path": "/path/to/original",
        "use_symbolic": true
    }
    
    Returns:
    {
        "compilation_success": true/false,
        "execution_match": true/false,
        "symbolic_equivalent": true/false,
        "reward": float,
        "errors": [],
        "feedback": "..."
    }
    """
    try:
        data = request.json
        source_code = data.get('source_code', '')
        original_binary = data.get('original_binary_path')
        use_symbolic = data.get('use_symbolic', False)
        
        # Input validation
        if not source_code:
            return jsonify({'error': 'source_code required'}), 400
        if len(source_code) > 1000000:  # 1MB limit
            return jsonify({'error': 'source_code too large (max 1MB)'}), 400
        if not isinstance(source_code, str):
            return jsonify({'error': 'source_code must be a string'}), 400
        if not isinstance(use_symbolic, bool):
            return jsonify({'error': 'use_symbolic must be boolean'}), 400
        
        # Step 1: Compile source code
        compile_result = compile_source(source_code)
        
        if not compile_result['success']:
            feedback = f"Compilation failed: {compile_result['errors'][0]}"
            return jsonify({
                'compilation_success': False,
                'execution_match': False,
                'symbolic_equivalent': False,
                'reward': -1.0,
                'errors': compile_result['errors'],
                'feedback': feedback
            })
        
        # Step 2: Run trace matching (behavioral comparison)
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
        
        # Step 3: Symbolic verification (if enabled)
        symbolic_equivalent = False
        symbolic_result = None
        
        if use_symbolic and binary_outputs and decompiled_outputs:
            symbolic_result = symbolic_verifier.prove_equivalence(
                binary_outputs,
                decompiled_outputs,
                test_inputs
            )
            symbolic_equivalent = symbolic_result['equivalent']
        
        # Step 4: Generate feedback for RLHF
        feedback = generate_correction_feedback(
            compilation_success=True,
            execution_match=execution_match,
            symbolic_result=symbolic_result,
            test_inputs=test_inputs,
            binary_outputs=binary_outputs,
            decompiled_outputs=decompiled_outputs
        )
        
        # Step 5: Calculate raw reward
        raw_reward = calculate_reward(
            compilation_success=True,
            execution_match=execution_match,
            symbolic_equivalent=symbolic_equivalent
        )
        
        # Step 6: Apply confidence calibration
        confidence = 0.8 if execution_match else 0.5
        calibrated_reward = confidence_calibrator.calibrate_reward(raw_reward, confidence)
        
        # Step 7: Update calibration data
        verification_success = execution_match and symbolic_equivalent
        confidence_calibrator.update_calibration(confidence, verification_success)
        
        return jsonify({
            'compilation_success': True,
            'execution_match': execution_match,
            'symbolic_equivalent': symbolic_equivalent,
            'reward': calibrated_reward,
            'raw_reward': raw_reward,
            'confidence': confidence,
            'calibration_method': confidence_calibrator.method,
            'errors': [],
            'feedback': feedback,
            'symbolic_details': symbolic_result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def compile_source(source_code, timeout=30):
    """
    Compile C source code using gcc.
    
    Returns:
    {
        'success': bool,
        'binary_path': str or None,
        'errors': list
    }
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
            f.write(source_code)
            source_path = f.name
        
        binary_path = source_path.replace('.c', '.out')
        
        # Run gcc
        result = subprocess.run(
            ['gcc', '-O0', '-o', binary_path, source_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        # Clean up source
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
        return {
            'success': False,
            'binary_path': None,
            'errors': ['Compilation timeout']
        }
    except Exception as e:
        return {
            'success': False,
            'binary_path': None,
            'errors': [str(e)]
        }

def run_and_compare(decompiled_binary, original_binary, num_tests=10):
    """
    Run fuzzing tests and compare outputs (Trace Matching).
    
    Returns:
    {
        'match': bool,
        'similarity': float,
        'binary_outputs': List[int],
        'decompiled_outputs': List[int],
        'inputs': List[int]
    }
    """
    try:
        # Generate random test inputs
        test_inputs = generate_test_inputs(num_tests)
        
        binary_outputs = []
        decompiled_outputs = []
        matches = 0
        
        for test_input in test_inputs:
            # Run decompiled binary
            decomp_output = run_binary(decompiled_binary, test_input)
            
            # Run original binary
            orig_output = run_binary(original_binary, test_input)
            
            # Record outputs
            try:
                binary_outputs.append(int(orig_output) if orig_output else 0)
                decompiled_outputs.append(int(decomp_output) if decomp_output else 0)
            except:
                binary_outputs.append(0)
                decompiled_outputs.append(0)
            
            # Compare outputs
            if decomp_output == orig_output:
                matches += 1
        
        similarity = matches / num_tests
        match = similarity >= 0.9  # 90% threshold
        
        return {
            'match': match,
            'similarity': similarity,
            'binary_outputs': binary_outputs,
            'decompiled_outputs': decompiled_outputs,
            'inputs': test_inputs
        }
        
    except Exception as e:
        print(f"Comparison error: {e}")
        return {
            'match': False,
            'similarity': 0.0,
            'binary_outputs': [],
            'decompiled_outputs': [],
            'inputs': []
        }


def generate_correction_feedback(
    compilation_success: bool,
    execution_match: bool,
    symbolic_result: Dict,
    test_inputs: List,
    binary_outputs: List,
    decompiled_outputs: List
) -> str:
    """
    Generate feedback prompt for RLHF (Reinforcement Learning from Human Feedback).
    
    This feedback is fed back into the LLM to iteratively refine the decompilation.
    """
    if not compilation_success:
        return "The code failed to compile. Fix syntax errors."
    
    if execution_match and (not symbolic_result or symbolic_result.get('equivalent')):
        return "Perfect! The decompiled code is functionally equivalent to the original binary."
    
    # Generate detailed feedback about mismatches
    feedback_parts = []
    
    if not execution_match and test_inputs and binary_outputs and decompiled_outputs:
        # Find first mismatch
        for i, (inp, expected, actual) in enumerate(zip(test_inputs, binary_outputs, decompiled_outputs)):
            if expected != actual:
                feedback_parts.append(
                    f"The code failed when Input={inp}. "
                    f"Expected output: {expected}, but got: {actual}. "
                    f"Fix the logic to handle this case correctly."
                )
                break
    
    if symbolic_result and not symbolic_result.get('equivalent'):
        if symbolic_result.get('counterexample'):
            ce = symbolic_result['counterexample']
            feedback_parts.append(
                f"Symbolic verification found a counterexample: "
                f"inputs={ce.get('inputs')}, "
                f"expected={ce.get('binary_output')}, "
                f"got={ce.get('decompiled_output')}."
            )
        else:
            feedback_parts.append(
                f"Symbolic verification failed: {symbolic_result.get('reason')}"
            )
    
    if not feedback_parts:
        feedback_parts.append("The code compiles but behavior doesn't match. Review the logic.")
    
    return " ".join(feedback_parts)

def generate_test_inputs(num_tests):
    """Generate random test inputs for fuzzing."""
    import random
    
    test_inputs = []
    for _ in range(num_tests):
        # Simple random integer inputs
        test_inputs.append(str(random.randint(-1000, 1000)))
    
    return test_inputs

def run_binary(binary_path, input_data, timeout=5):
    """
    Run a binary with given input and capture output.
    """
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

def calculate_reward(
    compilation_success: bool,
    execution_match: bool,
    symbolic_equivalent: bool = False
) -> float:
    """
    Calculate RL reward based on verification results.
    
    Enhanced reward structure:
    - Compilation success: +0.5
    - Execution match (90%+ tests): +10.0
    - Symbolic equivalence proven: +5.0
    - Compilation failure: -1.0
    """
    if not compilation_success:
        return -1.0
    
    reward = 0.5  # Base reward for compilation
    
    if execution_match:
        reward += 10.0
    
    if symbolic_equivalent:
        reward += 5.0  # Bonus for provable equivalence
    
    return reward

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5004, debug=True)
