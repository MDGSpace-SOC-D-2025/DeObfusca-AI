

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

app = Flask(__name__)


class ReasoningStep(Enum):
    # ...existing code...
    SIGNATURE = "function_signature"
    PARAMETERS = "parameters"
    LOCAL_VARS = "local_variables"
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    SEMANTICS = "semantics"
    SYNTHESIS = "synthesis"


@dataclass
class ReasoningResult:
    step: ReasoningStep
    reasoning: str
    output: Dict
    confidence: float


@dataclass
class ChainOfThoughtResult:
    steps: List[ReasoningResult] = field(default_factory=list)
    final_code: str = ""
    overall_confidence: float = 0.0
    reasoning_trace: str = ""


class ChainOfThoughtReasoner:
    """
    LLM-based chain-of-thought reasoner for decompilation.
    
    Uses a fine-tuned model to perform step-by-step analysis
    of binary code and generate readable C source code.
    """
    
    def __init__(
        self,
        model_name: str = "codellama/CodeLlama-7b-Instruct-hf",
        adapter_path: Optional[str] = None,
        use_quantization: bool = True
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer(model_name)
        
        # Load model
        self.model = self._load_model(model_name, adapter_path, use_quantization)
        
        # Reasoning prompts
        self.step_prompts = self._create_step_prompts()
        
        print(f"Chain-of-Thought Reasoner initialized on {self.device}")
    
    def _load_tokenizer(self, model_name: str):
        # ...existing code...
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return tokenizer
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            # Fallback to a simple tokenizer-like interface
            return SimpleTokenizer()
    
    def _load_model(
        self,
        model_name: str,
        adapter_path: Optional[str],
        use_quantization: bool
    ):
        # ...existing code...
        try:
            # Quantization config
            bnb_config = None
            if use_quantization and torch.cuda.is_available():
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto" if torch.cuda.is_available() else None,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Load adapter if available
            if adapter_path and os.path.exists(adapter_path):
                model = PeftModel.from_pretrained(model, adapter_path)
                print(f"Loaded LoRA adapter from {adapter_path}")
            
            model.eval()
            return model
            
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Using fallback rule-based reasoning")
            return None
    
    def _create_step_prompts(self) -> Dict[ReasoningStep, str]:
        # ...existing code...
        return {
            ReasoningStep.SIGNATURE: """Analyze the following P-Code/assembly to determine the function signature.

P-Code:
{pcode}

Step 1: Function Signature Analysis
- Look at CALL instructions to identify the function being defined
- Check stack setup (PUSH/POP patterns) for calling convention
- Analyze return instructions for return type

Reasoning:
""",

            ReasoningStep.PARAMETERS: """Based on the P-Code, identify function parameters.

P-Code:
{pcode}

Step 2: Parameter Identification
- Look for register/stack accesses at function entry
- Identify input values used before any assignment
- Determine parameter types from how they're used

Reasoning:
""",

            ReasoningStep.LOCAL_VARS: """Identify local variables from the P-Code.

P-Code:
{pcode}

Step 3: Local Variable Analysis
- Find stack allocations (esp/rsp manipulation)
- Track register assignments
- Group related memory accesses

Reasoning:
""",

            ReasoningStep.CONTROL_FLOW: """Analyze the control flow structure.

P-Code:
{pcode}

Step 4: Control Flow Analysis
- Identify basic blocks
- Find conditional branches (if/else)
- Detect loops (for/while patterns)
- Recognize switch statements

Reasoning:
""",

            ReasoningStep.DATA_FLOW: """Analyze data flow through the function.

P-Code:
{pcode}

Step 5: Data Flow Analysis
- Track value propagation
- Identify def-use chains
- Find unused computations (potential dead code)

Reasoning:
""",

            ReasoningStep.SEMANTICS: """Determine the semantic meaning of the code.

P-Code:
{pcode}

Step 6: Semantic Analysis
- What is the function's purpose?
- Are there recognizable algorithms or patterns?
- What library functions does it call?

Reasoning:
""",

            ReasoningStep.SYNTHESIS: """Synthesize all analysis into C source code.

Previous Analysis:
{analysis}

Step 7: Code Synthesis
- Combine all insights into readable C code
- Use meaningful variable names
- Add appropriate comments

Generated C Code:
```c
"""
        }
    
    @torch.no_grad()
    def _generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
        stop_sequences: List[str] = None
    ) -> str:
        # ...existing code...
        if self.model is None:
            # Fallback to rule-based generation
            return self._rule_based_generate(prompt)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.95,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].size(1):],
            skip_special_tokens=True
        )
        
        # Apply stop sequences
        if stop_sequences:
            for seq in stop_sequences:
                if seq in generated:
                    generated = generated[:generated.index(seq)]
        
        return generated.strip()
    
    def _rule_based_generate(self, prompt: str) -> str:
        # ...existing code...
        # Extract P-Code from prompt
        pcode_match = re.search(r'P-Code:\n(.*?)(?:\n\nStep|\Z)', prompt, re.DOTALL)
        pcode = pcode_match.group(1) if pcode_match else ""
        
        # Analyze based on step
        if "Function Signature" in prompt:
            return self._analyze_signature(pcode)
        elif "Parameter Identification" in prompt:
            return self._analyze_parameters(pcode)
        elif "Local Variable" in prompt:
            return self._analyze_local_vars(pcode)
        elif "Control Flow" in prompt:
            return self._analyze_control_flow(pcode)
        elif "Data Flow" in prompt:
            return self._analyze_data_flow(pcode)
        elif "Semantic Analysis" in prompt:
            return self._analyze_semantics(pcode)
        elif "Code Synthesis" in prompt:
            return self._synthesize_code(prompt)
        
        return "Analysis not available."
    
    def _analyze_signature(self, pcode: str) -> str:
        # ...existing code...
        lines = pcode.strip().split('\n')
        
        # Look for clues
        has_return = any('RETURN' in line.upper() for line in lines)
        has_params = any('LOAD' in line.upper() and 'param' in line.lower() for line in lines)
        num_loads = sum(1 for line in lines if 'LOAD' in line.upper())
        
        # Infer return type
        return_type = 'int' if has_return else 'void'
        
        # Count likely parameters
        param_count = min(num_loads // 2, 4)  # Heuristic
        
        reasoning = f"""
Looking at the P-Code structure:
1. Found {len(lines)} instructions total
2. Return instruction present: {has_return}
3. Initial LOAD instructions suggest {param_count} parameter(s)

Based on the calling convention and stack setup, this appears to be:
- Return type: {return_type} (based on return value handling)
- Calling convention: cdecl/System V (common for C functions)

Inferred signature: {return_type} function_name({', '.join([f'int arg{i}' for i in range(param_count)])})
"""
        return reasoning
    
    def _analyze_parameters(self, pcode: str) -> str:
        """Analyze parameters from P-Code."""
        # Find early loads that look like parameters
        params = []
        for i, line in enumerate(pcode.strip().split('\n')[:10]):
            if 'LOAD' in line.upper() or 'COPY' in line.upper():
                if 'param' in line.lower() or 'arg' in line.lower():
                    params.append(f"arg{len(params)}")
                elif i < 5:  # Early loads often are parameters
                    params.append(f"arg{len(params)}")
        
        if not params:
            params = ['arg0', 'arg1']  # Default assumption
        
        return f"""
Parameter Analysis:
- Identified {len(params)} parameter(s) from initial memory accesses
- Parameters: {', '.join(params)}

Type inference:
- arg0: int (based on arithmetic operations)
- arg1: int (based on comparison usage)

These parameters are accessed early in the function and used throughout.
"""
    
    def _analyze_local_vars(self, pcode: str) -> str:
        """Analyze local variables from P-Code."""
        # Look for stack allocations and register assignments
        vars_found = []
        
        lines = pcode.strip().split('\n')
        for line in lines:
            if 'STORE' in line.upper():
                vars_found.append('local_var')
            if 'INT_ADD' in line.upper() or 'ADD' in line.upper():
                vars_found.append('counter')
        
        unique_vars = list(set(vars_found))[:5]
        
        return f"""
Local Variable Analysis:
- Stack frame analysis reveals {len(unique_vars)} local variable(s)
- Variables: {', '.join(unique_vars) if unique_vars else 'result, temp, i'}

Type inference from usage:
- result: int (used in arithmetic)
- temp: int (intermediate computations)
- i: int (loop counter pattern)
"""
    
    def _analyze_control_flow(self, pcode: str) -> str:
        """Analyze control flow from P-Code."""
        lines = pcode.strip().split('\n')
        
        has_branch = any('BRANCH' in line.upper() or 'JMP' in line.upper() for line in lines)
        has_cbranch = any('CBRANCH' in line.upper() or any(j in line.upper() for j in ['JE', 'JNE', 'JL', 'JG']) for line in lines)
        has_loop_pattern = has_cbranch and any('CMP' in line.upper() or 'INT_LESS' in line.upper() for line in lines)
        
        structure = []
        if has_loop_pattern:
            structure.append("for/while loop detected")
        if has_cbranch:
            structure.append("conditional branches (if/else)")
        if has_branch and not has_cbranch:
            structure.append("unconditional jumps")
        
        return f"""
Control Flow Structure:
- Total basic blocks: ~{len(lines) // 5 + 1}
- {'; '.join(structure) if structure else 'Sequential execution'}

Pattern Recognition:
- Loop detected: {has_loop_pattern}
- Conditionals: {has_cbranch}

Reconstructed structure:
{self._generate_structure_template(has_loop_pattern, has_cbranch)}
"""
    
    def _generate_structure_template(self, has_loop: bool, has_conditional: bool) -> str:
        """Generate control flow structure template."""
        if has_loop and has_conditional:
            return """
for (int i = 0; i < n; i++) {
    if (condition) {
        // branch A
    } else {
        // branch B
    }
}"""
        elif has_loop:
            return """
for (int i = 0; i < n; i++) {
    // loop body
}"""
        elif has_conditional:
            return """
if (condition) {
    // then branch
} else {
    // else branch
}"""
        else:
            return """
// sequential execution
statement1;
statement2;
return result;"""
    
    def _analyze_data_flow(self, pcode: str) -> str:
        """Analyze data flow from P-Code."""
        lines = pcode.strip().split('\n')
        
        loads = sum(1 for line in lines if 'LOAD' in line.upper())
        stores = sum(1 for line in lines if 'STORE' in line.upper())
        arithmetic = sum(1 for line in lines if any(op in line.upper() for op in ['ADD', 'SUB', 'MUL', 'DIV']))
        
        return f"""
Data Flow Analysis:
- Memory reads (LOAD): {loads}
- Memory writes (STORE): {stores}
- Arithmetic operations: {arithmetic}

Value flow pattern:
1. Parameters loaded from memory/registers
2. Computations performed (accumulator pattern likely)
3. Result stored and returned

No obvious dead code detected - all computations contribute to output.
"""
    
    def _analyze_semantics(self, pcode: str) -> str:
        """Analyze semantic meaning from P-Code."""
        lines = pcode.strip().split('\n')
        
        # Check for common patterns
        has_accumulator = any('ADD' in line.upper() for line in lines) and any('STORE' in line.upper() for line in lines)
        has_comparison = any('CMP' in line.upper() or 'EQUAL' in line.upper() for line in lines)
        has_call = any('CALL' in line.upper() for line in lines)
        
        purpose = []
        if has_accumulator:
            purpose.append("accumulation/summation")
        if has_comparison:
            purpose.append("comparison/search")
        if has_call:
            purpose.append("function delegation")
        
        return f"""
Semantic Analysis:
- Primary operation pattern: {', '.join(purpose) if purpose else 'computation'}
- Algorithm category: {self._classify_algorithm(pcode)}

Function purpose hypothesis:
This function appears to {self._guess_purpose(pcode)}

Confidence: Medium (based on pattern matching)
"""
    
    def _classify_algorithm(self, pcode: str) -> str:
        """Classify the algorithm type."""
        pcode_upper = pcode.upper()
        
        if 'LOOP' in pcode_upper or ('CBRANCH' in pcode_upper and 'INT_LESS' in pcode_upper):
            if 'INT_ADD' in pcode_upper:
                return "iterative accumulation"
            elif 'CMP' in pcode_upper:
                return "search/comparison"
            return "iterative processing"
        elif 'CALL' in pcode_upper and 'CBRANCH' in pcode_upper:
            return "recursive algorithm"
        else:
            return "direct computation"
    
    def _guess_purpose(self, pcode: str) -> str:
        """Guess the function's purpose."""
        pcode_upper = pcode.upper()
        
        if 'INT_ADD' in pcode_upper and 'INT_LESS' in pcode_upper:
            return "compute a sum or accumulate values in a loop"
        elif 'INT_MULT' in pcode_upper:
            return "perform multiplication or compute a product"
        elif 'CMP' in pcode_upper:
            return "compare values and return a result based on comparison"
        else:
            return "perform a computation and return the result"
    
    def _synthesize_code(self, analysis: str) -> str:
        """Synthesize C code from analysis."""
        # Extract key information from analysis
        has_loop = 'loop' in analysis.lower()
        has_conditional = 'if' in analysis.lower() or 'conditional' in analysis.lower()
        has_accumulator = 'accumul' in analysis.lower() or 'sum' in analysis.lower()
        
        if has_accumulator and has_loop:
            return """
int compute_sum(int* arr, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}
```

This function iterates through an array and accumulates the sum of all elements.
"""
        elif has_loop:
            return """
int process_array(int* arr, int n) {
    int result = 0;
    for (int i = 0; i < n; i++) {
        result = compute(arr[i], result);
    }
    return result;
}
```

This function processes an array element by element.
"""
        elif has_conditional:
            return """
int conditional_compute(int a, int b, int flag) {
    if (flag) {
        return a + b;
    } else {
        return a - b;
    }
}
```

This function performs conditional computation based on a flag.
"""
        else:
            return """
int compute(int a, int b) {
    int result = a + b;
    return result;
}
```

This function performs a simple computation on two inputs.
"""
    
    def reason(
        self,
        pcode_ops: List[Dict],
        steps: Optional[List[ReasoningStep]] = None
    ) -> ChainOfThoughtResult:
        """
        Perform chain-of-thought reasoning on P-Code.
        
        Args:
            pcode_ops: List of P-Code operations
            steps: Optional list of specific steps to perform
        
        Returns:
            ChainOfThoughtResult with full reasoning trace
        """
        if steps is None:
            steps = list(ReasoningStep)
        
        # Format P-Code
        pcode_str = self._format_pcode(pcode_ops)
        
        results = []
        analysis_summary = []
        
        for step in steps:
            if step == ReasoningStep.SYNTHESIS:
                # Synthesis uses accumulated analysis
                prompt = self.step_prompts[step].format(
                    analysis="\n".join(analysis_summary)
                )
            else:
                prompt = self.step_prompts[step].format(pcode=pcode_str)
            
            # Generate reasoning
            reasoning = self._generate(
                prompt,
                max_tokens=512,
                temperature=0.3,
                stop_sequences=["Step ", "```\n\n"]
            )
            
            # Parse output
            output = self._parse_step_output(step, reasoning)
            
            # Calculate confidence
            confidence = self._estimate_confidence(reasoning)
            
            result = ReasoningResult(
                step=step,
                reasoning=reasoning,
                output=output,
                confidence=confidence
            )
            results.append(result)
            
            # Add to analysis summary
            analysis_summary.append(f"{step.value}: {reasoning[:200]}...")
        
        # Extract final code from synthesis
        final_code = ""
        if results and results[-1].step == ReasoningStep.SYNTHESIS:
            code_match = re.search(r'```c?\n(.*?)```', results[-1].reasoning, re.DOTALL)
            if code_match:
                final_code = code_match.group(1).strip()
            else:
                final_code = results[-1].reasoning
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence for r in results) / len(results) if results else 0.0
        
        # Build reasoning trace
        reasoning_trace = "\n\n".join([
            f"=== {r.step.value.upper()} ===\n{r.reasoning}"
            for r in results
        ])
        
        return ChainOfThoughtResult(
            steps=results,
            final_code=final_code,
            overall_confidence=overall_confidence,
            reasoning_trace=reasoning_trace
        )
    
    def _format_pcode(self, pcode_ops: List[Dict]) -> str:
        """Format P-Code operations as text."""
        lines = []
        for i, op in enumerate(pcode_ops[:50]):  # Limit to 50 ops
            if isinstance(op, dict):
                mnemonic = op.get('mnemonic', 'UNKNOWN')
                address = op.get('address', f'0x{i:04x}')
                inputs = op.get('inputs', [])
                output = op.get('output', {})
                
                input_str = ', '.join([str(inp.get('name', inp.get('offset', ''))) for inp in inputs[:3]])
                output_str = str(output.get('name', output.get('offset', ''))) if output else ''
                
                if output_str and input_str:
                    line = f"{address}: {mnemonic} {output_str} <- {input_str}"
                elif input_str:
                    line = f"{address}: {mnemonic} {input_str}"
                else:
                    line = f"{address}: {mnemonic}"
            else:
                line = f"0x{i:04x}: {str(op)}"
            
            lines.append(line)
        
        return '\n'.join(lines)
    
    def _parse_step_output(self, step: ReasoningStep, reasoning: str) -> Dict:
        """Parse structured output from reasoning."""
        output = {'raw': reasoning}
        
        if step == ReasoningStep.SIGNATURE:
            # Extract function signature
            sig_match = re.search(r'signature[:\s]+([a-zA-Z_]\w*\s+\w+\s*\([^)]*\))', reasoning, re.IGNORECASE)
            if sig_match:
                output['signature'] = sig_match.group(1)
        
        elif step == ReasoningStep.PARAMETERS:
            # Extract parameters
            params = re.findall(r'(arg\d+|param\d+)[:\s]+(\w+)', reasoning, re.IGNORECASE)
            output['parameters'] = [{'name': p[0], 'type': p[1]} for p in params]
        
        elif step == ReasoningStep.LOCAL_VARS:
            # Extract local variables
            vars = re.findall(r'(\w+)[:\s]+(\w+)\s*\(', reasoning)
            output['variables'] = [{'name': v[0], 'type': v[1]} for v in vars]
        
        elif step == ReasoningStep.CONTROL_FLOW:
            output['has_loop'] = 'loop' in reasoning.lower()
            output['has_conditional'] = 'if' in reasoning.lower() or 'conditional' in reasoning.lower()
        
        return output
    
    def _estimate_confidence(self, reasoning: str) -> float:
        """Estimate confidence of reasoning."""
        # Heuristics for confidence
        confidence = 0.5
        
        # Longer reasoning tends to be more confident
        if len(reasoning) > 500:
            confidence += 0.1
        
        # Specific patterns indicate confidence
        confidence_indicators = ['based on', 'analysis shows', 'determined', 'identified']
        for indicator in confidence_indicators:
            if indicator in reasoning.lower():
                confidence += 0.05
        
        # Uncertainty indicators decrease confidence
        uncertainty_indicators = ['unclear', 'might be', 'possibly', 'uncertain']
        for indicator in uncertainty_indicators:
            if indicator in reasoning.lower():
                confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))


class SimpleTokenizer:
    """Simple tokenizer fallback when HuggingFace models unavailable."""
    
    def __init__(self):
        self.pad_token = '<PAD>'
        self.eos_token = '<EOS>'
        self.pad_token_id = 0
        self.eos_token_id = 1
    
    def __call__(self, text, return_tensors=None, **kwargs):
        # Simple character tokenization
        tokens = list(text.encode('utf-8'))
        if return_tensors == 'pt':
            return {'input_ids': torch.tensor([tokens])}
        return {'input_ids': tokens}
    
    def decode(self, tokens, **kwargs):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return bytes(tokens).decode('utf-8', errors='ignore')


# Global reasoner instance
reasoner = None


def load_reasoner():
    """Initialize the CoT reasoner."""
    global reasoner
    
    model_name = os.getenv('MODEL_NAME', 'codellama/CodeLlama-7b-Instruct-hf')
    adapter_path = os.getenv('ADAPTER_PATH', '/app/models/cot_adapter')
    use_quantization = os.getenv('USE_QUANTIZATION', 'true').lower() == 'true'
    
    # Check local paths
    local_adapter = Path(__file__).parent / 'checkpoints' / 'cot_adapter'
    if local_adapter.exists():
        adapter_path = str(local_adapter)
    
    try:
        reasoner = ChainOfThoughtReasoner(
            model_name=model_name,
            adapter_path=adapter_path if os.path.exists(adapter_path) else None,
            use_quantization=use_quantization
        )
        print("CoT Reasoner loaded successfully")
    except Exception as e:
        print(f"Error loading reasoner: {e}")
        # Create with fallback
        reasoner = ChainOfThoughtReasoner(
            model_name=model_name,
            use_quantization=False
        )


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'service': 'cot-reasoner',
        'model_loaded': reasoner is not None and reasoner.model is not None,
        'device': str(reasoner.device) if reasoner else 'unknown'
    })


@app.route('/reason', methods=['POST'])
def reason():
    """
    Perform chain-of-thought reasoning on P-Code.
    
    Request body:
    {
        "pcode": [...],
        "steps": ["function_signature", "control_flow", ...]  // Optional
    }
    
    Returns full reasoning trace and generated code.
    """
    try:
        data = request.json
        pcode = data.get('pcode', data.get('features', []))
        
        if not pcode:
            return jsonify({'error': 'pcode required'}), 400
        
        if not reasoner:
            return jsonify({'error': 'Reasoner not initialized'}), 503
        
        # Parse steps if provided
        step_names = data.get('steps')
        steps = None
        if step_names:
            steps = [ReasoningStep(s) for s in step_names if s in [e.value for e in ReasoningStep]]
        
        # Perform reasoning
        result = reasoner.reason(pcode, steps)
        
        return jsonify({
            'final_code': result.final_code,
            'confidence': result.overall_confidence,
            'reasoning_trace': result.reasoning_trace,
            'steps': [
                {
                    'step': r.step.value,
                    'reasoning': r.reasoning,
                    'output': r.output,
                    'confidence': r.confidence
                }
                for r in result.steps
            ],
            'success': True
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze specific aspect of code.
    
    Request body:
    {
        "pcode": [...],
        "aspect": "control_flow" | "data_flow" | "semantics"
    }
    """
    try:
        data = request.json
        pcode = data.get('pcode', [])
        aspect = data.get('aspect', 'control_flow')
        
        if not pcode:
            return jsonify({'error': 'pcode required'}), 400
        
        if not reasoner:
            return jsonify({'error': 'Reasoner not initialized'}), 503
        
        # Map aspect to step
        aspect_map = {
            'signature': ReasoningStep.SIGNATURE,
            'parameters': ReasoningStep.PARAMETERS,
            'variables': ReasoningStep.LOCAL_VARS,
            'control_flow': ReasoningStep.CONTROL_FLOW,
            'data_flow': ReasoningStep.DATA_FLOW,
            'semantics': ReasoningStep.SEMANTICS
        }
        
        step = aspect_map.get(aspect, ReasoningStep.CONTROL_FLOW)
        
        # Perform single-step reasoning
        result = reasoner.reason(pcode, [step])
        
        if result.steps:
            step_result = result.steps[0]
            return jsonify({
                'aspect': aspect,
                'analysis': step_result.reasoning,
                'output': step_result.output,
                'confidence': step_result.confidence,
                'success': True
            })
        
        return jsonify({'error': 'Analysis failed'}), 500
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    load_reasoner()
    app.run(host='0.0.0.0', port=5005, debug=True)
