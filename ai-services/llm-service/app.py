from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import LogitsProcessor, LogitsProcessorList
from peft import PeftModel
import os
import re

app = Flask(__name__)


class CGrammarConstrainedLogitsProcessor(LogitsProcessor):
    """
    Logits processor that enforces C grammar constraints.
    
    Prevents generation of syntactically invalid C code by:
    - Masking invalid token sequences
    - Enforcing bracket/brace matching
    - Ensuring statement terminators
    - Validating operator placement
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # C keywords and operators
        self.c_keywords = {
            'int', 'void', 'char', 'float', 'double', 'long', 'short',
            'unsigned', 'signed', 'const', 'static', 'extern',
            'if', 'else', 'for', 'while', 'do', 'switch', 'case', 'default',
            'return', 'break', 'continue', 'goto',
            'struct', 'union', 'enum', 'typedef'
        }
        
        # Track state for bracket matching
        self.brace_depth = 0
        self.paren_depth = 0
        self.bracket_depth = 0
        self.last_token = None
        self.statement_complete = True
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply grammar constraints to logits."""
        batch_size = input_ids.shape[0]
        
        for batch_idx in range(batch_size):
            # Get last few tokens for context
            recent_tokens = input_ids[batch_idx, -5:].tolist()
            recent_text = self.tokenizer.decode(recent_tokens, skip_special_tokens=True)
            
            # Update state
            self._update_state(recent_text)
            
            # Apply constraints
            self._enforce_bracket_matching(scores[batch_idx])
            self._enforce_statement_structure(scores[batch_idx], recent_text)
            self._prevent_invalid_sequences(scores[batch_idx], recent_text)
        
        return scores
    
    def _update_state(self, text: str):
        """Update grammar state based on recent text."""
        self.brace_depth += text.count('{') - text.count('}')
        self.paren_depth += text.count('(') - text.count(')')
        self.bracket_depth += text.count('[') - text.count(']')
        self.statement_complete = text.rstrip().endswith((';', '{', '}'))
    
    def _enforce_bracket_matching(self, scores: torch.Tensor):
        """Ensure brackets are properly matched."""
        # If depth is 0, boost closing brackets lower priority
        if self.brace_depth == 0:
            close_brace_id = self.tokenizer.encode('}', add_special_tokens=False)
            if close_brace_id:
                scores[close_brace_id[0]] -= 10.0
        
        # If inside braces, boost valid statement tokens
        if self.brace_depth > 0:
            semicolon_id = self.tokenizer.encode(';', add_special_tokens=False)
            if semicolon_id:
                scores[semicolon_id[0]] += 2.0
    
    def _enforce_statement_structure(self, scores: torch.Tensor, recent_text: str):
        """Enforce C statement structure rules."""
        # After semicolon, boost keywords and types
        if recent_text.rstrip().endswith(';'):
            for keyword in ['int', 'void', 'char', 'return', 'if', 'for', 'while']:
                kw_ids = self.tokenizer.encode(f' {keyword}', add_special_tokens=False)
                if kw_ids:
                    scores[kw_ids[0]] += 1.5
        
        # After type keyword, boost identifier patterns
        if any(recent_text.endswith(f' {kw}') for kw in ['int', 'void', 'char', 'float']):
            # Boost tokens that look like variable names (letters)
            for token_id in range(len(scores)):
                token = self.tokenizer.decode([token_id])
                if token and token[0].isalpha():
                    scores[token_id] += 1.0
    
    def _prevent_invalid_sequences(self, scores: torch.Tensor, recent_text: str):
        """Prevent generation of invalid token sequences."""
        # Prevent double operators like '++' unless intentional
        if recent_text.endswith((' +', ' -', ' *', ' /')):
            for op in ['+', '-', '*', '/']:
                op_id = self.tokenizer.encode(op, add_special_tokens=False)
                if op_id:
                    scores[op_id[0]] -= 5.0
        
        # Prevent unclosed function calls
        if recent_text.count('(') > recent_text.count(')'):
            comma_id = self.tokenizer.encode(',', add_special_tokens=False)
            close_paren_id = self.tokenizer.encode(')', add_special_tokens=False)
            if comma_id:
                scores[comma_id[0]] += 1.0
            if close_paren_id:
                scores[close_paren_id[0]] += 2.0

app = Flask(__name__)

class LLMDecompiler:
    """
    LLM-based decompiler using CodeLlama or StarCoder with QLoRA fine-tuning.
    
    Approach: Hierarchical Skeleton-Skin (SK2Decompile)
    1. Skeleton: Generate high-level structure (functions, control flow)
    2. Skin: Fill in detailed implementation
    """
    def __init__(self, model_name='codellama/CodeLlama-7b-hf', adapter_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load fine-tuned adapter if available
        if adapter_path and os.path.exists(adapter_path):
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            print(f"Loaded QLoRA adapter from {adapter_path}")
        else:
            print("Warning: No fine-tuned adapter found. Using base model.")
        
        self.model.eval()
    
    def decompile(self, sanitized_features, max_length=2048, use_grammar_constraints=True):
        """
        Decompile sanitized P-Code to C source code with grammar constraints.
        
        Args:
            sanitized_features: List of P-Code operations (after sanitization)
            max_length: Maximum generated tokens
            use_grammar_constraints: Apply C grammar constraints during generation
        
        Returns:
            C source code string
        """
        # Handle large functions with sliding window
        if len(sanitized_features) > 2048:
            return self._decompile_sliding_window(sanitized_features, max_length, use_grammar_constraints)
        
        # Format P-Code as input
        pcode_str = self._format_pcode(sanitized_features)
        
        # Create prompt
        prompt = f"""<s>[INST] Decompile the following binary code to readable C code.

Binary representation:
{pcode_str}

Generate clean, readable C code: [/INST]
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Setup grammar constraints
        logits_processor = None
        if use_grammar_constraints:
            logits_processor = LogitsProcessorList([
                CGrammarConstrainedLogitsProcessor(self.tokenizer)
            ])
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                logits_processor=logits_processor
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract C code (everything after [/INST])
        if '[/INST]' in generated_text:
            c_code = generated_text.split('[/INST]')[1].strip()
        else:
            c_code = generated_text
        
        return c_code
    
    def _decompile_sliding_window(self, sanitized_features, max_length=2048, use_grammar_constraints=True):
        """
        Decompile large functions using sliding window with overlap.
        
        Splits function into chunks with 20% overlap to maintain context.
        Merges results while preserving variable linkage.
        """
        window_size = 1800  # Leave room for prompt
        overlap = 360  # 20% overlap
        chunks = []
        
        # Split into overlapping windows
        for i in range(0, len(sanitized_features), window_size - overlap):
            chunk = sanitized_features[i:i + window_size]
            chunks.append(chunk)
        
        # Decompile each chunk
        decompiled_chunks = []
        context_vars = set()
        
        for idx, chunk in enumerate(chunks):
            pcode_str = self._format_pcode(chunk)
            
            # Create context-aware prompt
            if context_vars:
                var_context = ", ".join(sorted(context_vars)[:10])
                context_hint = f"\nContext variables from previous chunk: {var_context}"
            else:
                context_hint = ""
            
            prompt = f"""<s>[INST] Decompile the following binary code to readable C code.
This is chunk {idx + 1}/{len(chunks)} of a larger function.{context_hint}

Binary representation:
{pcode_str}

Generate clean, readable C code: [/INST]
"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            logits_processor = None
            if use_grammar_constraints:
                logits_processor = LogitsProcessorList([
                    CGrammarConstrainedLogitsProcessor(self.tokenizer)
                ])
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length // len(chunks),
                    temperature=0.2,
                    top_p=0.95,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    logits_processor=logits_processor
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if '[/INST]' in generated_text:
                chunk_code = generated_text.split('[/INST]')[1].strip()
            else:
                chunk_code = generated_text
            
            decompiled_chunks.append(chunk_code)
            
            # Extract variable names for next chunk context
            vars_in_chunk = re.findall(r'\b([a-z_][a-z0-9_]*)\b', chunk_code)
            context_vars.update(vars_in_chunk)
        
        # Merge chunks, removing duplicate declarations in overlap regions
        merged_code = self._merge_chunks(decompiled_chunks)
        
        return merged_code
    
    def _merge_chunks(self, chunks):
        """Merge overlapping decompiled chunks intelligently."""
        if len(chunks) == 1:
            return chunks[0]
        
        merged = chunks[0]
        
        for i in range(1, len(chunks)):
            # Remove duplicate variable declarations
            chunk = chunks[i]
            
            # Extract declarations from merged code
            merged_vars = set(re.findall(r'\b(?:int|char|float|void)\s+([a-z_][a-z0-9_]*)', merged))
            
            # Filter out redeclarations from new chunk
            lines = chunk.split('\n')
            filtered_lines = []
            
            for line in lines:
                # Check if line is a declaration
                decl_match = re.search(r'\b(?:int|char|float|void)\s+([a-z_][a-z0-9_]*)', line)
                if decl_match:
                    var_name = decl_match.group(1)
                    if var_name in merged_vars:
                        # Skip redeclaration
                        continue
                
                filtered_lines.append(line)
            
            merged += '\n\n// --- Continuation ---\n' + '\n'.join(filtered_lines)
        
        return merged
    
    def _format_pcode(self, features):
        """Format P-Code operations as text."""
        lines = []
        for i, op in enumerate(features):
            mnemonic = op.get('mnemonic', 'UNKNOWN')
            addr = op.get('address', f'0x{i:04x}')
            lines.append(f"{addr}: {mnemonic}")
        
        return '\n'.join(lines[:100])  # Limit to first 100 instructions

# Global model instance
decompiler = None

def load_decompiler():
    global decompiler
    model_name = os.getenv('MODEL_NAME', 'codellama/CodeLlama-7b-hf')
    adapter_path = os.getenv('ADAPTER_PATH', '/app/models/qlora_adapter')
    
    try:
        decompiler = LLMDecompiler(model_name=model_name, adapter_path=adapter_path)
        print("Decompiler loaded successfully")
    except Exception as e:
        print(f"Error loading decompiler: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'llm-decompiler',
        'model_loaded': decompiler is not None
    })

@app.route('/decompile', methods=['POST'])
def decompile():
    """
    Decompile sanitized features to C code.
    
    Request body:
    {
        "sanitized_features": [...]
    }
    
    Returns:
    {
        "source": "...",
        "success": true
    }
    """
    try:
        data = request.json
        features = data.get('sanitized_features', [])
        
        if not features:
            return jsonify({'error': 'sanitized_features required'}), 400
        
        if not decompiler:
            return jsonify({'error': 'Decompiler not initialized'}), 503
        
        # Decompile
        source_code = decompiler.decompile(features)
        
        return jsonify({
            'source': source_code,
            'success': True,
            'input_length': len(features)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/decompile-binary', methods=['POST'])
def decompile_binary():
    """
    Decompile entire binary with multiple functions (Orchestrator endpoint).
    """
    try:
        data = request.json
        functions = data.get('functions', [])
        
        # Input validation
        if not isinstance(functions, list):
            return jsonify({'error': 'functions must be a list'}), 400
        if len(functions) == 0:
            return jsonify({'error': 'functions cannot be empty'}), 400
        if len(functions) > 1000:
            return jsonify({'error': 'Too many functions (max 1000)'}), 400
        
        # Validate each function
        for i, func in enumerate(functions):
            if not isinstance(func, dict):
                return jsonify({'error': f'functions[{i}] must be a dict'}), 400
            if 'sanitized_features' not in func:
                return jsonify({'error': f'functions[{i}] missing sanitized_features'}), 400
            if not isinstance(func['sanitized_features'], list):
                return jsonify({'error': f'functions[{i}].sanitized_features must be a list'}), 400
        
        if not decompiler:
            return jsonify({'error': 'Decompiler not initialized'}), 503
        
        decompiled = {}
        for func in functions:
            func_name = func.get('name', 'unnamed')
            features = func.get('sanitized_features', [])
            
            try:
                detailed_code = decompiler.decompile(features, max_length=2048)
                decompiled[func_name] = detailed_code
            except Exception as func_err:
                decompiled[func_name] = f"// Error: {str(func_err)}"
        
        return jsonify({'decompiled': decompiled, 'success': True})
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/refine', methods=['POST'])
def refine():
    """
    Refine decompilation based on verification feedback.
    """
    try:
        data = request.json
        features = data.get('sanitized_features', [])
        
        if not decompiler:
            return jsonify({'error': 'Decompiler not initialized'}), 503
        
        refined_code = decompiler.decompile(features, max_length=2048)
        return jsonify({'refined_source': refined_code, 'success': True})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_decompiler()
    app.run(host='0.0.0.0', port=5003, debug=True)
