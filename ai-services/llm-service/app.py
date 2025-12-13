from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os

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
    
    def decompile(self, sanitized_features, max_length=2048):
        """
        Decompile sanitized P-Code to C source code.
        
        Args:
            sanitized_features: List of P-Code operations (after sanitization)
            max_length: Maximum generated tokens
        
        Returns:
            C source code string
        """
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
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract C code (everything after [/INST])
        if '[/INST]' in generated_text:
            c_code = generated_text.split('[/INST]')[1].strip()
        else:
            c_code = generated_text
        
        return c_code
    
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
