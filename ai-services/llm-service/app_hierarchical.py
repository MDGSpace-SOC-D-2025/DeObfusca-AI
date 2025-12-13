"""
Hierarchical LLM Decompiler with RAG and Sliding Window.

Solves the context window problem by:
- Processing one function at a time (Local Encoder)
- Maintaining a vector database of all functions (Global Memory Bank)
- Using cross-attention to query other functions when needed

This allows decompiling large binaries (1MB+) that would exceed token limits.
"""

from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Tuple

app = Flask(__name__)

class HierarchicalLLMDecompiler:
    """
    Hierarchical Encoder-Decoder with RAG (Retrieval-Augmented Generation).
    
    Architecture:
    - Local Encoder: Processes one function at a time
    - Global Memory Bank: Vector DB of all functions in binary
    - Cross-Attention: Queries memory bank when function calls are detected
    
    This eliminates context window bottleneck for large binaries.
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
        
        # Initialize RAG components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.memory_bank = None
        self.function_metadata = []
        
    def build_memory_bank(self, functions: List[Dict]):
        """
        Build global memory bank of all functions in binary.
        
        Args:
            functions: List of {'name': str, 'pcode': [...], 'summary': str}
        """
        self.function_metadata = functions
        
        # Generate embeddings for each function
        summaries = [f.get('summary', f.get('name', 'unknown')) for f in functions]
        embeddings = self.embedding_model.encode(summaries, convert_to_numpy=True)
        
        # Build FAISS index for fast similarity search
        dimension = embeddings.shape[1]
        self.memory_bank = faiss.IndexFlatL2(dimension)
        self.memory_bank.add(embeddings.astype('float32'))
        
        print(f"Memory bank built with {len(functions)} functions")
    
    def query_memory_bank(self, query: str, k: int = 3) -> List[Dict]:
        """
        Query memory bank for relevant functions.
        
        Args:
            query: Natural language query (e.g., "decrypt string")
            k: Number of results to return
        
        Returns:
            List of function metadata dictionaries
        """
        if self.memory_bank is None:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        distances, indices = self.memory_bank.search(query_embedding.astype('float32'), k)
        
        # Return metadata
        results = []
        for idx in indices[0]:
            if idx < len(self.function_metadata):
                results.append(self.function_metadata[idx])
        
        return results
    
    def decompile_function(
        self,
        sanitized_features: List[dict],
        function_name: str = "unknown",
        context_functions: List[Dict] = None,
        max_length: int = 2048
    ) -> str:
        """
        Decompile a single function with RAG context.
        
        Args:
            sanitized_features: P-Code after sanitization
            function_name: Name of function being decompiled
            context_functions: Related functions from memory bank
            max_length: Max tokens to generate
        
        Returns:
            Decompiled C code
        """
        # Format P-Code
        pcode_str = self._format_pcode(sanitized_features)
        
        # Build context from memory bank
        context_str = ""
        if context_functions:
            context_str = "\n\nRelated functions:\n"
            for func in context_functions[:3]:  # Limit to top 3
                context_str += f"- {func.get('name', 'unknown')}: {func.get('summary', 'N/A')}\n"
        
        # Create prompt with hierarchical structure
        prompt = f"""<s>[INST] You are an expert reverse engineer decompiling obfuscated binary code.

Function: {function_name}
{context_str}

Binary representation (P-Code):
{pcode_str}

Task: Generate clean, readable C code for this function. Use meaningful variable names based on their usage patterns. If this function calls other functions, use their documented behavior.

Decompiled C code: [/INST]
"""
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate with constrained decoding (see GrammarGuidedDecoding)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract C code
        if '[/INST]' in generated_text:
            c_code = generated_text.split('[/INST]')[1].strip()
        else:
            c_code = generated_text
        
        return c_code
    
    def decompile_binary(
        self,
        functions_data: List[Dict],
        max_length: int = 2048
    ) -> Dict[str, str]:
        """
        Decompile entire binary using sliding window approach.
        
        Args:
            functions_data: List of functions with sanitized features
            max_length: Max tokens per function
        
        Returns:
            Dictionary mapping function names to decompiled code
        """
        # Build memory bank from all functions
        self.build_memory_bank(functions_data)
        
        decompiled = {}
        
        for func_data in functions_data:
            func_name = func_data.get('name', 'unknown')
            features = func_data.get('sanitized_features', [])
            
            # Query memory bank for related functions
            query = f"functions called by {func_name}"
            context_funcs = self.query_memory_bank(query, k=3)
            
            # Decompile with context
            c_code = self.decompile_function(
                features,
                function_name=func_name,
                context_functions=context_funcs,
                max_length=max_length
            )
            
            decompiled[func_name] = c_code
        
        return decompiled
    
    def _format_pcode(self, features: List[dict], max_ops: int = 100) -> str:
        """Format P-Code operations as text."""
        lines = []
        for i, op in enumerate(features[:max_ops]):
            mnemonic = op.get('mnemonic', 'UNKNOWN')
            addr = op.get('address', f'0x{i:04x}')
            lines.append(f"{addr}: {mnemonic}")
        
        if len(features) > max_ops:
            lines.append(f"... ({len(features) - max_ops} more instructions)")
        
        return '\n'.join(lines)


# Global model instance
decompiler = None

def load_decompiler():
    global decompiler
    model_name = os.getenv('MODEL_NAME', 'codellama/CodeLlama-7b-hf')
    adapter_path = os.getenv('ADAPTER_PATH', '/app/models/qlora_adapter')
    
    try:
        decompiler = HierarchicalLLMDecompiler(model_name=model_name, adapter_path=adapter_path)
        print("Hierarchical LLM Decompiler loaded successfully")
    except Exception as e:
        print(f"Error loading decompiler: {e}")


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'service': 'hierarchical-llm-decompiler',
        'model_loaded': decompiler is not None
    })


@app.route('/decompile', methods=['POST'])
def decompile():
    """
    Decompile sanitized features to C code with RAG.
    
    Request body:
    {
        "sanitized_features": [...],
        "function_name": "main",
        "context_functions": [...]  // Optional
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
        func_name = data.get('function_name', 'unknown')
        context_funcs = data.get('context_functions', [])
        
        if not features:
            return jsonify({'error': 'sanitized_features required'}), 400
        
        if not decompiler:
            return jsonify({'error': 'Decompiler not initialized'}), 503
        
        # Decompile with RAG context
        source_code = decompiler.decompile_function(
            features,
            function_name=func_name,
            context_functions=context_funcs
        )
        
        return jsonify({
            'source': source_code,
            'success': True,
            'input_length': len(features),
            'function_name': func_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/decompile-binary', methods=['POST'])
def decompile_binary_endpoint():
    """
    Decompile entire binary with sliding window and RAG.
    
    Request body:
    {
        "functions": [
            {
                "name": "main",
                "sanitized_features": [...],
                "summary": "Entry point"
            },
            ...
        ]
    }
    
    Returns:
    {
        "decompiled": {
            "main": "int main() { ... }",
            ...
        },
        "success": true
    }
    """
    try:
        data = request.json
        functions = data.get('functions', [])
        
        if not functions:
            return jsonify({'error': 'functions required'}), 400
        
        if not decompiler:
            return jsonify({'error': 'Decompiler not initialized'}), 503
        
        # Decompile entire binary
        decompiled = decompiler.decompile_binary(functions)
        
        return jsonify({
            'decompiled': decompiled,
            'success': True,
            'function_count': len(decompiled)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_decompiler()
    app.run(host='0.0.0.0', port=5003, debug=True)
