

from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from model import D3PMCodeGenerator, CodeTokenizer, create_diffusion_model

app = Flask(__name__)


@dataclass
class GenerationResult:
    code: str
    confidence: float
    num_steps: int
    tokens_generated: int


class DiffusionCodeGenerator:

    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        pcode_vocab_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default config
        self.config = config or {
            'hidden_dim': 512,
            'num_layers': 12,
            'max_code_len': 512,
            'max_pcode_len': 256,
            'num_timesteps': 1000,
            'num_inference_steps': 50,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.95
        }
        
        # Load tokenizers
        self.code_tokenizer = self._load_code_tokenizer(tokenizer_path)
        self.pcode_vocab = self._load_pcode_vocab(pcode_vocab_path)
        
        # Load model
        self.model = self._load_model(model_path)
        
        print(f"Diffusion Code Generator initialized on {self.device}")
    
    def _load_code_tokenizer(self, path: Optional[str]) -> CodeTokenizer:

        if path and os.path.exists(path):
            return CodeTokenizer.load(path)
        return CodeTokenizer()
    
    def _load_pcode_vocab(self, path: Optional[str]) -> Dict[str, int]:

        if path and os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        
        # Default vocabulary
        mnemonics = [
            '<PAD>', '<UNK>', '<START>', '<END>',
            'COPY', 'LOAD', 'STORE', 'PIECE', 'SUBPIECE',
            'INT_ADD', 'INT_SUB', 'INT_MULT', 'INT_DIV', 'INT_SDIV',
            'INT_REM', 'INT_SREM', 'INT_NEGATE',
            'INT_AND', 'INT_OR', 'INT_XOR', 'INT_NOT',
            'INT_LEFT', 'INT_RIGHT', 'INT_SRIGHT',
            'INT_EQUAL', 'INT_NOTEQUAL', 'INT_LESS', 'INT_SLESS',
            'INT_LESSEQUAL', 'INT_SLESSEQUAL',
            'BOOL_AND', 'BOOL_OR', 'BOOL_XOR', 'BOOL_NEGATE',
            'FLOAT_ADD', 'FLOAT_SUB', 'FLOAT_MULT', 'FLOAT_DIV',
            'BRANCH', 'CBRANCH', 'BRANCHIND', 'CALL', 'CALLIND', 'RETURN',
            'PUSH', 'POP', 'MOV', 'LEA', 'NOP', 'JMP',
            'CMP', 'TEST', 'XOR', 'AND', 'OR', 'ADD', 'SUB', 'MUL', 'DIV',
        ]
        return {m: i for i, m in enumerate(mnemonics)}
    
    def _load_model(self, path: Optional[str]) -> D3PMCodeGenerator:

        model = create_diffusion_model(
            code_vocab_size=self.code_tokenizer.current_vocab_size,
            pcode_vocab_size=len(self.pcode_vocab),
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers']
        )
        
        if path and os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded model from {path}")
        else:
            print("Warning: No pretrained model found. Using random initialization.")
            print("Run train_diffusion.py to train the model first.")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def encode_pcode(self, pcode_ops: List[Dict]) -> torch.Tensor:

        max_len = self.config['max_pcode_len']
        
        tokens = []
        for op in pcode_ops[:max_len]:
            if isinstance(op, dict):
                mnemonic = op.get('mnemonic', 'UNKNOWN').upper()
            else:
                mnemonic = str(op).upper()
            
            token_id = self.pcode_vocab.get(mnemonic, self.pcode_vocab.get('<UNK>', 1))
            tokens.append(token_id)
        
        # Pad
        while len(tokens) < max_len:
            tokens.append(self.pcode_vocab.get('<PAD>', 0))
        
        return torch.tensor([tokens[:max_len]], dtype=torch.long, device=self.device)
    
    @torch.no_grad()
    def generate(
        self,
        pcode_ops: List[Dict],
        max_length: Optional[int] = None,
        num_steps: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> GenerationResult:

        max_length = max_length or self.config['max_code_len']
        num_steps = num_steps or self.config['num_inference_steps']
        temperature = temperature or self.config['temperature']
        top_k = top_k or self.config.get('top_k')
        top_p = top_p or self.config.get('top_p')
        
        # Encode P-Code
        pcode_tokens = self.encode_pcode(pcode_ops)
        
        # Generate using diffusion
        generated_tokens = self.model.generate(
            pcode_tokens=pcode_tokens,
            max_length=max_length,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        # Decode to text
        token_ids = generated_tokens[0].cpu().tolist()
        code = self.code_tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Calculate confidence (based on how many tokens were generated vs masked)
        mask_id = self.code_tokenizer.mask_token_id
        non_mask_count = sum(1 for t in token_ids if t != mask_id)
        confidence = non_mask_count / max_length
        
        return GenerationResult(
            code=code,
            confidence=confidence,
            num_steps=num_steps,
            tokens_generated=non_mask_count
        )
    
    @torch.no_grad()
    def refine(
        self,
        code: str,
        pcode_ops: List[Dict],
        num_steps: int = 10,
        mask_ratio: float = 0.15
    ) -> GenerationResult:

        max_length = self.config['max_code_len']
        
        # Encode existing code
        code_ids = self.code_tokenizer.encode(
            code,
            max_length=max_length,
            add_special_tokens=True
        )
        code_tokens = torch.tensor([code_ids], dtype=torch.long, device=self.device)
        
        # Randomly mask some tokens
        mask = torch.rand(1, max_length, device=self.device) < mask_ratio
        mask_token_id = self.code_tokenizer.mask_token_id
        code_tokens[mask] = mask_token_id
        
        # Encode P-Code
        pcode_tokens = self.encode_pcode(pcode_ops)
        
        # Encode context
        context = self.model.pcode_encoder(pcode_tokens)
        
        # Iteratively refine
        for step in range(num_steps):
            t_val = self.model.num_timesteps // (step + 1)
            t = torch.full((1,), t_val, device=self.device, dtype=torch.long)
            
            # Predict original tokens
            logits = self.model.denoiser(code_tokens, t, context)
            logits = logits / self.config['temperature']
            
            # Sample for masked positions only
            probs = F.softmax(logits, dim=-1)
            new_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1)
            new_tokens = new_tokens.view(1, max_length)
            
            # Replace only masked tokens
            code_tokens = torch.where(
                code_tokens == mask_token_id,
                new_tokens,
                code_tokens
            )
        
        # Decode
        token_ids = code_tokens[0].cpu().tolist()
        refined_code = self.code_tokenizer.decode(token_ids, skip_special_tokens=True)
        
        non_mask_count = sum(1 for t in token_ids if t != mask_token_id)
        confidence = non_mask_count / max_length
        
        return GenerationResult(
            code=refined_code,
            confidence=confidence,
            num_steps=num_steps,
            tokens_generated=non_mask_count
        )


# Global generator instance
generator = None


def load_generator():

    global generator
    
    model_path = os.getenv('MODEL_PATH', '/app/models/diffusion_model.pth')
    tokenizer_path = os.getenv('TOKENIZER_PATH', '/app/models/code_tokenizer.json')
    vocab_path = os.getenv('VOCAB_PATH', '/app/models/pcode_vocab.json')
    
    # Check local paths
    local_model = Path(__file__).parent / 'checkpoints' / 'best_model.pth'
    local_tokenizer = Path(__file__).parent / 'checkpoints' / 'code_tokenizer.json'
    local_vocab = Path(__file__).parent / 'checkpoints' / 'pcode_vocab.json'
    
    if local_model.exists():
        model_path = str(local_model)
    if local_tokenizer.exists():
        tokenizer_path = str(local_tokenizer)
    if local_vocab.exists():
        vocab_path = str(local_vocab)
    
    config = {
        'hidden_dim': int(os.getenv('HIDDEN_DIM', '512')),
        'num_layers': int(os.getenv('NUM_LAYERS', '12')),
        'max_code_len': int(os.getenv('MAX_CODE_LEN', '512')),
        'max_pcode_len': int(os.getenv('MAX_PCODE_LEN', '256')),
        'num_inference_steps': int(os.getenv('NUM_STEPS', '50')),
        'temperature': float(os.getenv('TEMPERATURE', '0.8')),
        'top_k': int(os.getenv('TOP_K', '50')) if os.getenv('TOP_K') else None,
        'top_p': float(os.getenv('TOP_P', '0.95')) if os.getenv('TOP_P') else None,
    }
    
    try:
        generator = DiffusionCodeGenerator(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            pcode_vocab_path=vocab_path,
            config=config
        )
        print("Diffusion Code Generator loaded successfully")
    except Exception as e:
        print(f"Error loading generator: {e}")
        generator = DiffusionCodeGenerator(config=config)


@app.route('/health', methods=['GET'])
def health():

    return jsonify({
        'status': 'ok',
        'service': 'diffusion-code-generator',
        'model_loaded': generator is not None,
        'device': str(generator.device) if generator else 'unknown'
    })


@app.route('/generate', methods=['POST'])
def generate():

    try:
        data = request.json
        pcode = data.get('pcode', data.get('features', []))
        
        if not pcode:
            return jsonify({'error': 'pcode required'}), 400
        
        if not generator:
            return jsonify({'error': 'Generator not initialized'}), 503
        
        result = generator.generate(
            pcode_ops=pcode,
            max_length=data.get('max_length'),
            num_steps=data.get('num_steps'),
            temperature=data.get('temperature'),
            top_k=data.get('top_k'),
            top_p=data.get('top_p')
        )
        
        return jsonify({
            'code': result.code,
            'confidence': result.confidence,
            'num_steps': result.num_steps,
            'tokens_generated': result.tokens_generated,
            'success': True
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/refine', methods=['POST'])
def refine():

    try:
        data = request.json
        code = data.get('code', '')
        pcode = data.get('pcode', [])
        
        if not code:
            return jsonify({'error': 'code required'}), 400
        
        if not generator:
            return jsonify({'error': 'Generator not initialized'}), 503
        
        result = generator.refine(
            code=code,
            pcode_ops=pcode,
            num_steps=data.get('num_steps', 10),
            mask_ratio=data.get('mask_ratio', 0.15)
        )
        
        return jsonify({
            'refined_code': result.code,
            'confidence': result.confidence,
            'success': True
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/batch-generate', methods=['POST'])
def batch_generate():

    try:
        data = request.json
        functions = data.get('functions', [])
        
        if not functions:
            return jsonify({'error': 'functions required'}), 400
        
        if not generator:
            return jsonify({'error': 'Generator not initialized'}), 503
        
        results = {}
        for func in functions:
            name = func.get('name', 'unnamed')
            pcode = func.get('pcode', [])
            
            try:
                result = generator.generate(pcode_ops=pcode)
                results[name] = {
                    'code': result.code,
                    'confidence': result.confidence,
                    'success': True
                }
            except Exception as e:
                results[name] = {
                    'code': f'// Error: {str(e)}',
                    'confidence': 0.0,
                    'success': False
                }
        
        return jsonify({
            'results': results,
            'total': len(functions)
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


if __name__ == '__main__':
    load_generator()
    app.run(host='0.0.0.0', port=5004, debug=True)
