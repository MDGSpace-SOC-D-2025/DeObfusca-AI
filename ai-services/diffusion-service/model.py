"""
Discrete Denoising Diffusion Model for Code Generation (D3PM)

This module implements a Discrete Diffusion model specifically designed for
generating C source code from decompiled representations.

Architecture: D3PM (Discrete Denoising Diffusion Probabilistic Models)
- Absorbing state diffusion process for discrete tokens
- Transformer-based denoiser with cross-attention to P-Code
- Code-aware noise scheduling

Reference: "Structured Denoising Diffusion Models in Discrete State-Spaces"
           (Austin et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass


class SinusoidalPositionEmbeddings(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    
    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0
    ):
        super().__init__()
        inner_dim = dim_head * heads
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        
        # Reshape for multi-head attention
        q = q.view(batch, seq_len, self.heads, -1).transpose(1, 2)
        k = k.view(batch, -1, self.heads, -1).transpose(1, 2)
        v = v.view(batch, -1, self.heads, -1).transpose(1, 2)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, -1)
        
        return self.to_out(out)


class TransformerBlock(nn.Module):
    
    def __init__(
        self,
        dim: int,
        context_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        
        # Cross-attention
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(
            dim, context_dim, heads, dim_head, dropout
        )
        
        # Feed-forward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = residual + x
        
        # Cross-attention
        residual = x
        x = self.norm2(x)
        x = residual + self.cross_attn(x, context, cross_attn_mask)
        
        # Feed-forward
        residual = x
        x = self.norm3(x)
        x = residual + self.ff(x)
        
        return x


class CodeDenoiser(nn.Module):
    
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 2048,
        dim: int = 512,
        context_dim: int = 256,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.dim = dim
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, dim)
        
        # Position embedding
        self.pos_embed = nn.Embedding(max_seq_len, dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, context_dim, heads, dim_head, dropout=dropout)
            for _ in range(depth)
        ])
        
        # Output projection
        self.norm_out = nn.LayerNorm(dim)
        self.proj_out = nn.Linear(dim, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len] - noisy tokens
        t: torch.Tensor,  # [batch] - timesteps
        context: torch.Tensor,  # [batch, context_len, context_dim] - P-Code embeddings
        context_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len = x.shape
        
        # Embed tokens
        h = self.token_embed(x)
        
        # Add position embeddings
        positions = torch.arange(seq_len, device=x.device)
        h = h + self.pos_embed(positions)
        
        # Add time embeddings (broadcast to all positions)
        time_emb = self.time_embed(t)[:, None, :]
        h = h + time_emb
        
        # Transformer blocks
        for block in self.blocks:
            h = block(h, context, cross_attn_mask=context_mask)
        
        # Output
        h = self.norm_out(h)
        logits = self.proj_out(h)
        
        return logits


class D3PMScheduler:
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        vocab_size: int = 32000,
        mask_token_id: int = 1,  # [MASK] token
        schedule: str = 'cosine'
    ):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        
        # Create noise schedule (probability of masking at each timestep)
        if schedule == 'cosine':
            self.mask_probs = self._cosine_schedule()
        elif schedule == 'linear':
            self.mask_probs = self._linear_schedule()
        elif schedule == 'sqrt':
            self.mask_probs = self._sqrt_schedule()
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def _cosine_schedule(self) -> torch.Tensor:
        steps = torch.linspace(0, self.num_timesteps, self.num_timesteps + 1)
        alpha_bar = torch.cos((steps / self.num_timesteps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        # Convert to masking probability
        mask_probs = 1 - alpha_bar[1:]
        return mask_probs
    
    def _linear_schedule(self) -> torch.Tensor:
        return torch.linspace(0.0, 1.0, self.num_timesteps)
    
    def _sqrt_schedule(self) -> torch.Tensor:
        """Square root schedule (slower start)."""
        steps = torch.linspace(0, 1, self.num_timesteps)
        return torch.sqrt(steps)
    
    def add_noise(
        self,
        x: torch.Tensor,  # [batch, seq_len] - original tokens
        t: torch.Tensor  # [batch] - timesteps
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise by randomly masking tokens.
        
        Returns:
            noisy_x: Tokens with some masked
            mask: Boolean mask of which tokens were corrupted
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Get mask probability for each sample
        mask_prob = self.mask_probs[t].to(device)  # [batch]
        
        # Sample which tokens to mask
        random_vals = torch.rand(batch_size, seq_len, device=device)
        mask = random_vals < mask_prob[:, None]  # [batch, seq_len]
        
        # Apply masking
        noisy_x = x.clone()
        noisy_x[mask] = self.mask_token_id
        
        return noisy_x, mask
    
    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)


class PCodeEncoder(nn.Module):
    """
    Encoder for P-Code operations to create context for diffusion.
    
    Produces fixed-size embeddings from variable-length P-Code sequences.
    """
    
    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Instruction embedding
        self.embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position embedding
        self.pos_embed = nn.Embedding(2048, embed_dim)
        
        # Project to hidden dim
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len] - P-Code instruction indices
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode P-Code to context embeddings.
        
        Returns: [batch, seq_len, hidden_dim]
        """
        positions = torch.arange(x.size(1), device=x.device)



class D3PMCodeGenerator(nn.Module):
    """
    Complete D3PM model for code generation from P-Code.
    
    Architecture:
    1. PCodeEncoder: Encode P-Code to context
    2. CodeDenoiser: Denoise code tokens conditioned on context
    3. D3PMScheduler: Manage noise scheduling
    
    Training: Predict original tokens from partially masked code
    Inference: Iteratively unmask tokens starting from all [MASK]
    """
    
    def __init__(
        self,
        code_vocab_size: int = 32000,
        pcode_vocab_size: int = 256,
        max_code_len: int = 2048,
        max_pcode_len: int = 512,
        hidden_dim: int = 512,
        context_dim: int = 256,
        num_denoiser_layers: int = 12,
        num_encoder_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_timesteps: int = 1000,
        mask_token_id: int = 1
    ):
        super().__init__()
        
        self.code_vocab_size = code_vocab_size
        self.max_code_len = max_code_len
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        
        # P-Code encoder
        self.pcode_encoder = PCodeEncoder(
            vocab_size=pcode_vocab_size,
            embed_dim=context_dim,
            hidden_dim=context_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Code denoiser
        self.denoiser = CodeDenoiser(
            vocab_size=code_vocab_size,
            max_seq_len=max_code_len,
            dim=hidden_dim,
            context_dim=context_dim,
            depth=num_denoiser_layers,
            heads=num_heads,
            dropout=dropout
        )
        
        # Scheduler
        self.scheduler = D3PMScheduler(
            num_timesteps=num_timesteps,
            vocab_size=code_vocab_size,
            mask_token_id=mask_token_id,
            schedule='cosine'
        )
    
    def forward(
        self,
        code_tokens: torch.Tensor,  # [batch, code_len] - target code
        pcode_tokens: torch.Tensor,  # [batch, pcode_len] - input P-Code
        pcode_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.
        
        Returns:
            logits: [batch, code_len, vocab_size] - predicted token logits
            loss: Scalar cross-entropy loss
        """
        batch_size = code_tokens.size(0)
        device = code_tokens.device
        
        # Encode P-Code context
        context = self.pcode_encoder(pcode_tokens, pcode_mask)
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(batch_size, device)
        
        # Add noise to code
        noisy_code, noise_mask = self.scheduler.add_noise(code_tokens, t)
        
        # Predict original tokens
        logits = self.denoiser(noisy_code, t, context)
        
        # Compute loss (only on masked positions)
        loss = F.cross_entropy(
            logits.view(-1, self.code_vocab_size),
            code_tokens.view(-1),
            reduction='none'
        )
        loss = loss.view(batch_size, -1)
        
        # Weight loss by noise mask (focus on corrupted tokens)
        weighted_loss = (loss * noise_mask.float()).sum() / (noise_mask.sum() + 1e-8)
        
        return logits, weighted_loss
    
    @torch.no_grad()
    def generate(
        self,
        pcode_tokens: torch.Tensor,  # [batch, pcode_len]
        max_length: int = 512,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Generate code by iteratively unmasking tokens.
        
        Starts from all [MASK] tokens and progressively reveals.
        """
        batch_size = pcode_tokens.size(0)
        device = pcode_tokens.device
        
        # Encode P-Code context
        context = self.pcode_encoder(pcode_tokens)
        
        # Start with all masked tokens
        x = torch.full((batch_size, max_length), self.mask_token_id, device=device)
        
        # Iteratively unmask
        for step in range(num_steps):
            # Current timestep (from high to low)
            t_val = self.num_timesteps - 1 - (step * self.num_timesteps // num_steps)
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            
            # Predict original tokens
            logits = self.denoiser(x, t, context)
            logits = logits / temperature
            
            # Apply top-k/top-p filtering
            if top_k is not None:
                logits = self._top_k_filtering(logits, top_k)
            if top_p is not None:
                logits = self._top_p_filtering(logits, top_p)
            
            # Sample tokens
            probs = F.softmax(logits, dim=-1)
            new_tokens = torch.multinomial(probs.view(-1, self.code_vocab_size), 1)
            new_tokens = new_tokens.view(batch_size, max_length)
            
            # Determine which tokens to unmask at this step
            # Use confidence-based selection
            confidences = probs.max(dim=-1)[0]  # [batch, seq_len]
            
            # Number of tokens to unmask
            mask_count = (x == self.mask_token_id).sum(dim=1)  # Per sample
            unmask_count = (mask_count.float() * (1 - t_val / self.num_timesteps)).long()
            unmask_count = torch.clamp(unmask_count, min=1)
            
            # Unmask highest confidence masked positions
            for b in range(batch_size):
                masked_positions = (x[b] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                
                # Get confidences for masked positions
                masked_confidences = confidences[b, masked_positions]
                
                # Select top positions to unmask
                num_to_unmask = min(unmask_count[b].item(), len(masked_positions))
                _, top_indices = masked_confidences.topk(num_to_unmask)
                positions_to_unmask = masked_positions[top_indices]
                
                # Unmask
                x[b, positions_to_unmask] = new_tokens[b, positions_to_unmask]
        
        return x
    
    def _top_k_filtering(
        self,
        logits: torch.Tensor,
        top_k: int
    ) -> torch.Tensor:
        """Filter to top-k tokens."""
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    def _top_p_filtering(
        self,
        logits: torch.Tensor,
        top_p: float
    ) -> torch.Tensor:
        """Nucleus sampling - filter to top-p probability mass."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            -1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits


class CodeTokenizer:
    """
    Tokenizer for C source code using BPE-style tokenization.
    
    Handles:
    - C keywords and operators
    - Identifiers and literals
    - Whitespace and indentation
    """
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'
    MASK_TOKEN = '<MASK>'
    BOS_TOKEN = '<BOS>'
    EOS_TOKEN = '<EOS>'
    
    # C keywords
    C_KEYWORDS = [
        'auto', 'break', 'case', 'char', 'const', 'continue', 'default',
        'do', 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto',
        'if', 'inline', 'int', 'long', 'register', 'restrict', 'return',
        'short', 'signed', 'sizeof', 'static', 'struct', 'switch',
        'typedef', 'union', 'unsigned', 'void', 'volatile', 'while',
        '_Bool', '_Complex', '_Imaginary'
    ]
    
    # Common operators
    C_OPERATORS = [
        '+', '-', '*', '/', '%', '++', '--',
        '==', '!=', '<', '>', '<=', '>=',
        '&&', '||', '!',
        '&', '|', '^', '~', '<<', '>>',
        '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
        '->', '.', ',', ';', ':', '?',
        '(', ')', '[', ']', '{', '}',
    ]
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        
        # Build initial vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Add special tokens
        special_tokens = [
            self.PAD_TOKEN, self.MASK_TOKEN, self.UNK_TOKEN,
            self.BOS_TOKEN, self.EOS_TOKEN
        ]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        idx = len(special_tokens)
        
        # Add keywords
        for kw in self.C_KEYWORDS:
            self.token_to_id[kw] = idx
            self.id_to_token[idx] = kw
            idx += 1
        
        # Add operators
        for op in self.C_OPERATORS:
            self.token_to_id[op] = idx
            self.id_to_token[idx] = op
            idx += 1
        
        # Add common identifiers and patterns
        common_patterns = [
            'main', 'printf', 'scanf', 'malloc', 'free', 'sizeof',
            'NULL', 'true', 'false',
            'i', 'j', 'k', 'n', 'm', 'x', 'y', 'z',
            'ptr', 'temp', 'result', 'count', 'size', 'len',
            'arr', 'str', 'buf', 'data', 'node', 'head', 'tail',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '\\n', '\\t', '\\0', ' ', '  ', '    ',
        ]
        
        for pattern in common_patterns:
            if pattern not in self.token_to_id:
                self.token_to_id[pattern] = idx
                self.id_to_token[idx] = pattern
                idx += 1
        
        # Add single characters for character-level fallback
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_':
            if c not in self.token_to_id:
                self.token_to_id[c] = idx
                self.id_to_token[idx] = c
                idx += 1
        
        self.current_vocab_size = idx
        
        # Special token IDs
        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.mask_token_id = self.token_to_id[self.MASK_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]
        self.bos_token_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
    
    def tokenize(self, text: str) -> List[str]:
        tokens = []
        i = 0
        
        while i < len(text):
            # Skip whitespace but preserve newlines for structure
            if text[i] == '\n':
                tokens.append('\\n')
                i += 1
                continue
            elif text[i] in ' \t':
                # Count indentation
                spaces = 0
                while i < len(text) and text[i] in ' \t':
                    spaces += 1 if text[i] == ' ' else 4
                    i += 1
                
                # Add indentation tokens
                while spaces >= 4:
                    tokens.append('    ')
                    spaces -= 4
                while spaces >= 1:
                    tokens.append(' ')
                    spaces -= 1
                continue
            
            # Check for multi-character operators
            found = False
            for op in sorted(self.C_OPERATORS, key=len, reverse=True):
                if text[i:].startswith(op):
                    tokens.append(op)
                    i += len(op)
                    found = True
                    break
            
            if found:
                continue
            
            # Check for keywords and identifiers
            if text[i].isalpha() or text[i] == '_':
                j = i
                while j < len(text) and (text[j].isalnum() or text[j] == '_'):
                    j += 1
                word = text[i:j]
                tokens.append(word)
                i = j
                continue
            
            # Check for numbers
            if text[i].isdigit():
                j = i
                while j < len(text) and (text[j].isdigit() or text[j] in '.xXaAbBcCdDeEfF'):
                    j += 1
                num = text[i:j]
                tokens.append(num)
                i = j
                continue
            
            # Check for string literals
            if text[i] in '"\'':
                quote = text[i]
                j = i + 1
                while j < len(text) and text[j] != quote:
                    if text[j] == '\\':
                        j += 2
                    else:
                        j += 1
                if j < len(text):
                    j += 1
                literal = text[i:j]
                tokens.append(literal)
                i = j
                continue
            
            # Single character fallback
            tokens.append(text[i])
            i += 1
        
        return tokens
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None,
        add_special_tokens: bool = True
    ) -> List[int]:
        # Encode text to token IDs
        tokens = self.tokenize(text)
        
        ids = []
        if add_special_tokens:
            ids.append(self.bos_token_id)
        
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Character-level fallback
                for c in token:
                    ids.append(self.token_to_id.get(c, self.unk_token_id))
        
        if add_special_tokens:
            ids.append(self.eos_token_id)
        
        # Truncate or pad
        if max_length is not None:
            if len(ids) > max_length:
                ids = ids[:max_length]
            else:
                ids = ids + [self.pad_token_id] * (max_length - len(ids))
        
        return ids
    
    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True
    ) -> str:
        # Decode token IDs to text
        tokens = []
        
        special_ids = {
            self.pad_token_id, self.mask_token_id, self.unk_token_id,
            self.bos_token_id, self.eos_token_id
        }
        
        for id in ids:
            if skip_special_tokens and id in special_ids:
                continue
            
            if id in self.id_to_token:
                token = self.id_to_token[id]
                # Handle escape sequences
                if token == '\\n':
                    tokens.append('\n')
                elif token == '\\t':
                    tokens.append('\t')
                else:
                    tokens.append(token)
        
        # Join tokens (add space between word tokens)
        result = []
        for i, token in enumerate(tokens):
            if i > 0 and token.isalnum() and result and result[-1].isalnum():
                result.append(' ')
            result.append(token)
        
        return ''.join(result)
    
    def save(self, path: str):
        # Save tokenizer vocabulary
        import json
        with open(path, 'w') as f:
            json.dump({
                'token_to_id': self.token_to_id,
                'vocab_size': self.vocab_size
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'CodeTokenizer':
        # Load tokenizer from file
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(vocab_size=data['vocab_size'])
        tokenizer.token_to_id = data['token_to_id']
        tokenizer.id_to_token = {int(v): k for k, v in data['token_to_id'].items()}
        return tokenizer


def create_diffusion_model(
    code_vocab_size: int = 32000,
    pcode_vocab_size: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 12,
    pretrained_path: Optional[str] = None
) -> D3PMCodeGenerator:
    # Factory function to create diffusion model
    model = D3PMCodeGenerator(
        code_vocab_size=code_vocab_size,
        pcode_vocab_size=pcode_vocab_size,
        hidden_dim=hidden_dim,
        num_denoiser_layers=num_layers
    )
    
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model
