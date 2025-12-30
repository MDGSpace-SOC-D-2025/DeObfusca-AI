# DeObfusca-AI: AI Services Technical Documentation

## Complete Workflow, Training Data, and Implementation Guide

**Repository:** https://github.com/chayan-bit/DeObfusca-AI  
**Last Updated:** 2025

---

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Service Dependencies](#service-dependencies)
4. [GNN Service](#gnn-service)
5. [Diffusion Service](#diffusion-service)
6. [Chain-of-Thought (CoT) Service](#chain-of-thought-cot-service)
7. [LLM Service](#llm-service)
8. [Multi-Agent Service](#multi-agent-service)
9. [Reinforcement Learning (RL) Service](#reinforcement-learning-rl-service)
10. [CPG Service](#cpg-service)
11. [Ghidra Service](#ghidra-service)
12. [Orchestrator Service](#orchestrator-service)
13. [Complete Pipeline Workflow](#complete-pipeline-workflow)
14. [Training Requirements](#training-requirements)

---

## Overview

DeObfusca-AI is a neural network-based binary deobfuscation system that transforms obfuscated binary code into readable C source code. The system employs a multi-stage pipeline using advanced ML techniques:

- **Edge-Augmented Graph Transformer** for junk instruction detection
- **Discrete Diffusion (D3PM)** for code generation
- **Chain-of-Thought Reasoning** for step-by-step decompilation
- **QLoRA Fine-tuned LLMs** for code generation
- **Multi-Agent Debate System** for consensus-based analysis
- **PPO-based Reinforcement Learning** with symbolic verification
- **Code Property Graphs** for semantic analysis

---

## System Architecture

### Parallel Candidate + Collaborative Refinement Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ORCHESTRATOR                                    │
│            (Parallel Candidate Generation + Collaborative Refinement)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│    GHIDRA     │          │      CPG      │          │      GNN      │
│   (P-Code)    │───────►  │  (Hypergraph) │───────►  │ (Sanitizer)   │
└───────────────┘          └───────────────┘          └───────────────┘
                                                              │
                    ┌─────────────────────────────────────────┘
                    │
                    ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║              PHASE A: PARALLEL CANDIDATE GENERATION               ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       ║
    ║  │      LLM      │   │   DIFFUSION   │   │  MULTI-AGENT  │       ║
    ║  │ (Candidate 1) │   │ (Candidate 2) │   │ (Candidate 3) │       ║
    ║  └───────┬───────┘   └───────┬───────┘   └───────┬───────┘       ║
    ╚══════════│═══════════════════│═══════════════════│═══════════════╝
               │                   │                   │
               └───────────────────┼───────────────────┘
                                   ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║              PHASE B: RL VERIFIER SCORES ALL CANDIDATES           ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║                      ┌───────────────┐                            ║
    ║                      │      RL       │                            ║
    ║                      │  (Z3 Verify)  │                            ║
    ║                      └───────┬───────┘                            ║
    ║                              │                                    ║
    ║            ┌─────────────────┼─────────────────┐                  ║
    ║            ▼                 ▼                 ▼                  ║
    ║     reward=8.2         reward=7.5        reward=9.1              ║
    ║     (LLM)              (Diffusion)       (Multi-Agent) ★BEST     ║
    ╚══════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
    ╔═══════════════════════════════════════════════════════════════════╗
    ║              PHASE C: COLLABORATIVE REFINEMENT                    ║
    ╠═══════════════════════════════════════════════════════════════════╣
    ║      All 3 generators refine the BEST candidate together         ║
    ║  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐       ║
    ║  │      LLM      │   │   DIFFUSION   │   │  MULTI-AGENT  │       ║
    ║  │ refine(best)  │   │ refine(best)  │   │ refine(best)  │       ║
    ║  └───────────────┘   └───────────────┘   └───────────────┘       ║
    ║                         ↓ repeat                                  ║
    ╚═══════════════════════════════════════════════════════════════════╝
                                   │
                          (until reward >= threshold)
                                   │
                                   ▼
                          ┌───────────────┐
                          │     CoT       │
                          │  (Reasoning)  │  (Optional post-processing)
                          └───────────────┘
```

### Key Improvement Over Previous Architecture

**Old approach**: Single generator per iteration, cycling through LLM → Diffusion → Multi-Agent sequentially.

**New approach**:
1. **Parallel Generation**: All 3 generators produce candidates simultaneously each iteration
2. **Competitive Scoring**: RL verifier scores ALL candidates with Z3 symbolic verification
3. **Best-of-N Selection**: Highest-scoring candidate becomes the base for next iteration
4. **Collaborative Refinement**: Instead of starting fresh, ALL generators refine the BEST code together
5. **Faster Convergence**: Leverages diversity of approaches while building on proven best results

---

## Service Dependencies

| Service | Port | Key Dependencies |
|---------|------|------------------|
| Ghidra | 5001 | Ghidra, subprocess |
| GNN | 5002 | PyTorch, torch_geometric |
| LLM | 5003 | transformers, peft, bitsandbytes |
| RL | 5004 | z3-solver, pycparser |
| CPG | 5005 | networkx, PyTorch |
| Diffusion | 5006 | PyTorch, transformers |
| Multi-Agent | 5007 | networkx, PyTorch |
| CoT | 5008 | transformers, peft |
| Orchestrator | 5000 | requests |

---

## GNN Service

### Location: `ai-services/gnn-service/`

### Purpose
Detects and removes junk/obfuscated instructions using Edge-Augmented Graph Transformer.

### Architecture

```python
class EdgeAugmentedAttention(nn.Module):
    """
    Multi-head attention with edge feature augmentation.
    
    Architecture:
    - Node embeddings: [batch, num_nodes, hidden_dim]
    - Edge features: [batch, num_nodes, num_nodes, edge_dim]
    - Output: Attention-weighted node representations
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_heads: int = 8,
        edge_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_dim = edge_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Edge feature projection
        self.edge_proj = nn.Linear(edge_dim, num_heads)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
```

### Model Components

1. **EdgeAugmentedAttention**: Multi-head attention augmented with edge features
2. **FeedForward**: GELU-activated feed-forward network
3. **GraphTransformerBlock**: Combines attention + FFN with pre-normalization
4. **EdgeAugmentedGraphTransformer**: Full 6-layer transformer

### Training Data Format

```json
{
  "instructions": [
    {"mnemonic": "LOAD", "inputs": [...], "output": {...}},
    {"mnemonic": "INT_ADD", "inputs": [...], "output": {...}},
    {"mnemonic": "NOP", "inputs": [], "output": null}
  ],
  "edges": [[0, 1], [1, 2], [2, 3]],
  "edge_types": [0, 0, 1],
  "labels": [0, 0, 1, 0]
}
```

**Label meanings:**
- `0`: Genuine instruction (keep)
- `1`: Junk instruction (remove)

### Training Pipeline (`train_gnn.py`)

```python
class JunkInstructionDataset(Dataset):
    """
    Dataset for training junk instruction detector.
    
    Expected data format:
    - instructions: List of P-Code operations
    - edges: List of [src, dst] pairs
    - edge_types: List of edge type indices
    - labels: Binary labels (0=genuine, 1=junk)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_nodes: int = 512,
        augment: bool = True
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_nodes = max_nodes
        self.augment = augment
        
        # Instruction vocabulary
        self.mnemonic_to_idx = self._build_vocab()
        self.edge_type_to_idx = {
            'sequential': 0, 'branch': 1, 'call': 2,
            'return': 3, 'data_dep': 4, 'control_dep': 5
        }
        
        self.samples = self._load_samples()
```

### Training Configuration

```python
config = {
    'batch_size': 32,
    'learning_rate': 1e-4,
    'epochs': 100,
    'hidden_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'edge_dim': 64,
    'patience': 10,
    'weight_decay': 0.01
}
```

### Technical Details
- **Attention logits:** $$e_{ij}^{(h)} = \frac{Q_i^{(h)} K_j^{(h)T}}{\sqrt{d_h}} + W^{(h)} \cdot E_{ij}$$ where $E_{ij}$ is the edge embedding; attention weights $$\alpha_{ij}^{(h)} = \mathrm{softmax}_j(e_{ij}^{(h)})$$ and output $$O_i = \mathrm{Concat}_h \sum_j \alpha_{ij}^{(h)} V_j^{(h)}$$.
- **Edge channels:** `sequential`, `branch`, `call`, `return`, `data_dep`, `control_dep`; edge embeddings gate attention bias to favor true control/data flow and suppress junk blocks.
- **Loss:** Binary cross-entropy on per-node junk labels with class weights to offset ~20% junk ratio: $$\mathcal{L} = - w_1 y \log p - w_0 (1-y) \log(1-p).$$
- **Regularization:** Dropout 0.1, LayerNorm pre-norm, weight decay 0.01; early-stop on validation F1.
- **Inference:** Mask nodes with $p(\text{junk}) > 0.5$; removed indices are propagated to downstream services.

### API Endpoints

```
POST /sanitize
  Input: {"pcode": [...], "cfg": {...}}
  Output: {"sanitized_features": [...], "removed_indices": [...]}

GET /health
  Output: {"status": "ok", "service": "gnn-service"}
```

---

## Diffusion Service

### Location: `ai-services/diffusion-service/`

### Purpose
Generates C code from P-Code using Discrete Diffusion with Absorbing States (D3PM).

### Architecture

```python
class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
```

### D3PM Scheduler

```python
class D3PMScheduler:
    """
    Discrete Diffusion scheduler with absorbing state.
    
    Forward process: Gradually corrupt tokens to [MASK] token
    Reverse process: Predict original tokens from masked
    """
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        vocab_size: int = 32000,
        mask_token_id: int = 1,
        schedule: str = 'cosine'
    ):
        self.num_timesteps = num_timesteps
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        
        # Noise schedules
        if schedule == 'cosine':
            self.mask_probs = self._cosine_schedule()
        elif schedule == 'linear':
            self.mask_probs = self._linear_schedule()
        elif schedule == 'sqrt':
            self.mask_probs = self._sqrt_schedule()
```

### CodeDenoiser Architecture

```python
class CodeDenoiser(nn.Module):
    """
    Transformer denoiser for D3PM code generation.
    
    Takes noisy token sequence + timestep + P-Code context
    and predicts original clean tokens.
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 12,
        context_dim: int = 256,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embed = nn.Embedding(max_seq_len, hidden_dim)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Context projection
        self.context_proj = nn.Linear(context_dim, hidden_dim)
        
        # Transformer blocks with cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, context_dim, dropout)
            for _ in range(num_layers)
        ])
```

### Training Data Format

```json
{
  "pcode": ["LOAD", "INT_ADD", "STORE", "RETURN"],
  "code": "int add(int a, int b) {\n    return a + b;\n}"
}
```

### Training Pipeline (`train_diffusion.py`)

```python
class CodeGenerationDataset(Dataset):
    """
    Dataset for code generation training.
    
    Each sample contains:
    - P-Code operations (input)
    - C source code (target)
    """
    
    def __init__(
        self,
        data_dir: str,
        code_tokenizer: CodeTokenizer,
        pcode_vocab: Dict[str, int],
        split: str = 'train',
        max_code_len: int = 512,
        max_pcode_len: int = 256
    ):
        # Load and process samples
        self.samples = self._load_samples()
```

### Training Configuration

```python
config = {
    'batch_size': 16,
    'learning_rate': 1e-4,
    'epochs': 100,
    'num_timesteps': 1000,
    'hidden_dim': 512,
    'num_layers': 12,
    'num_heads': 8,
    'patience': 10
}
```

### Technical Details
- **Forward corruption (D3PM):** $$q(x_t \mid x_{t-1}) = (1-\beta_t)\,x_{t-1} + \beta_t\,[\text{MASK}]$$ with cosine/linear/sqrt schedules for $\beta_t$; cumulative $$\bar\alpha_t = \prod_{s=1}^t (1-\beta_s).$$
- **Reverse denoising:** Transformer predicts $$p_\theta(x_{t-1}=k \mid x_t, t, c)$$ with cross-entropy loss $$\mathcal{L} = -\sum_t \log p_\theta(x_{t-1} = x_0 \mid x_t).$$
- **Conditioning:** P-Code context projected to hidden_dim and injected via cross-attention each block; timestep embedding added to token embeddings.
- **Sampling:** Start from full [MASK] sequence, iterate $t=T\to0$; use classifier-free guidance by dropping context 10% of steps and blending logits.
- **Stability:** Clip logits to avoid degenerate collapse at early timesteps; gradient clipping 1.0; EMA of weights for sampling.

---

## Chain-of-Thought (CoT) Service

### Location: `ai-services/cot-service/`

### Purpose
Performs step-by-step reasoning for decompilation using LLM with chain-of-thought prompting.

### Architecture

```python
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
        
        # Load tokenizer and model
        self.tokenizer = self._load_tokenizer(model_name)
        self.model = self._load_model(model_name, adapter_path, use_quantization)
        
        # Reasoning prompts for each step
        self.step_prompts = self._create_step_prompts()
```

### Reasoning Steps

```python
class ReasoningStep(Enum):
    """Steps in chain-of-thought reasoning."""
    SIGNATURE = "function_signature"
    PARAMETERS = "parameters"
    LOCAL_VARS = "local_variables"
    CONTROL_FLOW = "control_flow"
    DATA_FLOW = "data_flow"
    SEMANTICS = "semantics"
    SYNTHESIS = "synthesis"
```

### Step Prompts

```python
step_prompts = {
    ReasoningStep.SIGNATURE: """Analyze the following P-Code/assembly to determine the function signature.

P-Code:
{pcode}

Step 1: Function Signature Analysis
- Look at CALL instructions to identify the function being defined
- Check stack setup (PUSH/POP patterns) for calling convention
- Analyze return instructions for return type

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
```

### Rule-Based Fallback

When no LLM is available, the service uses rule-based analysis:

```python
def _analyze_control_flow(self, pcode: str) -> str:
    """Analyze control flow from P-Code."""
    lines = pcode.strip().split('\n')
    
    has_branch = any('BRANCH' in line.upper() for line in lines)
    has_cbranch = any('CBRANCH' in line.upper() for line in lines)
    has_loop_pattern = has_cbranch and any('CMP' in line.upper() for line in lines)
    
    structure = []
    if has_loop_pattern:
        structure.append("for/while loop detected")
    if has_cbranch:
        structure.append("conditional branches (if/else)")
    
    return f"Control Flow Structure: {'; '.join(structure)}"
```

### Technical Details
- **Model loading:** 4-bit NF4 quantization by default with optional LoRA adapters; falls back to CPU fp16 if no GPU.
- **Stepwise decoding:** Runs prompts in fixed order (signature→parameters→locals→control_flow→data_flow→semantics→synthesis); each step uses top-k (k=40) + nucleus (p=0.9) sampling with temperature 0.4 for determinism.
- **Scoring:** Combines token logprobs and heuristic bonuses for balanced braces/indentation; penalizes hallucinated APIs not present in P-Code constants.
- **Fallback:** If a step fails or times out, a rule-based heuristic fills the missing field and subsequent steps consume the substituted text.

---

## LLM Service

### Location: `ai-services/llm-service/`

### Purpose
Decompiles P-Code to C using QLoRA fine-tuned CodeLlama with grammar-constrained generation.

### Training Architecture (`train_llm.py`)

```python
@dataclass
class TrainingConfig:
    """Configuration for LLM fine-tuning."""
    # Model
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
```

### Grammar-Constrained Generation

```python
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
```

### Decompiler Class

```python
class LLMDecompiler:
    """
    LLM-based decompiler using CodeLlama with QLoRA fine-tuning.
    
    Approach: Hierarchical Skeleton-Skin (SK2Decompile)
    1. Skeleton: Generate high-level structure (functions, control flow)
    2. Skin: Fill in detailed implementation
    """
    
    def decompile(self, sanitized_features, max_length=2048, use_grammar_constraints=True):
        """
        Decompile sanitized P-Code to C source code.
        
        For large functions, uses sliding window with overlap.
        """
        if len(sanitized_features) > 2048:
            return self._decompile_sliding_window(sanitized_features)
        
        # Format P-Code as input
        pcode_str = self._format_pcode(sanitized_features)
        
        prompt = f"""<s>[INST] Decompile the following binary code to readable C code.

Binary representation:
{pcode_str}

Generate clean, readable C code: [/INST]
"""
```

### Training Data Format

```json
{
  "pcode": ["LOAD", "INT_ADD", "STORE", "RETURN"],
  "code": "int add(int a, int b) {\n    // Add two integers\n    int result = a + b;\n    return result;\n}"
}
```

### Technical Details
- **QLoRA setup:** 4-bit NF4 weights with 16-rank LoRA on attention/MLP projections; effective hidden dim unchanged; grads limited to adapter weights.
- **Loss:** Standard cross-entropy over C tokens with teacher forcing; label smoothing 0.1; weight decay 0.01.
- **Grammar constraints:** LogitsProcessor masks tokens that would break brace/paren balance or invalidate statement endings; disallows generating disallowed tokens after type keywords.
- **Long-context:** Sliding-window decompile with 256-token overlap; merges windows by AST stitching, preferring higher per-token logprob spans.
- **Decoding defaults:** Temperature 0.3, top-p 0.9, max_new_tokens 512; retry with greedy + grammar constraints if syntax check fails.

---

## Multi-Agent Service

### Location: `ai-services/multi-agent-service/`

### Purpose
Uses 5 specialized agents with neural backbones + debate protocol for consensus-based decompilation.

### Real Analysis Module (`analysis.py`)

#### Basic Block Structure

```python
@dataclass
class BasicBlock:
    """Represents a basic block in the CFG."""
    id: int
    instructions: List[Dict]
    entry_address: int
    exit_address: int
    predecessors: List[int] = field(default_factory=list)
    successors: List[int] = field(default_factory=list)
    dominators: Set[int] = field(default_factory=set)
    is_loop_header: bool = False
    is_conditional: bool = False
```

#### CFG Builder

```python
class CFGBuilder:
    """Build Control Flow Graph from parsed instructions."""
    
    def build_cfg(self, instructions: List[Dict]) -> Tuple[Dict[int, BasicBlock], nx.DiGraph]:
        """Build CFG from instructions."""
        # Step 1: Identify basic block leaders
        leaders = self._find_leaders(instructions)
        
        # Step 2: Create basic blocks
        self._create_blocks(instructions, leaders)
        
        # Step 3: Connect blocks with edges
        self._connect_blocks(instructions)
        
        # Step 4: Compute dominators
        self._compute_dominators()
        
        # Step 5: Identify loop headers
        self._identify_loops()
        
        return self.blocks, self.graph
```

#### Data Flow Analyzer

```python
class DataFlowAnalyzer:
    """Perform data flow analysis on CFG."""
    
    def analyze(self) -> Dict:
        """Perform complete data flow analysis."""
        # Extract variables from instructions
        self._extract_variables()
        
        # Reaching definitions analysis
        self._compute_reaching_definitions()
        
        # Live variable analysis
        self._compute_live_variables()
        
        # Build def-use chains
        def_use_chains = self._build_def_use_chains()
        
        return {
            'variables': {name: vars(v) for name, v in self.variables.items()},
            'reaching_definitions': {...},
            'live_variables': {...},
            'def_use_chains': def_use_chains
        }
```

### Agent Architecture (`app.py`)

```python
class AgentModel(nn.Module):
    """
    Neural network backbone for agent analysis.
    Uses attention mechanism to process instruction sequences.
    """
    
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 256, 
                 num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_head = nn.Linear(hidden_dim, hidden_dim)
        self.confidence_head = nn.Linear(hidden_dim, 1)
```

### Specialized Agents

1. **StructureAgent**: Control flow and program structure analysis
2. **DataFlowAgent**: Variable usage and data dependencies
3. **TypeInferenceAgent**: Type recovery for variables
4. **MemoryAgent**: Stack/heap analysis
5. **OptimizationAgent**: Optimization pattern detection

```python
class StructureAgent(BaseAgent):
    """
    Agent specialized in control flow and program structure.
    Uses real CFG analysis to understand program structure.
    """
    
    def analyze(self, features: List[Dict], context: Dict) -> AgentProposal:
        """Analyze control flow structure using real CFG."""
        full_analysis = analyze_code(features)
        
        cfg = full_analysis.get('cfg', {})
        blocks = cfg.get('blocks', [])
        
        # Analyze loop structures
        loop_headers = [b for b in blocks if b.get('is_loop_header')]
        
        # Analyze conditionals
        conditionals = [b for b in blocks if b.get('is_conditional')]
        
        # Generate structured code based on CFG
        code = self._generate_structure_code(blocks, loop_headers, conditionals)
        
        return AgentProposal(
            agent_name=self.name,
            code=code,
            confidence=self._get_confidence(full_analysis),
            reasoning="...",
            analysis_data={'cfg': cfg}
        )
```

### Debate Protocol

```python
class MultiAgentDebate:
    """
    Multi-round debate protocol for agent consensus.
    """
    
    def conduct_debate(self, proposals: List[AgentProposal], max_rounds: int = 3) -> AgentProposal:
        """
        Conduct debate between agents until consensus.
        
        1. Each agent critiques others' proposals
        2. Agents revise based on critiques
        3. Voting for final proposal
        """
        for round in range(max_rounds):
            critiques = self._collect_critiques(proposals)
            proposals = self._revise_proposals(proposals, critiques)
            
            if self._check_consensus(proposals):
                break
        
        return self._vote_best_proposal(proposals)
```

    ### Technical Details
    - **Agents:** Structure/DataFlow/Type/Memory/Optimization agents each output (code, confidence, rationale, analysis_data). Backbone is a 2-layer Transformer encoder with shared weights; heads specialize.
    - **Voting:** Confidence-weighted mean score with tie-break by syntax validity; proposals exceeding `confidence_threshold` are preferred even if minority.
    - **Debate rounds:** At each round, critiques highlight CFG inconsistencies, undefined variables, and type conflicts; revisions patch the AST while preserving earlier agreements.
    - **Integration:** Debate output flows to RL verifier; if verifier fails, debate restarts with RL feedback appended as critique.

---

## Reinforcement Learning (RL) Service

### Location: `ai-services/rl-service/`

### Purpose
Provides PPO-based iterative refinement with Z3 symbolic verification.

### Training Module (`train_rl.py`)

#### State Encoder

```python
class CodeStateEncoder(nn.Module):
    """
    Encode decompilation state (P-Code + current C code) into embedding.
    
    Architecture:
    - Token embedding for P-Code instructions
    - Transformer encoder for sequence
    - Pooling for fixed-size representation
    """
    
    def __init__(self, vocab_size: int = 1024, embed_dim: int = 256,
                 num_heads: int = 4, num_layers: int = 3, max_seq_len: int = 512):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=0.1,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

#### Policy Network

```python
class PolicyNetwork(nn.Module):
    """
    Policy network for RL decompilation agent.
    
    Action space includes various code transformations:
    """
    
    ACTIONS = [
        'keep_current',          # 0: Keep current decompilation
        'add_type_cast',         # 1: Add type cast
        'remove_redundant',      # 2: Remove redundant code
        'fix_loop_bounds',       # 3: Fix loop bounds
        'add_null_check',        # 4: Add NULL check
        'fix_operator',          # 5: Fix operator
        'add_initialization',    # 6: Add variable initialization
        'fix_array_access',      # 7: Fix array access
        'simplify_expression',   # 8: Simplify expression
        'add_return',            # 9: Add return statement
        'fix_condition',         # 10: Fix conditional logic
        'regenerate',            # 11: Request full regeneration
    ]
    
    def __init__(self, state_dim: int = 256, hidden_dim: int = 512, num_actions: int = 12):
        super().__init__()
        
        # Shared trunk
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
```

#### PPO Trainer

```python
class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) trainer for RL decompilation.
    
    Features:
    - Clipped surrogate objective
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus for exploration
    """
    
    def __init__(
        self,
        state_encoder: CodeStateEncoder,
        policy: PolicyNetwork,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        value_clip_range: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01
    ):
        # Initialize components...
```

### Verification Service (`app.py`)

#### Neural-Symbolic Verifier

```python
class NeuralSymbolicVerifier:
    """
    Neural-Symbolic Execution Engine using Z3 Solver.
    
    Proves mathematical equivalence between original binary and decompiled C code.
    Uses symbolic execution to verify behavior across all input spaces.
    """
    
    def symbolic_execute(self, source_code: str, inputs: List[int]) -> Dict:
        """
        Symbolically execute C code and extract constraints.
        """
        self.reset()
        
        # Create symbolic variables
        symbolic_vars = {}
        for i, val in enumerate(inputs):
            symbolic_vars[f'input_{i}'] = z3.Int(f'input_{i}')
            self.solver.add(symbolic_vars[f'input_{i}'] == val)
        
        # Use full AST parser if available (pycparser)
        if PYCPARSER_AVAILABLE:
            return self._symbolic_execute_ast(source_code, symbolic_vars)
        else:
            return self._symbolic_execute_pattern(source_code, symbolic_vars)
    
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
        # Create symbolic inputs and try to find counterexample
        self.solver.add(binary_out != decompiled_out)
        
        result = self.solver.check()
        
        if result == z3.sat:
            # Found counterexample - not equivalent
            return {'equivalent': False, 'counterexample': ...}
        elif result == z3.unsat:
            # No counterexample exists - proven equivalent
            return {'equivalent': True, 'reason': 'No counterexample exists'}
```

#### Z3 AST Translation

```python
def _ast_to_z3(self, node, local_vars: Dict, symbolic_vars: Dict):
    """Convert AST expression to Z3 expression."""
    
    if isinstance(node, c_ast.BinaryOp):
        left = self._ast_to_z3(node.left, local_vars, symbolic_vars)
        right = self._ast_to_z3(node.right, local_vars, symbolic_vars)
        
        op_map = {
            '+': lambda l, r: l + r,
            '-': lambda l, r: l - r,
            '*': lambda l, r: l * r,
            '/': lambda l, r: l / r,
            '<': lambda l, r: l < r,
            '>': lambda l, r: l > r,
            '==': lambda l, r: l == r,
            '&&': lambda l, r: z3.And(l, r),
            '||': lambda l, r: z3.Or(l, r),
        }
        
        return op_map[node.op](left, right)
```

### Confidence Calibration

```python
class ConfidenceCalibrator:
    """
    Calibrates confidence scores using temperature scaling.
    """
    
    def calibrate_reward(self, raw_reward: float, confidence: float) -> float:
        """Apply temperature scaling to reward."""
        scaled_confidence = confidence ** (1.0 / self.temperature)
        calibrated_reward = raw_reward * scaled_confidence
        calibrated_reward = 11.0 / (1.0 + np.exp(-0.5 * (calibrated_reward - 5.5)))
        return float(calibrated_reward)
```

    ### Technical Details
    - **PPO objective:** $$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t, \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$ with $r_t$ the policy ratio and $A_t$ GAE advantages.
    - **Rewards:** $r = r_{\text{syntax}} + r_{\text{semantic}} + r_{\text{verify}} + r_{\text{length}}$, where verify reward is +5 for Z3 equivalence, -5 if counterexample found.
    - **Action space:** 12 discrete transformations; `regenerate` triggers a fresh LLM decode, others apply AST-local edits (e.g., add null checks, fix operators).
    - **Verifier loop:** Z3 solver runs with 60s timeout; if unsat → positive reward, sat → negative with counterexample returned to policy buffer.
    - **Stability:** KL penalty to base policy if KL > 0.01; gradient clip 0.5; entropy bonus 0.01 encourages exploration.

---

## CPG Service

### Location: `ai-services/cpg-service/`

### Purpose
Constructs enhanced Code Property Graphs with semantic analysis, abstract interpretation, and taint analysis.

### Data Structures

```python
class NodeType(Enum):
    INSTRUCTION = "instruction"
    BASIC_BLOCK = "basic_block"
    FUNCTION = "function"
    VARIABLE = "variable"
    CONSTANT = "constant"

class EdgeType(Enum):
    CFG = "cfg"  # Control flow
    AST = "ast"  # Syntax tree parent-child
    PDG_DATA = "pdg_data"  # Data dependency
    PDG_CONTROL = "pdg_control"  # Control dependency
    CALL = "call"  # Function call
    RETURN = "return"  # Return from function
    ALIAS = "alias"  # Pointer aliasing

@dataclass
class AbstractValue:
    """Abstract value for abstract interpretation."""
    is_constant: bool = False
    constant_value: Optional[int] = None
    is_tainted: bool = False
    taint_source: Optional[str] = None
    value_range: Tuple[Optional[int], Optional[int]] = (None, None)
```

### GNN for Pattern Detection

```python
class CPGEmbedding(nn.Module):
    """
    GNN-based embedding for CPG nodes.
    Uses message passing to capture structural context.
    """
    
    def __init__(self, node_features: int = 64, hidden_dim: int = 128, 
                 num_layers: int = 3, num_edge_types: int = 7):
        super().__init__()
        
        self.node_type_emb = nn.Embedding(10, node_features)
        self.mnemonic_emb = nn.Embedding(256, node_features)
        self.edge_type_emb = nn.Embedding(num_edge_types, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim + hidden_dim, hidden_dim),
                nn.ReLU(), nn.LayerNorm(hidden_dim)
            )
            for i in range(num_layers)
        ])
        
        # Pattern detection head
        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)  # 5 pattern types
        )
```

### Code Property Graph Class

```python
class CodePropertyGraph:
    """
    Enhanced multi-layered graph representation combining CFG, AST, PDG, and CG.
    """
    
    def build_from_pcode(self, pcode_ops: List[Dict], cfg: Dict) -> Dict:
        """Build comprehensive CPG from P-Code and CFG."""
        
        # Step 1: Create instruction nodes
        instruction_nodes = self._create_instruction_nodes(pcode_ops)
        
        # Step 2: Add CFG edges
        self._add_cfg_edges(instruction_nodes, cfg)
        
        # Step 3: Build basic blocks
        basic_blocks = self._identify_basic_blocks(instruction_nodes)
        
        # Step 4: Compute dominators and post-dominators
        self._compute_dominators(instruction_nodes)
        self._compute_post_dominators(instruction_nodes)
        
        # Step 5: Build def-use chains (data dependencies)
        self._build_def_use_chains(instruction_nodes, pcode_ops)
        
        # Step 6: Add control dependencies
        self._add_control_dependencies(instruction_nodes)
        
        # Step 7: Abstract interpretation
        self._abstract_interpret(instruction_nodes, pcode_ops)
        
        # Step 8: Taint analysis
        self._taint_analysis(instruction_nodes, pcode_ops)
        
        # Step 9: Detect semantic patterns
        patterns = self._detect_patterns(instruction_nodes, pcode_ops)
        
        # Step 10: Detect anomalies
        anomalies = self._detect_anomalies(instruction_nodes)
        
        return {
            'nodes': self._export_nodes(),
            'edges': self._export_edges(),
            'basic_blocks': basic_blocks,
            'dominators': {...},
            'post_dominators': {...},
            'def_use_chains': dict(self.def_use_chains),
            'abstract_values': {...},
            'taint_map': {...},
            'patterns': [...],
            'anomalies': anomalies
        }
```

### Dominator Computation

```python
def _compute_dominators(self, nodes: List[int]):
    """Compute dominator sets using iterative algorithm."""
    entry = nodes[0]
    all_nodes = set(nodes)
    
    # Initialize
    self.dominators = {entry: {entry}}
    for node in nodes[1:]:
        self.dominators[node] = all_nodes.copy()
    
    # Iterate until fixpoint
    changed = True
    while changed:
        changed = False
        for node in nodes[1:]:
            preds = [p for p in self.graph.predecessors(node)
                    if any(e.get('edge_type') == 'cfg' 
                          for e in self.graph.get_edge_data(p, node).values())]
            
            if preds:
                # Intersect predecessor dominators
                new_dom = all_nodes.copy()
                for pred in preds:
                    new_dom &= self.dominators[pred]
                new_dom.add(node)
                
                if new_dom != self.dominators[node]:
                    self.dominators[node] = new_dom
                    changed = True
```

### Technical Details
- **Abstract interpretation:** Interval domain tracks $(min, max)$ per variable; constant propagation marks `is_constant`; merges use join over intervals.
- **Taint analysis:** Sources are input params and external calls; taint flows through assignments and pointer aliases; `taint_map` stored per node.
- **Pattern detection:** Message passing GNN consumes node/edge embeddings to classify patterns (crypto loops, obfuscated branches, stack canaries); outputs logits for 5 pattern types.
- **Anomaly scoring:** Nodes with high taint + unusual control edges flagged; dominator/post-dominator mismatches highlight opaque predicates.

---

## Ghidra Service

### Location: `ai-services/ghidra-service/`

### Purpose
Extracts P-Code, CFG, and function information from binary files using Ghidra.

### Implementation

```python
@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze a binary file and extract P-Code, CFG, and function information.
    
    Request body:
    {
        "file_path": "/path/to/binary",
        "project_name": "analysis_project"
    }
    
    Returns:
    {
        "pcode": [...],
        "cfg": {...},
        "functions": [...]
    }
    """
    data = request.json
    file_path = data.get('file_path')
    project_name = data.get('project_name', 'temp_project')
    
    # Create temp project directory
    project_dir = f'/tmp/ghidra_projects/{project_name}'
    os.makedirs(project_dir, exist_ok=True)
    
    # Run Ghidra headless analyzer
    cmd = [
        f'{GHIDRA_INSTALL_PATH}/support/analyzeHeadless',
        project_dir,
        project_name,
        '-import', file_path,
        '-postScript', str(SCRIPT_PATH),
        '-scriptPath', str(SCRIPT_PATH.parent)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    
    # Parse output
    output_file = Path(project_dir) / f'{Path(file_path).stem}_analysis.json'
    
    if output_file.exists():
        with open(output_file) as f:
            analysis_data = json.load(f)
        return jsonify(analysis_data)
```

### Environment Variables

```
GHIDRA_INSTALL_PATH=/opt/ghidra
```

---

## Orchestrator Service

### Location: `ai-services/orchestrator/`

### Purpose
Coordinates the verify-refine loop pipeline across all services.

### Service Configuration

```python
# Service URLs - Advanced Architecture
GHIDRA_URL = os.getenv('GHIDRA_SERVICE_URL', 'http://ghidra-service:5001')
CPG_URL = os.getenv('CPG_SERVICE_URL', 'http://cpg-service:5005')
GNN_URL = os.getenv('GNN_SERVICE_URL', 'http://gnn-service:5002')
LLM_URL = os.getenv('LLM_SERVICE_URL', 'http://llm-service:5003')
RL_URL = os.getenv('RL_SERVICE_URL', 'http://rl-service:5004')
DIFFUSION_URL = os.getenv('DIFFUSION_SERVICE_URL', 'http://diffusion-service:5006')
MULTI_AGENT_URL = os.getenv('MULTI_AGENT_SERVICE_URL', 'http://multi-agent-service:5007')
COT_URL = os.getenv('COT_SERVICE_URL', 'http://cot-service:5008')

# Verify-Refine Loop Configuration
MAX_REFINEMENT_ITERATIONS = 3
REWARD_THRESHOLD = 10.5
```

### Main Pipeline

```python
@app.route('/sanitize', methods=['POST'])
def sanitize():
    """
    Advanced Verify-Refine Loop Pipeline:
    Binary → Ghidra → CPG → Graph Transformer → Hierarchical LLM → Symbolic Verifier
    
    With iterative refinement based on verification feedback.
    """
    
    # STEP 1: Ghidra Analysis - Extract P-Code
    ghidra_resp = safe_request(f'{GHIDRA_URL}/analyze', json_data={'file_path': file_path})
    analysis_data = ghidra_resp.json()
    
    # STEP 2: Build Code Property Graph (CPG)
    for func in analysis_data['functions']:
        cpg_resp = safe_request(f'{CPG_URL}/build-cpg', json_data={'pcode': pcode, 'cfg': cfg})
        cpg_analysis[func['name']] = cpg_resp.json()['cpg']
    
    # STEP 3: Edge-Augmented Graph Transformer - Detect obfuscation
    for func in analysis_data['functions']:
        gnn_resp = safe_request(f'{GNN_URL}/sanitize', json_data={...})
        sanitized_functions.append({
            'name': func['name'],
            'sanitized_features': gnn_resp.json()['sanitized_features']
        })
    
    # STEP 4 & 5: Verify-Refine Loop
    for iteration in range(max_iterations):
        # Decompile with LLM
        llm_resp = safe_request(f'{LLM_URL}/decompile-binary', json_data={'functions': sanitized_functions})
        decompiled = llm_resp.json()['decompiled']
        
        # Verify with symbolic execution
        for func_name, source_code in decompiled.items():
            verify_resp = safe_request(f'{RL_URL}/verify', json_data={
                'source_code': source_code,
                'original_binary_path': file_path,
                'use_symbolic': True
            })
            total_reward += verify_resp.json().get('reward', 0)
        
        # Check if acceptable quality
        if total_reward >= REWARD_THRESHOLD:
            break
        
        # Apply refinement (diffusion, multi-agent, or cot)
        refinement_method = ['diffusion', 'multi-agent', 'cot'][iteration % 3]
        # ... refinement calls ...
    
    return jsonify({
        'decompilation': best_decompilation,
        'final_reward': best_reward,
        'iterations_used': len(refinement_history)
    })
```

### Technical Details
- **Timeouts (s):** Ghidra 300, CPG 60, GNN 60, LLM 180, RL 60, Diffusion 120, Multi-Agent 120, CoT 60.
- **Refinement policy:** Iterate up to `MAX_REFINEMENT_ITERATIONS`; cycle refiners per iteration: 0→diffusion, 1→multi-agent, 2→CoT.
- **Acceptance:** Track `refinement_history`; accept best decompilation when total_reward ≥ 10.5 or budget exhausted; otherwise return highest-reward candidate.
- **Failure handling:** If a service fails, reuse last good artifacts (pcode/cpg/sanitized_features); loop continues unless first iteration fails critically.

---

## Complete Pipeline Workflow

### Step 1: Binary Ingestion (Ghidra)
```
Binary File → Ghidra Headless Analysis → P-Code + CFG + Functions
```

### Step 2: Graph Construction (CPG)
```
P-Code → Instruction Nodes → CFG Edges → Dominators → Data Flow → Control Dependencies
```

### Step 3: Sanitization (GNN)
```
P-Code Graph → Edge-Augmented Transformer → Junk Detection → Sanitized P-Code
```

### Step 4: Decompilation (LLM + Multi-Agent + CoT + Diffusion)
```
Sanitized P-Code → LLM Generation → Grammar Constraints → Initial C Code
                → Multi-Agent Debate → Consensus Refinement
                → CoT Reasoning → Step-by-step Analysis
                → Diffusion Model → Code Completion
```

### Step 5: Verification (RL + Z3)
```
C Code → Symbolic Execution → Z3 Constraints → Equivalence Proof
      → RL Policy → Refinement Actions → Iteration
```

### Step 6: Refinement Loop
```
Verification Feedback → Select Refinement Strategy → Re-decompile → Re-verify
                     → Until reward >= threshold OR max iterations
```

---

## Training Requirements

### Hardware Requirements

| Service | Minimum GPU | Recommended GPU | RAM | Storage |
|---------|-------------|-----------------|-----|---------|
| GNN | 8GB VRAM | 16GB VRAM | 16GB | 10GB |
| Diffusion | 16GB VRAM | 24GB VRAM | 32GB | 50GB |
| LLM | 16GB VRAM | 24GB VRAM | 32GB | 100GB |
| Multi-Agent | 8GB VRAM | 16GB VRAM | 16GB | 10GB |
| RL | 8GB VRAM | 16GB VRAM | 16GB | 20GB |

### Dataset Requirements

#### GNN Training Data
- **Format**: JSON with instructions, edges, edge_types, labels
- **Size**: 10,000+ samples
- **Label Ratio**: ~20% junk instructions

#### Diffusion Training Data
- **Format**: JSON with P-Code and C code pairs
- **Size**: 50,000+ samples
- **Code Length**: Up to 512 tokens

#### LLM Training Data
- **Format**: JSON with P-Code and documented C code
- **Size**: 100,000+ samples
- **Quality**: Well-commented, readable C code

#### RL Training Environment
- **Episodes**: 10,000+ complete decompilation episodes
- **Reward Signal**: Verification scores from Z3

### Training Commands

```bash
# GNN Training
cd ai-services/gnn-service
python train_gnn.py --data_dir /path/to/data --output_dir ./checkpoints --epochs 100

# Diffusion Training
cd ai-services/diffusion-service
python train_diffusion.py --data_dir /path/to/data --output_dir ./checkpoints --epochs 100

# LLM Fine-tuning
cd ai-services/llm-service
python train_llm.py --data_dir /path/to/data --output_dir ./checkpoints --epochs 3

# RL Training
cd ai-services/rl-service
python train_rl.py --episodes 10000 --output_dir ./checkpoints
```

---

## Post-Training Deployment

### Loading Trained Models

Each service automatically loads checkpoints from its `checkpoints/` directory:

```python
# GNN Service
model = EdgeAugmentedGraphTransformer(...)
model.load_state_dict(torch.load('checkpoints/best_model.pth'))

# LLM Service
model = PeftModel.from_pretrained(base_model, 'checkpoints/lora_adapter')

# RL Service
policy.load_state_dict(torch.load('checkpoints/policy_best.pth'))
encoder.load_state_dict(torch.load('checkpoints/encoder_best.pth'))
```

### Docker Deployment

```yaml
# docker-compose.yml
services:
  orchestrator:
    build: ./ai-services/orchestrator
    ports: ["5000:5000"]
  
  ghidra-service:
    build: ./ai-services/ghidra-service
    ports: ["5001:5001"]
    volumes:
      - /opt/ghidra:/opt/ghidra:ro
  
  gnn-service:
    build: ./ai-services/gnn-service
    ports: ["5002:5002"]
    deploy:
      resources:
        reservations:
          devices: [{capabilities: [gpu]}]
  
  # ... other services ...
```

---

## API Reference Summary

| Service | Endpoint | Method | Description |
|---------|----------|--------|-------------|
| Orchestrator | /sanitize | POST | Full pipeline |
| Orchestrator | /decompile | POST | Legacy direct decompile |
| Ghidra | /analyze | POST | Binary analysis |
| CPG | /build-cpg | POST | Build property graph |
| GNN | /sanitize | POST | Remove junk code |
| LLM | /decompile | POST | Generate C code |
| LLM | /decompile-binary | POST | Multi-function decompile |
| RL | /verify | POST | Symbolic verification |
| RL | /refine | POST | RL-based refinement |
| Diffusion | /generate | POST | Generate code |
| Diffusion | /refine | POST | Refine code |
| Multi-Agent | /analyze | POST | Multi-agent analysis |
| Multi-Agent | /refine | POST | Debate-based refinement |
| CoT | /reason | POST | Chain-of-thought |
| CoT | /analyze | POST | Step-by-step analysis |

---

## Conclusion

DeObfusca-AI provides a comprehensive neural-symbolic approach to binary deobfuscation. Each AI service is designed to be trainable, deployable, and interoperable. The verify-refine loop ensures high-quality decompilation through iterative improvement guided by symbolic verification.

For questions or contributions, please refer to the GitHub repository.
