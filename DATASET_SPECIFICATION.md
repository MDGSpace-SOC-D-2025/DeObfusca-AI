# Dataset Specification for DeObfusca-AI Training

## Overview
This document defines the exact dataset format required to train all models in the DeObfusca-AI pipeline.

---

## Dataset Structure

```
training-data/
├── metadata.json                    # Dataset metadata
├── binaries/                        # Raw binary files
│   ├── sample_001.bin
│   ├── sample_002.bin
│   └── ...
├── ground_truth/                    # Verified source code
│   ├── sample_001.c
│   ├── sample_002.c
│   └── ...
├── preprocessed/                    # Preprocessed features
│   ├── gnn/                        # GNN training data
│   │   ├── sample_001.json
│   │   └── ...
│   ├── llm/                        # LLM training data
│   │   ├── sample_001.json
│   │   └── ...
│   ├── diffusion/                  # Diffusion training data
│   │   ├── sample_001.json
│   │   └── ...
│   └── rl/                         # RL training data
│       ├── sample_001.json
│       └── ...
└── splits.json                     # Train/val/test splits
```

---

## 1. Raw Data Format

### 1.1 Binary Files (`binaries/`)
- **Format**: ELF binaries (Linux x86-64)
- **Size**: 10KB - 10MB per binary
- **Content**: Compiled from C source with varying optimization levels
- **Obfuscation**: Mix of clean (30%), OLLVM (40%), commercial (30%)

### 1.2 Ground Truth Source (`ground_truth/`)
- **Format**: C source code (.c files)
- **Requirements**:
  - Must compile without errors using `gcc -std=c11`
  - Valid, readable C code (not assembly-like)
  - Properly formatted with consistent style
  - Contains function names, variable names, comments
- **Size**: 50-2000 lines per file

### 1.3 Metadata (`metadata.json`)
```json
{
  "dataset_version": "1.0.0",
  "created": "2025-12-21",
  "num_samples": 50000,
  "obfuscation_types": {
    "clean": 15000,
    "ollvm_bcf": 10000,
    "ollvm_fla": 10000,
    "ollvm_sub": 10000,
    "commercial": 5000
  },
  "compiler_settings": {
    "compiler": "gcc",
    "versions": ["9.4.0", "11.3.0"],
    "optimizations": ["-O0", "-O1", "-O2", "-O3", "-Os"]
  },
  "source_programs": {
    "coreutils": 5000,
    "openssl": 3000,
    "custom_algorithms": 12000,
    "real_world_malware": 30000
  }
}
```

---

## 2. Preprocessed Data Formats

### 2.1 GNN Training Data (`preprocessed/gnn/`)

**Purpose**: Train Graph Neural Network to identify junk instructions and generate embeddings.

**Format**: One JSON file per binary function

```json
{
  "sample_id": "sample_001_func_main",
  "binary_path": "binaries/sample_001.bin",
  "function_address": "0x401000",
  "function_name": "main",
  
  "node_features": [
    [0.1, 0.5, 0.0, ..., 0.3],  // 128-dim feature vector per node
    [0.2, 0.3, 0.1, ..., 0.4],
    ...
  ],
  
  "edge_index": [
    [0, 0, 1, 2, ...],  // Source nodes
    [1, 2, 2, 3, ...]   // Target nodes
  ],
  
  "edge_attr": [
    [1.0, 0.0],  // [control_flow, data_flow] per edge
    [1.0, 0.0],
    [0.0, 1.0],
    ...
  ],
  
  "labels": [0, 1, 0, 0, 1, ...],  // 0=real instruction, 1=junk
  
  "metadata": {
    "num_nodes": 150,
    "num_edges": 200,
    "num_junk": 30,
    "cyclomatic_complexity": 5,
    "obfuscation_type": "ollvm_bcf"
  }
}
```

**Node Features (128-dim)**:
- Opcode embedding (64-dim): One-hot or learned embedding of P-Code operation
- Operand features (32-dim): Number of operands, types, addressing modes
- Control flow (16-dim): Is branch, is loop header, dominator depth
- Data flow (16-dim): Def-use distance, liveness, register pressure

**Edge Types**:
- Control flow edges: Sequential, branch, loop back
- Data flow edges: Def-use, use-def chains
- Dominator edges: Immediate dominators

**Labels**:
- Binary classification: 0 (legitimate), 1 (junk/obfuscated)
- Ground truth obtained by comparing obfuscated vs clean CFG

---

### 2.2 LLM Training Data (`preprocessed/llm/`)

**Purpose**: Fine-tune CodeLlama for assembly-to-C translation.

**Format**: JSON Lines format (one JSON object per line)

```json
{
  "sample_id": "sample_001_func_main",
  "assembly": "push rbp\nmov rbp, rsp\nsub rsp, 16\nmov DWORD PTR [rbp-4], edi\ncmp DWORD PTR [rbp-4], 0\njle .L2\nmov eax, DWORD PTR [rbp-4]\nimul eax, eax\njmp .L3\n.L2:\nmov eax, 0\n.L3:\nleave\nret",
  "cfg_embedding": [0.15, -0.23, 0.87, ..., 0.45],  // 768-dim from GNN
  "source_code": "int square_positive(int x) {\n    if (x > 0) {\n        return x * x;\n    }\n    return 0;\n}",
  "metadata": {
    "num_instructions": 13,
    "num_basic_blocks": 4,
    "function_name": "square_positive",
    "has_loops": false,
    "has_conditionals": true
  }
}
```

**Training Format for HuggingFace**:
```json
{
  "input": "### Assembly:\n{assembly}\n\n### CFG Context:\n{cfg_summary}\n\n### C Code:",
  "output": "{source_code}",
  "cfg_embedding": [...]
}
```

---

### 2.3 Diffusion Training Data (`preprocessed/diffusion/`)

**Purpose**: Train diffusion model for code refinement.

**Format**: JSON with tokenized code and conditioning

```json
{
  "sample_id": "sample_001_func_main",
  
  "tokens": [150, 2341, 89, 234, 1043, 56, ...],  // Tokenized C code
  "token_length": 256,
  
  "condition": {
    "assembly_embedding": [0.15, -0.23, ..., 0.45],  // 768-dim
    "cfg_features": {
      "num_blocks": 4,
      "num_edges": 5,
      "complexity": 2,
      "has_loops": false
    },
    "llm_confidence": 0.85,
    "verification_hints": {
      "failed_constraints": [],
      "suggested_fixes": []
    }
  },
  
  "source_code": "int square_positive(int x) { ... }",
  
  "metadata": {
    "is_refinement": false,  // false=initial, true=refinement iteration
    "iteration": 0,
    "previous_errors": []
  }
}
```

**Tokenization**:
- Use CodeLlama tokenizer (vocab_size=50,000)
- Max length: 2048 tokens
- Padding: Right-pad with token_id=0

---

### 2.4 RL Training Data (`preprocessed/rl/`)

**Purpose**: Train PPO agent to select optimal refinement strategies.

**Format**: Episode trajectories

```json
{
  "sample_id": "sample_001_episode_001",
  
  "initial_state": {
    "pcode_features": [0.1, 0.5, ..., 0.3],  // 128-dim
    "cfg_complexity": 5,
    "num_constraints": 12,
    "initial_confidence": 0.75
  },
  
  "trajectory": [
    {
      "step": 0,
      "state": [0.1, 0.5, ..., 0.3],
      "action": 2,  // 0=llm_only, 1=diffusion, 2=multi_agent, 3=cot
      "reward": 5.5,
      "next_state": [0.2, 0.6, ..., 0.4],
      "done": false,
      "info": {
        "compilation_success": true,
        "z3_sat": true,
        "num_violations": 3
      }
    },
    {
      "step": 1,
      "state": [0.2, 0.6, ..., 0.4],
      "action": 1,
      "reward": 8.2,
      "next_state": [0.3, 0.7, ..., 0.5],
      "done": false,
      "info": {
        "compilation_success": true,
        "z3_sat": true,
        "num_violations": 1
      }
    },
    {
      "step": 2,
      "state": [0.3, 0.7, ..., 0.5],
      "action": 3,
      "reward": 10.5,
      "next_state": [0.4, 0.8, ..., 0.6],
      "done": true,
      "info": {
        "compilation_success": true,
        "z3_sat": true,
        "num_violations": 0
      }
    }
  ],
  
  "total_reward": 24.2,
  "num_steps": 3,
  "success": true
}
```

**State Features (128-dim)**:
- P-Code statistics (32-dim): Instruction counts, branch/loop ratios
- CFG features (32-dim): Complexity, depth, width
- Constraint features (32-dim): Number of constraints, SAT/UNSAT history
- Confidence scores (32-dim): Current confidence, historical performance

**Actions (4 strategies)**:
- 0: LLM-only refinement
- 1: Diffusion refinement
- 2: Multi-agent refinement
- 3: Chain-of-thought refinement

**Rewards**:
- Compilation success: +0.5
- Z3 satisfiability: +5.0
- Behavioral equivalence: +5.0
- Bonus for quick convergence: +2.0 * (1 - steps/max_steps)

---

## 3. Data Splits (`splits.json`)

```json
{
  "train": [
    "sample_001",
    "sample_002",
    ...
  ],
  "validation": [
    "sample_045001",
    "sample_045002",
    ...
  ],
  "test": [
    "sample_048001",
    "sample_048002",
    ...
  ],
  "split_ratios": {
    "train": 0.80,
    "validation": 0.10,
    "test": 0.10
  }
}
```

---

## 4. Minimum Dataset Sizes

| Component | Training | Validation | Test | Total |
|-----------|----------|------------|------|-------|
| **GNN** | 40,000 functions | 5,000 | 5,000 | 50,000 |
| **LLM** | 40,000 functions | 5,000 | 5,000 | 50,000 |
| **Diffusion** | 30,000 functions | 3,000 | 3,000 | 36,000 |
| **RL** | 20,000 episodes | 2,000 | 2,000 | 24,000 |

---

## 5. Data Collection Pipeline

### Step 1: Compile Source Programs
```bash
# Compile with different optimizations
for opt in O0 O1 O2 O3 Os; do
  gcc -$opt -o binary_${opt} source.c
done
```

### Step 2: Apply Obfuscation
```bash
# OLLVM obfuscation
clang -mllvm -bcf -mllvm -fla -mllvm -sub -o obfuscated source.c

# Or use Tigress, Themida, etc.
```

### Step 3: Extract Features with Ghidra
```python
# Run Ghidra headless to extract:
# - Disassembly
# - CFG
# - P-Code
# - Data flow
```

### Step 4: Build Ground Truth Labels
```python
# For GNN: Compare obfuscated vs clean CFG
# Mark nodes present in obfuscated but not in clean as junk
```

---

## 6. Data Quality Requirements

### Binary Quality
- ✅ Must be valid ELF format
- ✅ Must execute without crashes
- ✅ Must have symbol table (for verification)
- ✅ Function boundaries clearly defined

### Source Code Quality
- ✅ Must compile with gcc -std=c11 -Wall -Wextra
- ✅ No undefined behavior (checked with UBSAN)
- ✅ Properly formatted (clang-format)
- ✅ Includes function/variable names (not just mangled)

### Feature Quality
- ✅ No NaN or Inf values in embeddings
- ✅ Balanced dataset (50/50 junk vs real for GNN)
- ✅ Diverse obfuscation techniques
- ✅ Multiple difficulty levels

---

## 7. Example: Creating One Sample

```python
# 1. Write source code
source = """
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""

# 2. Compile
os.system("gcc -O2 -o factorial.bin factorial.c")

# 3. Obfuscate
os.system("ollvm -bcf -o factorial_obf.bin factorial.c")

# 4. Extract features with Ghidra
ghidra_output = ghidra.analyze("factorial_obf.bin")

# 5. Create GNN sample
gnn_sample = {
    "node_features": extract_node_features(ghidra_output),
    "edge_index": extract_edges(ghidra_output),
    "labels": label_junk_nodes(ghidra_output, clean_cfg)
}

# 6. Create LLM sample
llm_sample = {
    "assembly": ghidra_output['disassembly'],
    "cfg_embedding": gnn_model.encode(gnn_sample),
    "source_code": source
}

# 7. Save
save_json(gnn_sample, "preprocessed/gnn/factorial.json")
save_json(llm_sample, "preprocessed/llm/factorial.json")
```

---

## 8. Public Datasets (Starting Points)

### Recommended Sources:
1. **OLLVM Test Suite**: 500 obfuscated programs
2. **GNU Coreutils**: 100+ utility programs
3. **OpenSSL**: Crypto functions (complex logic)
4. **BinKit**: Binary similarity dataset (10K+ functions)
5. **Kaggle Malware Datasets**: Real-world obfuscated binaries

### Custom Generation:
- Generate synthetic C programs with known properties
- Apply systematic obfuscation transformations
- Ensures ground truth is 100% accurate

---

## Summary Checklist

For each sample, you need:
- [ ] Raw binary file (ELF)
- [ ] Ground truth C source code
- [ ] GNN features: nodes, edges, labels
- [ ] LLM features: assembly, CFG embedding, source
- [ ] Diffusion features: tokens, condition, metadata
- [ ] RL features: state, actions, rewards, trajectory

**Total Storage**: ~100GB for 50,000 samples
**Preprocessing Time**: ~48 hours on 64-core server
