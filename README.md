This roadmap is divided into three pillars:

The Curriculum: Concepts and resources.

The Engineering Pipeline: Step-by-step implementation with code.

The Research Front: Key papers to replicate.

## Pillar 1: The Curriculum (What to Learn)
Before writing code, you need to master the intersection of Compiler Theory, Binary Analysis, and Deep Learning.

### Module A: Binary Analysis & Ghidra Internals

**Concept:** Intermediate Representations (IR). You cannot train models on raw assembly (too noisy). You must learn P-Code (Ghidra's RTL) or LLVM IR.

**Concept:** Headless Analysis. Analyzing 100k binaries requires automation, not a GUI.

**Resources:**

- Book: "The Ghidra Book" (specifically chapters on scripting and P-Code).
- Docs: Ghidra API: ghidra.program.model.pcode.
- Action: Write a script to print the P-Code of main.

### Module B: Adversarial Compilation (The Enemy)

**Concept:** Pass Management. How compilers transform code in stages. Obfuscation is just a malicious pass.

**Concept:** Control Flow Flattening (CFF) & Opaque Predicates. Understand mathematically how these destroy graph structures.

**Resources:**

- Source Code: Read BogusControlFlow.cpp in the Obfuscator-LLVM (OLLVM) repository.
- Paper: "Surreptitious Software" by Christian Collberg (The bible of obfuscation).

### Module C: Graph Neural Networks (The Sanitizer)

**Concept:** Message Passing. How a node (instruction) learns from its neighbors (control flow).

**Concept:** Gated Graph Neural Networks (GGNN). Best for sequential data flow on graphs.

**Resources:**

- Library: PyTorch Geometric (PyG).
- Tutorial: "Graph Neural Networks for Source Code" (Microsoft Research).

### Module D: Large Language Models (The Translator)

**Concept:** Instruction Tuning vs. Pre-training. You are fine-tuning, not pre-training.

**Concept:** Hierarchical Attention. How to handle 10,000 assembly tokens without running out of memory (Nova architecture).

**Resources:**

- Library: HuggingFace TRL (Transformer Reinforcement Learning).
- Technique: QLoRA (Quantized Low-Rank Adaptation).

## Pillar 2: The Implementation Pipeline (The Build)
This is the exact process to build the "Sanitize-then-Translate" architecture.

### Phase 1: The Data Foundry (Solving "Ground Truth")
**Problem:** How do we know which instruction is "trash" in a compiled binary? 

**Solution:** Modify the Obfuscator to leave "breadcrumbs" (metadata) that Ghidra can read, but the CPU ignores.

#### Step 1.1: Instrument OLLVM (C++)

We modify the OLLVM source code to tag "junk" blocks with a special LLVM metadata flag.

#### Step 1.2: Generate the Dataset

Compile 100,000 C functions (from AnghaBench or GNU Coreutils) using this custom OLLVM.

**Command:** `clang -O2 -mllvm -bcf -c sample.c -o sample.o`

**Result:** A binary where every "trash" instruction technically has a marker (or you extract the mapping during compilation to a JSON file).

### Phase 2: Feature Extraction (Solving "Garbage In")
**Problem:** Junk bytes crash standard disassemblers. 

**Solution:** Use a Headless Ghidra Script to parse the binary, extract the Control Flow Graph (CFG), and label nodes based on your OLLVM logs.

### Phase 3: The Neural Sanitizer (Solving "Graph Explosion")
**Problem:** Obfuscated graphs are messy. 

**Solution:** A Gated Graph Neural Network (GGNN). It looks at the topology (shapes) of the graph to find the "fake" loops created by OLLVM.

### Phase 4: The Neural Decompiler (Solving "Hallucination")
**Problem:** LLMs guess variable names wrong and lose track of long context. 

**Solution:**

- **SK2Decompile Approach:** Split generation into "Skeleton" (Logic) and "Skin" (Names).
- **Context:** Use a model with 16k+ context window (CodeLlama-Instruct).
