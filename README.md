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

### Phase 5: Verification (Solving "Does it work?")
**Problem:** The code looks right but crashes or deviates in behavior.

**Solution:** Reinforcement Learning (RL) with compiler and execution feedback. Create a loop where the model generates decompiled code, you attempt to compile and run it, and reward the model based on compilation success and behavioral equivalence.

**Algorithm (Conceptual):**

- **Generate:** Model produces `decompile.c`.
- **Compile:** Run `gcc -c decompile.c`.
- **Fail:** Reward = `-1.0`.
- **Pass (compiles):** Reward = `+0.5`.
- **Fuzz:** Execute the original binary and the decompiled binary on the same inputs.
- **Output Match:** Reward = `+10.0`.
- **Update:** Use PPO (Proximal Policy Optimization) to adjust model weights from the collected rewards.

**Notes:**

- Use isolated sandboxes and time limits when running generated executables.
- Start with small unit tests and then scale to fuzzing for behavioral equivalence.

## Pillar 3: Recent Research (2024-2025) to Read
You must reference these to ensure your method is state-of-the-art.

- **"Nova: Generative Language Models for Binaries" (ICLR 2025)**
  - Why: Introduces Hierarchical Attention to handle the massive length of assembly code. Essential for Phase 4.
- **"SK2Decompile: Decompiling from Skeleton to Skin" (2025)**n  - Why: Proves that separating logic recovery from variable naming reduces hallucinations.
- **"DisasLLM: AI-driven Disassembly" (2024)**
  - Why: Solves the "Junk Byte" problem by using a small model to filter bytes before Ghidra sees them.
- **"Codealign: Instruction-Level Equivalence" (2025)**
  - Why: Solves the "Ground Truth" problem by aligning instructions based on execution traces rather than static position.

## Final Execution Checklist

- **Setup:** Install `Ghidra`, `PyTorch Geometric`, and `HuggingFace transformers`.
- **Data:** Spend 3 weeks generating the OLLVM/Tigress dataset — if the dataset is poor, the project will fail.
- **Sanitizer:** Train the GNN to >90% accuracy on trash detection before proceeding to decompilation.
- **Decompiler:** Fine-tune CodeLlama (or a similarly capable model) on the sanitized outputs.
- **RL:** Implement the compilation + fuzzing reward loop to fix syntax issues and improve behavioral equivalence.

