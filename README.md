# DeObfusca-AI

Binary deobfuscation using neural networks and symbolic execution. Attempts to reverse engineer obfuscated binaries back to readable C code.

## What is this?

A platform that tries to automatically deobfuscate binaries using a combination of:
- Code Property Graphs to understand binary structure
- Neural networks (GNN + LLM) for code generation
- Z3 theorem prover to verify correctness
- Reinforcement learning to improve results

It's research-level work, so expect things to break sometimes.

## Prerequisites

- Docker & Docker Compose
- Node.js 18+ (if running frontend locally)
- Python 3.11+ (for running AI services locally)
- At least 50GB free disk space for training data

## Setup

```bash
git clone <repo>
cd DeObfusca-AI
docker-compose up --build
```

Then open http://localhost:3000 in your browser.

If Docker Compose fails, check that:
- Docker daemon is running
- You have 8GB+ RAM available
# DeObfusca-AI

A practical, notebook-first research project for binary deobfuscation. The repository focuses on a single Colab-ready notebook that demonstrates the working parts of the pipeline and contains prototypes for planned refinements.

The goal: convert obfuscated binaries to readable, verifiable C code using a hybrid pipeline of graph neural networks, external decompilers, symbolic verification, and reinforcement learning.

---

## Snapshot — Current Progress

- ✅ GNN junk-instruction detector: implemented and trained (edge-aware graph transformer; training and evaluation cells are in the notebook).
- ✅ SK2 (Snowman) decompiler integration: implemented and wired into the notebook pipeline; SK2 is the primary lifter (RetDec and pattern-based fallback available).
- ⚠️ RL structural recovery (opaque predicate removal & control-flow unflattening): prototype actions implemented in the notebook but not fully trained/validated.
- ⚠️ Multi-agent and diffusion refiners: partial implementations and helper utilities exist, but they require tuning and additional training.
- ❌ Large LLM training and full LLM-based decompilation: intentionally omitted due to cost; the pipeline uses SK2/RetDec for lifting instead.

Branch with these updates: `Revised-deobfusca` (remote branch contains notebook and README updates).

---

## Quick Start (notebook-first)

1. Open the notebook locally:

```bash
jupyter lab DeObfusca_AI_Complete.ipynb
```

2. Or upload `DeObfusca_AI_Complete.ipynb` to Google Colab and run the cells interactively.

What you’ll find in the notebook:

- Synthetic dataset generator and preprocessing utilities
- GNN model definition, training loop, and evaluation cells
- SK2 decompiler integration (assembly → temporary binary → Snowman) and pattern-fallbacks
- RL action prototypes for `PRUNE_OPAQUE`, `UNFLATTEN`, `CLEANUP`, `REFACTOR`
- Demo cell showing an end-to-end run on a toy example

---

## Revised Pipeline (implemented today)

1. **Binary analysis** (Ghidra or simulated)
   - Extract assembly and build CFG/CPG.

2. **GNN: Junk Detection & Sanitization**
   - Convert CFG → graph, make node-level predictions P(junk).
   - Remove high-confidence junk instructions (bogus control flow / opaque junk).

3. **SK2 Decompilation — Lifting**
   - Assemble cleaned instructions to a temporary binary and run Snowman (`nocode`).
   - Fallback chain: SK2 → RetDec → pattern-based fallback in the notebook.

4. **RL Structural Recovery (prototype)**
   - Actions implemented in the notebook:
     - `PRUNE_OPAQUE`: Remove branches Z3 proves unreachable.
     - `UNFLATTEN`: Detect and transform `while/switch` dispatcher patterns into structured control flow.
     - `CLEANUP`: Remove dead stores and trivial refactors.
     - `REFACTOR`: Request multi-agent formatting improvements.
   - Reward: compilation success + Z3 equivalence + complexity reduction.

5. **Z3 Verification**
   - Parse generated C (pycparser) and check semantic properties and path reachability.

6. **Iterate (bounded)**
   - Repeat refine & verify for a small number of rounds (configurable: default 3).

---

## Implementation Status — Detailed

- **GNN (Complete)**
  - Model: Edge-aware Graph Transformer
  - Training: Notebook contains dataset, training loop, metrics and checkpointing
  - Evaluation: Validation and test cells included

- **SK2 / Lifting (Complete)**
  - `SK2Decompiler` integrated in the notebook
  - Assembly → binary → SK2 flow with caching and fallbacks

- **RL Structural Recovery (Prototype)**
  - PPO scaffolding and action-space implemented in the notebook
  - Action handlers implemented as conservative transforms with basic Z3 checks
  - Pending: dataset & training loop validation

- **Multi-Agent & Diffusion (Partial)**
  - Helpers and model skeletons exist in `ai-services/` and notebook
  - Pending: training scripts, hyperparameter tuning, adversarial robustness tests

- **Orchestrator / Services (Deprecated for final)**
  - Microservice skeletons and Dockerfiles were removed to simplify final submission. The notebook is the canonical runbook.

---

## Files to keep (important)

- `DeObfusca_AI_Complete.ipynb` — main Colab / Jupyter notebook (final deliverable)
- `ai-services/gnn-service/` — GNN model & training helpers
- `ai-services/rl-service/` — RL training scripts and prototypes (for future work)
- `ai-services/diffusion-service/` — partial diffusion code (optional)
- `GNN_Architecture_and_Math.md` — supporting documentation and math notes

---

## How to reproduce the demo locally

1. Install prerequisites (minimal):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ai-services/gnn-service/requirements.txt
pip install -r ai-services/rl-service/requirements.txt
pip install pycparser z3-solver
```

2. Run the notebook or run the demo cell (the notebook also contains helper runner cells for the pipeline).

Note: Snowman (`nocode`) must be installed to run SK2 decompilation locally. If unavailable the notebook falls back to a pattern-based decompiler.

---

## Roadmap & Next Steps

1. Train & validate RL structural recovery on synthetic flattened/opaque samples (priority)
2. Expand Z3 precondition checks to make RL transforms provably safe
3. Add a small CI job that runs the notebook demo on a fixture binary
4. Optional: provide a small web demo (static hosting) for showcasing outputs

---

If you want, I can now:

- run the demo end-to-end on a few synthetic examples and save outputs,
- or create a small CI job that executes the demo cell and verifies the pipeline.

Tell me which you prefer and I’ll take the next step.

---

MIT License
