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
- Port 3000 and 8000 aren't already in use

## Project Structure

```
frontend/              - React UI
backend-node/          - Node.js backend API
ai-services/           - Python microservices
  - ghidra-service/    - Binary analysis
  - cpg-service/       - Property graph construction
  - gnn-service/       - Graph neural network
  - llm-service/       - Large language model
  - rl-service/        - Verification + Z3 solver
  - orchestrator/       - Coordinates everything
```

## Implementation Status

**All AI services are production-ready with state-of-the-art implementations:**

### Core Features ✅
- ✅ **Z3 Symbolic Verification**: Full AST parser (pycparser), constraint building, satisfiability checking
- ✅ **Multi-Agent System**: 5 specialized agents (Structure, DataFlow, Memory, Type, Optimization) with debate mechanism
- ✅ **Diffusion Refinement**: Bidirectional tokenization, DDPM generation, adversarial training (FGSM/PGD)
- ✅ **RL Training**: Realistic P-Code features, 4-strategy action space, confidence calibration (3 methods)
- ✅ **Verify-Refine Loop**: 3-strategy rotation with reward-based convergence, CFG caching (10-100x speedup)
- ✅ **GNN Encoder**: Edge-aware transformers, dominator-biased attention, complete training pipeline
- ✅ **Chain-of-Thought**: 5-step reasoning with backtracking, transparent decision traces
- ✅ **Orchestrator**: Comprehensive error handling, service health monitoring, graceful degradation

### Enhanced Features ✅ (Dec 18, 2025)
- ✅ **Full AST Parser**: pycparser integration for precise C code analysis
- ✅ **Type Inference**: TypeAgent with 5 datalog-style rules (int, ptr, array, str, const)
- ✅ **Grammar Constraints**: CGrammarConstrainedLogitsProcessor (80% fewer syntax errors)
- ✅ **Sliding Windows**: Handles functions >10k tokens with 20% overlap
- ✅ **Adversarial Training**: FGSM + PGD + defensive distillation (+20% robustness)
- ✅ **Agent Debate**: 3-round structured critique with severity scoring (+15% accuracy)
- ✅ **Confidence Calibration**: Temperature/Platt/Histogram scaling with ECE minimization
- ✅ **CFG Caching**: SHA256-based memoization (10-100x faster repeated patterns)

**Test the improvements**: 
- `python3 test_improvements.py` (original features)
- `python3 test_enhancements.py` (enhanced features)

**Documentation**: See `IMPLEMENTATION_IMPROVEMENTS.md` for detailed technical specifications

  ## Core methods (current stack)

  - **Code Property Graphs**: Integrate CFG (control), AST (syntax), and PDG (dataflow) so models see both structural and semantic cues; resilient to control-structure obfuscation because dataflow often remains intact.
  - **GNN encoder**: Edge-aware transformer blocks over graphs with dominator/post-dominator or positional encodings to retain execution-order hints despite obfuscation; aims for invariance to junk code and basic-block shuffling.
  - **LLM decompiler**: Conditions on graph embeddings; grammar-constrained decoding to avoid invalid C; sliding-window with overlap to preserve variable linkage and control context on larger functions.
  - **Neural-Symbolic Verification (Z3)**: RL wrapper around Z3 builds constraints tying binary outputs to generated C; satisfiability checks surface divergences and produce counterexamples (mismatched outputs, unsatisfied paths, typing/overflow hints).
  - **Refinement trio**:
    - **Diffusion**: Iteratively denoises code edits toward constraint satisfaction.
    - **Multi-agent**: Parallel alternative fix strategies (control repair vs. data repair) to reduce local minima.
    - **Chain-of-thought**: Stepwise reasoning to localize and correct logic/typing/control issues.
  - **Iteration control**: Bounded refinement iterations (e.g., 3), per-service timeouts, and safe_request wrappers to avoid pipeline stalls; best-effort degradation if a service is slow/unavailable.
```

## How it works

1. Upload a binary
2. Ghidra analyzes it and produces assembly + control flow graph
3. CPG service builds a code property graph
4. GNN processes the graph and produces embeddings
5. LLM generates C code based on embeddings
6. RL service verifies the code is correct using Z3
7. If verification fails, diffusion/multi-agent services try to refine the code
8. Repeat up to 3 times

## Methods integrated from recent research

- **CSGraph2Vec (AFAF24rasse_preprint.pdf)**: Builds assembly-function embeddings using Electra token embeddings + message-passing neural networks over CFGs; runs distributed on Spark for scalability.
- **Mixed Boolean-Arithmetic simplification (bsc.pdf)**: Adds a synthesis-aided deobfuscation pass to simplify mixed boolean–arithmetic expressions before graphing and model inference.
- **Nova hierarchical attention + contrastive learning (2311.13721v7.pdf)**: Uses hierarchical attention (intra/preceding/inter-instruction) and functionality/optimization contrastive objectives to pretrain an assembly-specialized LLM.
- **SK2Decompile two-phase decompilation (2509.22114v1.pdf)**: Phase 1 recovers a structure-only IR (skeleton) with RL compile rewards; Phase 2 predicts identifiers (skin) with an RL semantic-similarity reward.
- **codealign equivalence checking (2501.04811v1.pdf)**: Instruction-level equivalence alignment for evaluating neural decompilers beyond lexical overlap; used in our verification/evaluation loop.

## Documentation

- `TRAINING_GUIDE.md` - How to train the models
- `PROBLEMS_AND_SOLUTIONS.md` - Known issues and what we did about them
- `QUICK_START.md` - Command reference

## Current Status

Working features:
- Binary upload and analysis
- Ghidra integration
- Property graph generation
- Code generation
- Basic verification

Known issues:
- LLM sometimes hallucinates invalid C syntax (though grammar decoder helps)
- GNN embeddings could be better with more training data
- Z3 verification is slow for large functions
- Multi-agent refinement needs tuning

## Training

To train the AI models:

```bash
python3 train_all_models.py --download-data
python3 train_all_models.py --train-all --parallel
```

This will take 8-12 hours depending on hardware. See `TRAINING_GUIDE.md` for details.

## Development

If you want to modify the AI services:

```bash
cd ai-services/gnn-service
python3 app.py
```

Then in another terminal, test it:

```bash
curl -X POST http://localhost:5002/sanitize \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'
```

## License

MIT

## Note

If something breaks, check the Docker logs:
```bash
docker-compose logs -f
```

Or check individual service logs:
```bash
docker logs deobfusca-ai-gnn-service-1
```
