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

## How it works

1. Upload a binary
2. Ghidra analyzes it and produces assembly + control flow graph
3. CPG service builds a code property graph
4. GNN processes the graph and produces embeddings
5. LLM generates C code based on embeddings
6. RL service verifies the code is correct using Z3
7. If verification fails, diffusion/multi-agent services try to refine the code
8. Repeat up to 3 times

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

This will take 8-12 hours depending on your hardware. See `TRAINING_GUIDE.md` for details.

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

## Notes

This project was built as part of SOC-D-2025. It's research code, not production-ready. Use at your own risk.

If something breaks, check the Docker logs:
```bash
docker-compose logs -f
```

Or check individual service logs:
```bash
docker logs deobfusca-ai-gnn-service-1
```
