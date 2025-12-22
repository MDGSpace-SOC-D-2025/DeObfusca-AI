# Training Guide - Complete Pipeline

This guide shows you how to prepare data and train all models for DeObfusca-AI using the new automated training pipeline.

## Two Approaches

### Quick Approach (Legacy - For Docker Services)
Use the existing `train_all_models.py` for training within Docker containers.

### Comprehensive Approach (New - Full Pipeline)
Use the new `train_all.py` for complete control over training with custom datasets.

---

## Approach 1: Quick Docker Training (Legacy)

### Prerequisites
- Docker running
- 50GB free disk space
- 16GB+ RAM (32GB recommended)
- Python 3.11+

### Step 1: Start the services

```bash
docker-compose up -d
docker-compose ps  # Check all containers are running
```

### Step 2: Download training data

```bash
python3 train_all_models.py --download-data
```

Downloads OLLVM binaries, AnghaBench source (~500MB total).

### Step 3: Run training

```bash
# Train all models in parallel (8-12 hours)
python3 train_all_models.py --train-all --parallel

# Or sequential (slower, less RAM)
python3 train_all_models.py --train-all --sequential
```

---

## Approach 2: Comprehensive Training Pipeline (New)

### Prerequisites

**Software:**
```bash
pip install torch torchvision torchaudio
pip install torch-geometric transformers datasets
pip install numpy tqdm
```

**Hardware:**
- **Minimum**: 32GB RAM, 24GB GPU
- **Recommended**: 64GB RAM, A100 40GB GPU
- **Storage**: 200GB free space

### Step 1: Prepare Raw Data

**Option A: Create Custom Dataset**
```
raw_data/
├── binaries/           # ELF binaries (.bin files)
│   ├── sample_001.bin
│   ├── sample_002.bin
│   └── ...
└── ground_truth/       # Corresponding C source (.c files)
    ├── sample_001.c
    ├── sample_002.c
    └── ...
```

**Generate obfuscated binaries:**
```bash
# Using OLLVM
clang -mllvm -bcf -mllvm -fla -o obfuscated.bin source.c

# Or use Tigress, Themida, etc.
```

**Option B: Use Public Datasets**
- BinKit: Binary similarity dataset
- OLLVM Test Suite: Pre-obfuscated programs
- GNU Coreutils: Real-world utilities

### Step 2: Preprocess Data

```bash
python3 preprocess_data.py \
    --raw-dir ./raw_data \
    --output-dir ./training-data \
    --train-ratio 0.8 \
    --val-ratio 0.1
```

**Output structure:**
```
training-data/
├── preprocessed/
│   ├── gnn/          # Graph features
│   ├── llm/          # Assembly-source pairs
│   ├── diffusion/    # Tokenized code
│   └── rl/           # Episode trajectories
├── metadata.json
└── splits.json
```

**Preprocessing time:**
- 1,000 samples: ~30 minutes
- 10,000 samples: ~5 hours
- 50,000 samples: ~24 hours

### Step 3: Validate Dataset (Recommended)

```bash
python3 train_all.py \
    --data-dir ./training-data \
    --validate-only
```

Checks directory structure, JSON format, and data integrity.

### Step 4: Train All Models

```bash
# Full training (all models)
python3 train_all.py \
    --data-dir ./training-data \
    --output-dir ./models
```

**Training time (on A100 GPU):**
- GNN: ~6 hours (50 epochs)
- LLM: ~72 hours (3 epochs, 7B params)
- Diffusion: ~12 hours (50 epochs w/ adversarial)
- RL: ~8 hours (10K episodes)
- **Total: ~4 days**

**Skip models to save time:**
```bash
# Skip LLM (takes longest)
python3 train_all.py --skip-llm

# Train only GNN
python3 train_all.py --skip-llm --skip-diffusion --skip-rl
```

This trains one model at a time. Takes longer but uses less memory.

## Monitoring training

In another terminal, watch the logs:
```bash
tail -f /logs/training/orchestrator.log
```

Or if you want to watch resource usage:
```bash
watch -n 5 'docker stats --no-stream'
```

## If training fails

If it crashes or hangs, the logs will tell you why:

```bash
# See last 100 lines of training log
tail -100 /logs/training/orchestrator.log
```

If it ran out of memory, reduce batch sizes in `training_config.json` and try again.

If it's stuck, kill it and resume:
```bash
pkill -f train_all_models
python3 train_all_models.py --resume
```

## After training completes

Check that models exist:
```bash
ls -lh ai-services/gnn-service/models/
ls -lh ai-services/llm-service/models/
ls -lh ai-services/rl-service/models/
```

If they're there, you're done. Restart the services so they load the new models:
```bash
docker-compose restart
```

Then test a service:
```bash
curl http://localhost:5002/health
```

## Customizing training

Edit `training_config.json` to change:
- Learning rates
- Batch sizes
- Number of epochs
- Data sources

Example - faster training with lower quality:
```json
{
  "training": {
    "gnn": {"epochs": 20, "batch_size": 64},
    "llm": {"epochs": 1, "batch_size": 16},
    "rl": {"epochs": 10, "batch_size": 64}
  }
}
```

Then run with custom config:
```bash
python3 train_all_models.py --train-all --config training_config.json
```

## What actually gets trained

- **GNN**: Learns to understand binary structure and generate better embeddings
- **LLM**: Learns to generate more correct C code from embeddings
- **RL**: Learns to better verify if code is correct
- **Diffusion/Multi-Agent/CoT**: Learn to refine code iteratively

GNN and LLM are the most important. RL is crucial for verification.

## Notes

- Training is resource-intensive. Don't expect good results on a 4GB RAM machine.
- The models are decent after training but not perfect. Deobfuscation is a hard problem.
- If you're running on a system without much disk space, you can delete training data after training completes: `rm -rf /data/training/`
- Training times will vary wildly depending on hardware. 8 hours on a good machine, 24+ on an older one.
