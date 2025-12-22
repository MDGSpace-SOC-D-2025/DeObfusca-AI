# Training Data & Scripts - Quick Reference

## What's Included

### 📄 Documentation
1. **DATASET_SPECIFICATION.md** - Exact data format for all models
2. **TRAINING_GUIDE.md** - Step-by-step training instructions
3. **TRAINING_SUMMARY.md** - Complete overview and workflow
4. **This file** - Quick reference

### 🔧 Scripts
1. **preprocess_data.py** - Convert raw binaries → training data
2. **train_all.py** - Automated training for all models

---

## Quick Start (3 Commands)

```bash
# 1. Preprocess your data
python3 preprocess_data.py --raw-dir ./raw_data --output-dir ./training-data

# 2. Validate dataset
python3 train_all.py --data-dir ./training-data --validate-only

# 3. Train all models
python3 train_all.py --data-dir ./training-data --output-dir ./models
```

---

## Data Format (Brief)

### Input (What You Need)
```
raw_data/
├── binaries/sample_001.bin      # ELF binary
└── ground_truth/sample_001.c    # Corresponding C source
```

### Output (What Gets Created)
```
training-data/
├── preprocessed/
│   ├── gnn/        # Graph features for GNN
│   ├── llm/        # Assembly-source pairs for LLM
│   ├── diffusion/  # Tokenized code for Diffusion
│   └── rl/         # Trajectories for RL
├── splits.json     # Train/val/test splits
└── metadata.json   # Dataset info
```

---

## Training Times (A100 GPU)

| Model | Time | Can Skip? |
|-------|------|-----------|
| GNN | 6 hours | ❌ Required |
| LLM | 72 hours | ✅ `--skip-llm` |
| Diffusion | 12 hours | ✅ `--skip-diffusion` |
| RL | 8 hours | ✅ `--skip-rl` |
| **Total** | **~4 days** | |

---

## Dataset Sizes

| Size | Training Time | Use Case |
|------|---------------|----------|
| 100 samples | 2 hours | Quick test |
| 1,000 samples | 6 hours | Development |
| 10,000 samples | 1 day | Small-scale |
| 50,000 samples | 4 days | Production |

---

## Common Commands

### Skip Slow Models
```bash
# Skip LLM (saves 72 hours)
python3 train_all.py --skip-llm

# Train only GNN (fastest)
python3 train_all.py --skip-llm --skip-diffusion --skip-rl
```

### Partial Processing
```bash
# Preprocess with custom split ratios
python3 preprocess_data.py \
    --raw-dir ./raw_data \
    --output-dir ./training-data \
    --train-ratio 0.7 \
    --val-ratio 0.15
```

### Monitor Training
```bash
# Watch logs
tail -f training.log

# Check GPU
watch -n 1 nvidia-smi
```

---

## Minimal Test Example

```bash
# Generate 100 test samples
mkdir -p raw_data/{binaries,ground_truth}
for i in {1..100}; do
    echo "int f$i(int x){return x*2;}" > raw_data/ground_truth/s$i.c
    gcc -o raw_data/binaries/s$i.bin raw_data/ground_truth/s$i.c
done

# Preprocess & train (skip LLM for speed)
python3 preprocess_data.py --raw-dir raw_data --output-dir data-mini
python3 train_all.py --data-dir data-mini --skip-llm
```

**Result**: Trained GNN, Diffusion, RL in ~2 hours

---

## Troubleshooting

### "Dataset validation failed"
→ Run: `python3 train_all.py --validate-only`
→ Check error messages
→ Review DATASET_SPECIFICATION.md

### "CUDA out of memory"
→ Reduce batch sizes in `train_all.py`:
- GNN: `batch_size = 16`
- Diffusion: `batch_size = 8`
- LLM: `batch_size = 2`

### "Preprocessing too slow"
→ Process subset of data first
→ Test with 100 samples
→ Scale up gradually

---

## File Locations

After training:
```
models/
├── gnn_best.pth           ← Copy to ai-services/gnn-service/
├── llm_final/             ← Copy to ai-services/llm-service/
├── diffusion_best.pth     ← Copy to ai-services/diffusion-service/
└── rl_*.pth               ← Copy to ai-services/rl-service/
```

---

## Data Sources (Recommended)

1. **BinKit**: Binary similarity dataset (10K+ functions)
2. **OLLVM Test Suite**: Pre-obfuscated binaries (500+)
3. **GNU Coreutils**: Real-world utilities (100+ programs)
4. **Custom Generation**: Write C code → compile → obfuscate

---

## Requirements

### Hardware
- **CPU**: 8+ cores
- **RAM**: 32GB+ (64GB recommended)
- **GPU**: 24GB+ VRAM (A100 40GB ideal)
- **Storage**: 200GB free space

### Software
```bash
pip install torch torchvision torchaudio
pip install torch-geometric transformers datasets
pip install numpy tqdm
```

---

## Next Steps

1. Read **DATASET_SPECIFICATION.md** for detailed format
2. Read **TRAINING_GUIDE.md** for full instructions
3. Prepare your raw data (binaries + source code)
4. Run preprocessing → validation → training
5. Deploy trained models to services

---

## Support

- **Format questions**: See DATASET_SPECIFICATION.md
- **Training issues**: See TRAINING_GUIDE.md
- **Overview**: See TRAINING_SUMMARY.md
- **Logs**: Check `training.log`

---

**Everything you need to train DeObfusca-AI from scratch!** 🚀
