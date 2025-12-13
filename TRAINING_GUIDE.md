# Training Guide

This is how you train the models. It's not complicated but it takes a while.

## Prerequisites

- Docker running
- 50GB free disk space
- 16GB+ RAM (32GB recommended)
- Python 3.11+

## Step 1: Start the services

```bash
docker-compose up -d
```

Check they're running:
```bash
docker-compose ps
```

You should see 12 containers. If any say "unhealthy", wait a minute and check again.

## Step 2: Download training data

```bash
python3 train_all_models.py --download-data
```

This downloads:
- OLLVM binaries (~200MB)
- AnghaBench source code (~150MB)
- Other datasets (~100MB)

Total is maybe 500MB. Should take 10-20 minutes depending on your internet.

## Step 3: Run training

```bash
python3 train_all_models.py --train-all --parallel
```

This trains all 6 models at once. Will take 8-12 hours depending on your hardware.

If you want to go slower and use less RAM:
```bash
python3 train_all_models.py --train-all --sequential
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
