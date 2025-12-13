# DeObfusca-AI: Complete Setup Guide

## System Overview

DeObfusca-AI is a full-stack AI-powered binary deobfuscation system with:
- **Node.js Backend**: Express.js REST API with Firebase integration
- **React Frontend**: Modern UI with authentication and project management
- **AI Services**: 5 microservices for binary analysis and decompilation
  - Ghidra Analyzer: Extracts P-Code and CFG from binaries
  - GNN Sanitizer: Detects junk instructions using graph neural networks
  - LLM Decompiler: Generates C code using CodeLlama/StarCoder
  - RL Verifier: Validates output through compilation and fuzzing
  - Orchestrator: Coordinates the full pipeline

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│   React     │─────▶│   Node.js    │─────▶│  Orchestrator   │
│  Frontend   │      │   Backend    │      │   (Flask)       │
└─────────────┘      └──────────────┘      └─────────────────┘
                            │                        │
                            ▼                        ▼
                     ┌──────────────┐      ┌─────────────────┐
                     │   Firebase   │      │  AI Services    │
                     │  Auth+Store  │      │  (4 services)   │
                     └──────────────┘      └─────────────────┘
```

## Prerequisites

- **Docker & Docker Compose**: Latest version
- **Node.js 18+**: For local development
- **Python 3.11+**: For AI services
- **NVIDIA GPU** (optional): For LLM/GNN acceleration
- **20GB+ Disk Space**: For models and dependencies

## Quick Start

### 1. Clone and Setup

```bash
cd /Users/chayanaggarwal/DeObfusca-AI

# Create data directories
mkdir -p data/uploads
mkdir -p data/ghidra_projects
mkdir -p ai-services/gnn-service/models
mkdir -p ai-services/llm-service/models
mkdir -p huggingface_cache
```

### 2. Configure Environment

```bash
# Backend Node.js
cp backend-node/.env.example backend-node/.env

# Edit backend-node/.env if needed (defaults work with Docker Compose)
```

### 3. Start Services

```bash
# Start all services
docker-compose up --build

# Or start in background
docker-compose up -d --build
```

### 4. Access Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **Firebase Emulator UI**: http://localhost:4000
- **Orchestrator**: http://localhost:5000
- **Ghidra Service**: http://localhost:5001
- **GNN Service**: http://localhost:5002
- **LLM Service**: http://localhost:5003
- **RL Service**: http://localhost:5004

## Service Details

### Node.js Backend (`backend-node/`)

**Technology**: Express.js, Firebase Admin SDK, Multer

**Endpoints**:
- `POST /api/auth/register` - User registration
- `GET /api/profile` - Get user profile
- `POST /api/projects` - Create project
- `GET /api/projects` - List projects
- `GET /api/projects/:id` - Get project details
- `DELETE /api/projects/:id` - Delete project
- `GET /api/projects/:id/jobs` - Get project jobs
- `POST /api/batch-upload` - Upload multiple files
- `POST /api/batch-upload-zip` - Upload ZIP archive
- `GET /api/jobs/:id` - Get job details
- `GET /api/history` - Get user history

**Key Files**:
- `server.js` - Main Express app
- `routes/` - API route handlers
- `middleware/auth.js` - JWT authentication

### Ghidra Service (`ai-services/ghidra-service/`)

**Purpose**: Binary analysis and P-Code extraction

**Technology**: Python Flask, Ghidra Headless Analyzer

**Endpoint**: `POST /analyze`

**Request**:
```json
{
  "file_path": "/data/uploads/binary.exe",
  "project_name": "analysis_project"
}
```

**Response**:
```json
{
  "program_name": "binary.exe",
  "functions": [
    {
      "name": "main",
      "entry_point": "0x1000",
      "pcode": [...],
      "cfg": {
        "nodes": [...],
        "edges": [...]
      }
    }
  ]
}
```

### GNN Service (`ai-services/gnn-service/`)

**Purpose**: Junk instruction detection using graph neural networks

**Technology**: PyTorch, PyTorch Geometric, Gated Graph Neural Networks

**Architecture**:
- Input: P-Code instructions + CFG
- Model: 6-layer GGNN with 128 hidden dims
- Output: Binary classification per instruction (real/junk)

**Endpoint**: `POST /sanitize`

**Request**:
```json
{
  "pcode": [...],
  "cfg": {
    "nodes": [...],
    "edges": [...]
  }
}
```

**Response**:
```json
{
  "sanitized_features": [...],
  "junk_indices": [5, 12, 23],
  "confidence_scores": [0.95, 0.87, ...],
  "original_count": 100,
  "sanitized_count": 78
}
```

**Training**:
```bash
# Copy training dataset
cp -r /path/to/ollvm_dataset data/training/obfuscated_binaries/

# Run training
docker-compose exec gnn-service python train.py
```

### LLM Service (`ai-services/llm-service/`)

**Purpose**: Decompilation using large language models

**Technology**: HuggingFace Transformers, QLoRA, CodeLlama-7B

**Model**: CodeLlama-7b with 4-bit quantization + QLoRA adapter

**Endpoint**: `POST /decompile`

**Request**:
```json
{
  "sanitized_features": [...]
}
```

**Response**:
```json
{
  "source": "int main() { ... }",
  "success": true,
  "input_length": 78
}
```

**Fine-tuning**:
```bash
# Prepare dataset (AnghaBench format)
# Each line: {"assembly": "...", "source_code": "..."}

# Run fine-tuning
docker-compose exec llm-service python fine_tune.py
```

### RL Service (`ai-services/rl-service/`)

**Purpose**: Verify decompilation through compilation + fuzzing

**Technology**: Python Flask, GCC, Subprocess fuzzing

**Reward Structure**:
- Compilation success: +0.5
- Behavioral match (90%+ tests): +10.0
- Compilation failure: -1.0

**Endpoint**: `POST /verify`

**Request**:
```json
{
  "source_code": "int main() { return 0; }",
  "original_binary_path": "/data/uploads/binary.exe"
}
```

**Response**:
```json
{
  "compilation_success": true,
  "execution_match": true,
  "reward": 10.5,
  "errors": []
}
```

### Orchestrator Service (`ai-services/orchestrator/`)

**Purpose**: Coordinate full deobfuscation pipeline

**Endpoint**: `POST /sanitize`

**Full Pipeline**:
1. Ghidra: Extract P-Code + CFG
2. GNN: Sanitize (remove junk)
3. LLM: Decompile to C
4. RL: Verify output

**Request**:
```json
{
  "file_path": "/data/uploads/binary.exe"
}
```

**Response**:
```json
{
  "features": {
    "original_count": 100,
    "sanitized_count": 78,
    "junk_indices": [...]
  },
  "source": "int main() { ... }",
  "verification": {
    "compilation_success": true,
    "execution_match": true,
    "reward": 10.5
  },
  "success": true
}
```

## Frontend Usage

### 1. Create Account

Navigate to http://localhost:5173/signup

### 2. Create Project

Dashboard → "Create New Project" → Enter name and description

### 3. Upload Files

Project Detail → Upload binary files or ZIP archive

### 4. View Results

Jobs list → Click job → View decompiled C code

### 5. Download

Click "Download C Source" button

## Production Deployment

### 1. Firebase Setup

```bash
# Create Firebase project at console.firebase.google.com
# Download service account key → backend-node/firebase-credentials.json

# Update backend-node/.env:
FIREBASE_CREDENTIALS=./firebase-credentials.json
# Remove FIRESTORE_EMULATOR_HOST and FIREBASE_AUTH_EMULATOR_HOST
```

### 2. Model Preparation

```bash
# GNN: Train model first
docker-compose exec gnn-service python train.py

# LLM: Download and fine-tune CodeLlama
docker-compose exec llm-service python fine_tune.py
```

### 3. GPU Configuration

Ensure `docker-compose.yml` has GPU support:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 4. Update Frontend API URL

```bash
# frontend/.env
VITE_API_URL=https://your-production-api.com
```

### 5. Deploy

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Development

### Backend Development

```bash
cd backend-node
npm install
npm run dev  # Uses nodemon for auto-reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

### AI Service Development

```bash
cd ai-services/gnn-service
pip install -r requirements.txt
python app.py  # Runs on port 5002
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose logs backend
docker-compose logs ghidra-service

# Restart specific service
docker-compose restart backend
```

### Firebase Connection Issues

```bash
# Verify emulator is running
docker-compose logs firebase-emulator

# Check emulator UI
open http://localhost:4000
```

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Check Docker GPU support
docker info | grep -i runtime
```

### Model Loading Errors

```bash
# LLM Service: Download model manually
docker-compose exec llm-service bash
huggingface-cli login
huggingface-cli download codellama/CodeLlama-7b-hf

# GNN Service: Check model path
ls -la ai-services/gnn-service/models/
```

## API Reference

See `docs/API_REFERENCE.md` for complete endpoint documentation.

## Training Datasets

### GNN Sanitizer

**Dataset**: OLLVM obfuscated binaries with ground truth

**Format**:
```json
{
  "node_features": [[...], [...]],
  "edge_index": [[0, 1], [1, 2], ...],
  "labels": [0, 1, 0, 0, 1, ...]
}
```

**Preparation**:
1. Compile programs with OLLVM `-mllvm -fla` (flattening)
2. Analyze with Ghidra to extract P-Code
3. Manually label junk instructions
4. Save as JSON in `data/training/obfuscated_binaries/`

### LLM Decompiler

**Dataset**: AnghaBench, Exampler, or custom pairs

**Format**: JSONL with assembly→source pairs
```json
{"assembly": "push rbp\nmov rbp, rsp\n...", "source_code": "int main() { return 0; }"}
```

**Preparation**:
```bash
# Use existing datasets
wget https://github.com/brenocfg/AnghaBench/archive/master.zip

# Or generate custom
gcc -S program.c        # Generate assembly
# Pair with original C source
```

## Performance Metrics

### Expected Throughput

- **Ghidra Analysis**: 10-30s per binary
- **GNN Sanitization**: 0.5-2s per function
- **LLM Decompilation**: 5-15s per function (GPU)
- **RL Verification**: 2-10s per program
- **End-to-End**: 30-60s per binary

### Accuracy Targets

- **GNN Junk Detection**: >90% precision/recall
- **Decompilation Quality**: 70-85% code similarity
- **RL Verification**: 80%+ compilation success

## Contributing

See `CONTRIBUTING.md` for guidelines.

## License

MIT License - See `LICENSE` file.

## Support

- GitHub Issues: Report bugs and request features
- Documentation: See `docs/` directory
- Email: support@deobfusca-ai.com
