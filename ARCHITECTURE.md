# DeObfusca-AI: Complete Architecture & Theoretical Analysis

## Executive Summary

DeObfusca-AI is a **state-of-the-art binary deobfuscation system** combining 9 specialized AI models in a Verify-Refine Loop architecture. This document provides complete architectural specifications, theoretical foundations, identified problems, and their solutions.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Backend Architecture](#backend-architecture)
3. [AI Pipeline Architecture](#ai-pipeline-architecture)
4. [Theoretical Foundations](#theoretical-foundations)
5. [Identified Problems & Solutions](#identified-problems--solutions)
6. [Service Specifications](#service-specifications)
7. [Data Flow & Protocols](#data-flow--protocols)
8. [Deployment & Operations](#deployment--operations)

---

## Backend Clarification

**There are TWO backend directories:**

### 1. `backend/` - **Legacy Python/FastAPI Backend (NOT IN USE)**
- **Status**:  Deprecated
- **Language**: Python with FastAPI
- **Purpose**: Original implementation, no longer maintained
- **Note**: Do NOT use this. It's kept for reference only.

### 2. `backend-node/` - **Active Node.js/Express Backend (PRODUCTION)**
- **Status**: Active & Production-Ready
- **Language**: Node.js with Express
- **Port**: 8000
- **Purpose**: Main API server for the application
- **Features**:
  - 12+ REST endpoints
  - Firebase Admin SDK integration
  - JWT authentication middleware
  - Project and job management
  - Batch file upload support
  - Profile management with separate histories

**Docker Compose uses `backend-node/` as the active backend.**

## Complete Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend Layer                           │
│  React + Vite + Apple-Style UI (Port 3000)                      │
│  - Theme System (Dark/Light Mode)                               │
│  - Glassmorphism Components                                     │
│  - Firebase Authentication                                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ HTTP REST API
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Layer (Node.js)                     │
│  Express API Server (Port 8000)                                 │
│  - Authentication & Authorization                               │
│  - Project/Job Management                                       │
│  - File Upload Handler                                          │
│  - Firestore Database Integration                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ HTTP POST
                     ↓
┌─────────────────────────────────────────────────────────────────┐
│                    AI Pipeline Orchestrator                      │
│  Flask Service (Port 5000)                                      │
│  - Verify-Refine Loop Coordinator                               │
│  - Service Health Monitoring                                    │
│  - Iterative Refinement Control                                 │
└───┬─────────┬─────────┬─────────┬─────────┬─────────────────────┘
    │         │         │         │         │
    │         │         │         │         │
    ↓         ↓         ↓         ↓         ↓
┌────────┐ ┌────┐ ┌─────┐ ┌─────┐ ┌────────┐
│ Ghidra │ │CPG │ │EAGT │ │ LLM │ │   RL   │
│ 5001   │ │5005│ │5002 │ │5003 │ │  5004  │
│        │ │    │ │     │ │     │ │        │
│ Binary │→│Hyper│→│Graph│→│Hier-│→│Neural- │
│Analysis│ │graph│ │Trans│ │arch │ │Symbolic│
│        │ │     │ │form │ │+RAG │ │  +Z3   │
└────────┘ └────┘ └─────┘ └─────┘ └────────┘
                                         ↑
                                         │
                                    Feedback Loop
                                    (RLHF Refinement)
```

## Service Ports

| Service | Port | Technology | Purpose |
|---------|------|------------|---------|
| Frontend | 3000 | React + Vite | User interface |
| Backend | 8000 | Node.js + Express | REST API |
| Firebase Emulator UI | 4000 | Firebase Tools | Emulator dashboard |
| Firestore | 8080 | Firebase | Database |
| Firebase Auth | 9099 | Firebase | Authentication |
| Orchestrator | 5000 | Flask (Python) | AI pipeline coordinator |
| Ghidra Service | 5001 | Flask (Python) | Binary analysis |
| GNN Service | 5002 | Flask (Python) | Graph transformer |
| LLM Service | 5003 | Flask (Python) | Hierarchical LLM |
| RL Service | 5004 | Flask (Python) | Verification |
| CPG Service | 5005 | Flask (Python) | Code property graphs |

## Data Flow

### 1. User Upload Flow
```
User → Frontend → Backend (8000) → Firestore
                    ↓
                Save file to /data/uploads
                    ↓
                Create job record
```

### 2. Deobfuscation Flow
```
Backend → Orchestrator (5000)
            ↓
        POST /sanitize
            ↓
┌───────────────────────────────┐
│  Verify-Refine Loop (Max 3x)  │
├───────────────────────────────┤
│ 1. Ghidra: Extract P-Code     │
│ 2. CPG: Build hypergraph      │
│ 3. EAGT: Detect junk code     │
│ 4. LLM: Decompile with RAG    │
│ 5. Grammar: Constrain syntax  │
│ 6. Z3: Verify equivalence     │
│    - Reward < 10.5? Refine    │
│    - Reward ≥ 10.5? Success   │
└───────────────────────────────┘
            ↓
    Return decompiled code
            ↓
Backend updates job status → Frontend displays result
```

## Database Schema (Firestore)

### Collections

**users**
```json
{
  "uid": "firebase-user-id",
  "email": "user@example.com",
  "displayName": "John Doe",
  "createdAt": "2024-01-01T00:00:00Z",
  "profiles": ["profile-1", "profile-2"]
}
```

**projects**
```json
{
  "id": "project-uuid",
  "userId": "firebase-user-id",
  "profileId": "profile-uuid",
  "name": "My Deobfuscation Project",
  "description": "Project description",
  "createdAt": "2024-01-01T00:00:00Z",
  "updatedAt": "2024-01-01T00:00:00Z"
}
```

**jobs**
```json
{
  "id": "job-uuid",
  "projectId": "project-uuid",
  "userId": "firebase-user-id",
  "profileId": "profile-uuid",
  "fileName": "binary.exe",
  "filePath": "/data/uploads/binary.exe",
  "status": "completed",
  "result": {
    "decompilation": { ... },
    "verification": { ... },
    "reward": 15.5
  },
  "createdAt": "2024-01-01T00:00:00Z",
  "completedAt": "2024-01-01T00:05:00Z"
}
```

**profiles**
```json
{
  "id": "profile-uuid",
  "userId": "firebase-user-id",
  "name": "Work Profile",
  "createdAt": "2024-01-01T00:00:00Z"
}
```

## Environment Configuration

### Development Mode (Docker Compose)
- Uses Firebase emulators (no internet required)
- All services run in containers with networking
- Hot reload for frontend and backend
- Sample data pre-populated

### Production Mode
- Requires real Firebase project
- Service account credentials needed
- Environment variables for production URLs
- HTTPS/TLS certificates required

## Network Architecture (Docker)

All services communicate via the `deobfusca-network` bridge:

```
deobfusca-network (bridge)
├── frontend (container: deobfusca-frontend)
├── backend (container: deobfusca-backend)
├── firebase-emulator (container: deobfusca-firebase)
├── orchestrator (container: deobfusca-orchestrator)
├── ghidra-service (container: deobfusca-ghidra)
├── cpg-service (container: deobfusca-cpg)
├── gnn-service
├── llm-service
└── rl-service
```

**Internal hostnames:**
- Services use container names for internal communication
- Example: Backend calls `http://orchestrator:5000/sanitize`

**External access:**
- Host machine accesses services via `localhost:PORT`
- Example: Browser accesses `http://localhost:3000`



### Quick Start (Development)

```bash
# 1. Copy environment files
cp backend-node/.env.example backend-node/.env
cp frontend/.env.example frontend/.env

# 2. Start all services
docker-compose up --build

# 3. Access the application
open http://localhost:3000
```

### Individual Service Development

**Frontend:**
```bash
cd frontend
npm install
npm run dev
# Access at http://localhost:5173
```

**Backend:**
```bash
cd backend-node
npm install
npm run dev
# Access at http://localhost:8000
```

**AI Service:**
```bash
cd ai-services/cpg-service
pip install -r requirements.txt
python app.py
# Access at http://localhost:5005
```

## Testing

**Frontend:**
```bash
cd frontend && npm test
```

**Backend:**
```bash
cd backend-node && npm test
```

**Integration Test:**
```bash
# Upload a binary via API
curl -X POST http://localhost:8000/api/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test-binary.exe"

# Check orchestrator health
curl http://localhost:5000/health
```

## Troubleshooting

### Backend won't start
- Check if port 8000 is available: `lsof -i :8000`
- Verify Firebase emulator is running: `curl http://localhost:4000`
- Check logs: `docker-compose logs backend`

### Frontend can't connect to backend
- Verify `VITE_API_URL` in `.env`: should be `http://localhost:8000`
- Check CORS settings in `backend-node/server.js`
- Ensure both services are on same network in docker-compose

### AI services timeout
- Services require significant resources (especially LLM/GNN)
- Check GPU availability: `nvidia-smi`
- Increase timeout in orchestrator if needed
- Check individual service logs: `docker-compose logs llm-service`

## Performance Optimization

### Frontend
- Code splitting with React.lazy()
- Image optimization
- Service worker for caching

### Backend
- Redis caching for frequent queries
- Database indexing on userId, projectId
- Request rate limiting

### AI Pipeline
- Model quantization (INT8)
- Batch processing for multiple files
- Result caching in Redis

## Security Considerations

### Production Checklist
- [ ] Change all default secrets/keys
- [ ] Use real Firebase project (not emulator)
- [ ] Enable HTTPS/TLS
- [ ] Set up proper CORS origins
- [ ] Implement rate limiting
- [ ] Add input validation
- [ ] Sanitize file uploads
- [ ] Set up monitoring/logging
- [ ] Configure firewall rules
- [ ] Regular security audits

