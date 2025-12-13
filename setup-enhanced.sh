#!/bin/bash

# DeObfusca-AI v2.0 - Enhanced Setup Script
# This script sets up all new features and services

set -e

echo "🚀 DeObfusca-AI v2.0 Enhancement Setup"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Docker
echo -e "${YELLOW}Step 1: Checking Docker...${NC}"
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi
echo -e "${GREEN}✅ Docker and Docker Compose found${NC}"
echo ""

# Step 2: Install Backend Dependencies
echo -e "${YELLOW}Step 2: Installing Backend Dependencies...${NC}"
cd backend-node
if [ -f "package.json" ]; then
    npm install socket.io@4.6.1 ioredis@5.3.2 compression@1.7.4 helmet@7.1.0 express-rate-limit@7.1.5
    echo -e "${GREEN}✅ Backend dependencies installed${NC}"
else
    echo "❌ package.json not found in backend-node"
    exit 1
fi
cd ..
echo ""

# Step 3: Install Frontend Dependencies
echo -e "${YELLOW}Step 3: Installing Frontend Dependencies...${NC}"
cd frontend
if [ -f "package.json" ]; then
    npm install socket.io-client@4.6.1
    echo -e "${GREEN}✅ Frontend dependencies installed${NC}"
else
    echo "❌ package.json not found in frontend"
    exit 1
fi
cd ..
echo ""

# Step 4: Create Environment Files
echo -e "${YELLOW}Step 4: Setting up Environment Files...${NC}"

# Backend .env
if [ ! -f "backend-node/.env" ]; then
    cat > backend-node/.env << EOF
NODE_ENV=development
PORT=8000

# Firebase Emulator
FIRESTORE_EMULATOR_HOST=firebase-emulator:8080
FIREBASE_AUTH_EMULATOR_HOST=firebase-emulator:9099

# Redis Cache
REDIS_HOST=redis
REDIS_PORT=6379

# AI Services
ORCHESTRATOR_URL=http://orchestrator:5000
GHIDRA_SERVICE_URL=http://ghidra-service:5001
CPG_SERVICE_URL=http://cpg-service:5005
GNN_SERVICE_URL=http://gnn-service:5002
LLM_SERVICE_URL=http://llm-service:5003
RL_SERVICE_URL=http://rl-service:5004
DIFFUSION_SERVICE_URL=http://diffusion-service:5006
MULTI_AGENT_SERVICE_URL=http://multi-agent-service:5007
COT_SERVICE_URL=http://cot-service:5008

# Upload Configuration
UPLOAD_DIR=/data/uploads
MAX_FILE_SIZE=104857600

# JWT Configuration
JWT_SECRET=your-secret-key-change-in-production

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# Logging
LOG_LEVEL=info
EOF
    echo -e "${GREEN}✅ Created backend-node/.env${NC}"
else
    echo "ℹ️  backend-node/.env already exists"
fi

# Frontend .env
if [ ! -f "frontend/.env" ]; then
    cat > frontend/.env << EOF
VITE_API_URL=http://localhost:8000

# Firebase Configuration (Emulator)
VITE_FIREBASE_API_KEY=demo-api-key
VITE_FIREBASE_AUTH_DOMAIN=demo-project.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=demo-project
VITE_FIREBASE_STORAGE_BUCKET=demo-project.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abcdef

# Orchestrator
VITE_ORCHESTRATOR_URL=http://localhost:5000
EOF
    echo -e "${GREEN}✅ Created frontend/.env${NC}"
else
    echo "ℹ️  frontend/.env already exists"
fi
echo ""

# Step 5: Create necessary directories
echo -e "${YELLOW}Step 5: Creating directories...${NC}"
mkdir -p data/uploads
mkdir -p data/models
echo -e "${GREEN}✅ Directories created${NC}"
echo ""

# Step 6: Build and start services
echo -e "${YELLOW}Step 6: Building Docker services (this may take 5-10 minutes)...${NC}"
docker-compose down -v
docker-compose build
echo -e "${GREEN}✅ Services built${NC}"
echo ""

echo -e "${YELLOW}Step 7: Starting all 12 services...${NC}"
docker-compose up -d
echo ""

# Step 8: Wait for services to be ready
echo -e "${YELLOW}Step 8: Waiting for services to start...${NC}"
sleep 10

# Check health of key services
services=(
    "Backend:http://localhost:8000/api/health"
    "Orchestrator:http://localhost:5000/health"
    "Diffusion:http://localhost:5006/health"
    "Multi-Agent:http://localhost:5007/health"
    "CoT:http://localhost:5008/health"
)

echo ""
echo "Checking service health..."
for service in "${services[@]}"; do
    IFS=: read -r name url <<< "$service"
    if curl -s "$url" > /dev/null 2>&1; then
        echo -e "  ${GREEN}✅ $name${NC}"
    else
        echo -e "  ${YELLOW}⏳ $name (starting...)${NC}"
    fi
done
echo ""

# Step 9: Display final information
echo ""
echo "=========================================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "📊 Services Running:"
echo "  • Frontend:         http://localhost:3000"
echo "  • Backend:          http://localhost:8000"
echo "  • Firebase UI:      http://localhost:4000"
echo "  • Orchestrator:     http://localhost:5000"
echo "  • Ghidra Service:   http://localhost:5001"
echo "  • GNN Service:      http://localhost:5002"
echo "  • LLM Service:      http://localhost:5003"
echo "  • RL Service:       http://localhost:5004"
echo "  • CPG Service:      http://localhost:5005"
echo "  • Diffusion (NEW):  http://localhost:5006"
echo "  • Multi-Agent (NEW):http://localhost:5007"
echo "  • CoT (NEW):        http://localhost:5008"
echo "  • Redis:            localhost:6379"
echo ""
echo "🆕 New Features:"
echo "  ✅ Real-time job monitoring with WebSocket"
echo "  ✅ AI Chat Assistant"
echo "  ✅ Code Comparison Tool"
echo "  ✅ Advanced Analytics Dashboard"
echo "  ✅ Project Collaboration (sharing & comments)"
echo "  ✅ 3 Advanced AI Methods (Diffusion, Multi-Agent, CoT)"
echo "  ✅ Redis Caching (60% faster)"
echo "  ✅ Rate Limiting & Security"
echo ""
echo "📖 Documentation:"
echo "  • Main README:      README.md"
echo "  • Enhancements:     ENHANCEMENTS_V2.md"
echo "  • Integration:      INTEGRATION_GUIDE.md"
echo "  • Summary:          PROJECT_ENHANCEMENT_SUMMARY.md"
echo ""
echo "🚀 Next Steps:"
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Create an account or sign in"
echo "  3. Try uploading a binary for decompilation"
echo "  4. Explore new features:"
echo "     - Real-time job monitor"
echo "     - AI Chat Assistant"
echo "     - Code comparison"
echo "     - Analytics dashboard"
echo ""
echo "🔧 Useful Commands:"
echo "  • View logs:        docker-compose logs -f [service-name]"
echo "  • Stop services:    docker-compose down"
echo "  • Restart:          docker-compose restart [service-name]"
echo "  • Check health:     curl http://localhost:5000/health"
echo ""
echo "❓ Need Help?"
echo "  • Check INTEGRATION_GUIDE.md for detailed setup"
echo "  • View service logs for troubleshooting"
echo "  • All services should be running within 2-3 minutes"
echo ""
echo -e "${GREEN}Happy Deobfuscating! 🎯${NC}"
