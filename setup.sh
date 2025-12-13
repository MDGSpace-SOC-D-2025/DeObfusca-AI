#!/bin/bash

# DeObfusca-AI Quick Setup Script
# This script sets up the project for first-time use

set -e

echo "🚀 DeObfusca-AI Setup Script"
echo "================================"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Error: docker-compose is not installed."
    exit 1
fi

echo "✅ docker-compose is available"
echo ""

# Step 1: Copy environment files
echo "📝 Step 1: Setting up environment files..."

if [ ! -f backend-node/.env ]; then
    cp backend-node/.env.example backend-node/.env
    echo "✅ Created backend-node/.env"
else
    echo "⚠️  backend-node/.env already exists, skipping..."
fi

if [ ! -f frontend/.env ]; then
    cp frontend/.env.example frontend/.env
    echo "✅ Created frontend/.env"
else
    echo "⚠️  frontend/.env already exists, skipping..."
fi

echo ""

# Step 2: Create data directory
echo "📁 Step 2: Creating data directories..."
mkdir -p data/uploads
echo "✅ Created data/uploads directory"
echo ""

# Step 3: Pull required Docker images
echo "🐳 Step 3: Pulling Docker images (this may take a while)..."
docker-compose pull firebase-emulator
echo "✅ Firebase emulator image pulled"
echo ""

# Step 4: Build services
echo "🔨 Step 4: Building Docker services..."
echo "This will take 10-20 minutes on first run..."
docker-compose build
echo "✅ Services built successfully"
echo ""

# Step 5: Start services
echo "🎉 Step 5: Starting services..."
docker-compose up -d
echo "✅ Services started"
echo ""

# Wait for services to be ready
echo "⏳ Waiting for services to be healthy (30 seconds)..."
sleep 30

# Check service health
echo ""
echo "🏥 Checking service health..."
echo ""

# Check backend
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend (Node.js) - http://localhost:8000"
else
    echo "⚠️  Backend not responding yet"
fi

# Check frontend
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Frontend (React) - http://localhost:3000"
else
    echo "⚠️  Frontend not responding yet (may still be compiling)"
fi

# Check Firebase emulator
if curl -f http://localhost:4000 > /dev/null 2>&1; then
    echo "✅ Firebase Emulator UI - http://localhost:4000"
else
    echo "⚠️  Firebase Emulator not responding yet"
fi

# Check orchestrator
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ Orchestrator (AI Pipeline) - http://localhost:5000"
else
    echo "⚠️  Orchestrator not responding yet"
fi

echo ""
echo "================================"
echo "🎉 Setup Complete!"
echo "================================"
echo ""
echo "📱 Access the application:"
echo "   Frontend:  http://localhost:3000"
echo "   Backend:   http://localhost:8000"
echo "   Firebase:  http://localhost:4000"
echo ""
echo "📚 Next steps:"
echo "   1. Open http://localhost:3000 in your browser"
echo "   2. Create an account (using Firebase emulator)"
echo "   3. Upload a binary file for deobfuscation"
echo ""
echo "🛠️  Useful commands:"
echo "   View logs:     docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart:       docker-compose restart"
echo ""
echo "📖 Read ARCHITECTURE.md for detailed information"
echo ""
