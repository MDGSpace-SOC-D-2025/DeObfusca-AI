# Quick Reference

## Start everything

```bash
docker-compose up -d
```

## Check services are running

```bash
docker-compose ps
docker-compose logs
```

## Train models

```bash
# Download training data
python3 train_all_models.py --download-data

# Train all models (parallel, takes 8-12 hours)
python3 train_all_models.py --train-all --parallel

# Train models sequentially (slower but less RAM)
python3 train_all_models.py --train-all --sequential
```

## Test a service

```bash
# Test GNN
curl -X POST http://localhost:5002/sanitize \
  -H "Content-Type: application/json" \
  -d '{"features": [[1.0, 2.0, 3.0]]}'

# Test orchestrator health
curl http://localhost:5010/health
```

## Debug issues

```bash
# Watch logs
docker-compose logs -f

# Check specific service
docker logs deobfusca-ai-gnn-service-1

# See what's using resources
docker stats

# Kill a service and restart it
docker-compose restart gnn-service
```

## Development

```bash
# Build a specific service
docker-compose build gnn-service

# Rebuild all
docker-compose build

# Run service locally (not in Docker)
cd ai-services/gnn-service
python3 app.py
```

## Common problems

**Port 3000 already in use**:
```bash
lsof -i :3000
kill -9 <PID>
```

**Out of disk space during training**:
```bash
# Clean up Docker
docker system prune -a

# Or free up space manually
rm -rf /data/training/*
```

**Service not starting**:
```bash
# Check logs
docker logs <container_name>

# Rebuild from scratch
docker-compose down
docker-compose build --no-cache
docker-compose up
```

**Training stuck**:
```bash
# Kill training
pkill -f train_all_models

# Resume
python3 train_all_models.py --resume
```

## Useful commands

```bash
# Remove all Docker containers/images (careful!)
docker system prune -a --volumes

# See what's running
docker ps

# See all containers (including stopped)
docker ps -a

# Inspect a container
docker inspect <container_id>

# Execute command in running container
docker exec <container_id> ls /app

# Copy file from container
docker cp <container_id>:/path/to/file /local/path
```
