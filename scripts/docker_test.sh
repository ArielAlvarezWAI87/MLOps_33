#!/bin/bash
# Test Docker container functionality

set -e

echo "=========================================="
echo "Docker Container Testing"
echo "=========================================="

echo ""
echo "Step 1: Building Docker image..."
docker build -t steel-energy-mlops:latest .

echo ""
echo "Step 2: Running tests in container..."
docker-compose run --rm test

echo ""
echo "Step 3: Verifying reproducibility in container..."
docker-compose run --rm verify-reproducibility

echo ""
echo "✅ All Docker tests passed!"
