#!/bin/bash
# Build Docker image for reproducible MLOps environment

set -e

echo "=========================================="
echo "Building Docker Image"
echo "=========================================="

# Build the image
docker build -t steel-energy-mlops:latest .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "Image details:"
docker images steel-energy-mlops:latest

echo ""
echo "Available commands:"
echo "  docker-compose run test                    # Run tests"
echo "  docker-compose run train                   # Run training pipeline"
echo "  docker-compose run verify-reproducibility  # Verify reproducibility"
echo "  docker-compose up api                      # Start API server"
