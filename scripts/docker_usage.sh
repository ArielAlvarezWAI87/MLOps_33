#!/bin/bash
# Docker usage examples for ML service

echo "=========================================="
echo "ML Service Docker Commands"
echo "=========================================="
echo ""

echo "BUILD IMAGE"
echo "-----------"
echo "docker build -t ml-service:latest ."
echo ""

echo "RUN CONTAINER (foreground)"
echo "--------------------------"
echo "docker run -p 8000:8000 ml-service:latest"
echo ""

echo "RUN CONTAINER (background)"
echo "--------------------------"
echo "docker run -d -p 8000:8000 --name ml-service ml-service:latest"
echo ""

echo "TEST API"
echo "--------"
echo "# Health check"
echo "curl http://localhost:8000/health"
echo ""
echo "# Make prediction"
echo "curl -X POST http://localhost:8000/predict \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"data\": [{...}]}'"
echo ""

echo "VIEW LOGS"
echo "---------"
echo "docker logs ml-service"
echo "docker logs -f ml-service  # Follow logs"
echo ""

echo "STOP CONTAINER"
echo "--------------"
echo "docker stop ml-service"
echo "docker rm ml-service"
echo ""

echo "LIST CONTAINERS"
echo "---------------"
echo "docker ps                  # Running"
echo "docker ps -a               # All"
echo ""

echo "VIEW IMAGE INFO"
echo "---------------"
echo "docker images ml-service"
echo "docker inspect ml-service:latest"
echo ""
