#!/bin/bash
# Publish Docker image to DockerHub with versioned tags

set -e

# Configuration
IMAGE_NAME="ml-service"
VERSION="1.0.0"
DOCKERHUB_USERNAME="${DOCKERHUB_USERNAME:-your-username}"

echo "=========================================="
echo "Docker Image Publishing"
echo "=========================================="
echo ""
echo "Image: $IMAGE_NAME"
echo "Version: $VERSION"
echo "DockerHub: $DOCKERHUB_USERNAME"
echo ""

# Check if logged in to DockerHub
if ! docker info | grep -q "Username"; then
    echo "⚠️  Not logged in to DockerHub"
    echo "Please run: docker login"
    exit 1
fi

# Tag the image
echo "Step 1: Creating tags..."
docker tag $IMAGE_NAME:latest $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
docker tag $IMAGE_NAME:latest $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION
docker tag $IMAGE_NAME:latest $DOCKERHUB_USERNAME/$IMAGE_NAME:1.0
docker tag $IMAGE_NAME:latest $DOCKERHUB_USERNAME/$IMAGE_NAME:1

echo "✓ Tags created:"
echo "  - $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
echo "  - $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION"
echo "  - $DOCKERHUB_USERNAME/$IMAGE_NAME:1.0"
echo "  - $DOCKERHUB_USERNAME/$IMAGE_NAME:1"
echo ""

# Push to DockerHub
echo "Step 2: Pushing to DockerHub..."
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:latest
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:1.0
docker push $DOCKERHUB_USERNAME/$IMAGE_NAME:1

echo ""
echo "✅ Successfully published to DockerHub!"
echo ""
echo "Pull with:"
echo "  docker pull $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
echo "  docker pull $DOCKERHUB_USERNAME/$IMAGE_NAME:$VERSION"
echo ""
echo "Run with:"
echo "  docker run -p 8000:8000 $DOCKERHUB_USERNAME/$IMAGE_NAME:latest"
