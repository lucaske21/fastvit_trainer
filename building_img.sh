#!/bin/bash

# Build Docker image script
# Usage: ./building_img.sh [IMAGE_NAME] [IMAGE_TAG] [DOCKERFILE]
# Sample: ./building_img.sh fastvit_trainer latest Dockerfile
# Defaults: IMAGE_NAME=fastvit_trainer, IMAGE_TAG=latest, DOCKERFILE
set -e

IMAGE_NAME="${1:-fastvit_trainer}"
IMAGE_TAG="${2:-latest}"
DOCKERFILE="${3:-Dockerfile}"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Using Dockerfile: ${DOCKERFILE}"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" .

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo "✗ Docker image build failed"
    exit 1
fi
