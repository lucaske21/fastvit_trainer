#!/bin/bash

# Build Docker image script
# Usage: ./building_img.sh [IMAGE_NAME] [IMAGE_TAG] [DOCKERFILE]
# Sample: ./building_img.sh fastvit_trainer main-a1b2c3d Dockerfile
# Defaults: IMAGE_NAME=fastvit_trainer, IMAGE_TAG={branch}-{commit-hash}, DOCKERFILE
set -e

IMAGE_NAME="${1:-fastvit_trainer}"
if [ -n "${2:-}" ]; then
    IMAGE_TAG="${2}"
else
    if ! command -v git >/dev/null 2>&1; then
        echo "[ERROR] git is required to auto-generate IMAGE_TAG"
        exit 1
    fi

    BRANCH_RAW="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
    COMMIT_HASH="$(git rev-parse --short HEAD 2>/dev/null || true)"

    if [ -z "${BRANCH_RAW}" ] || [ -z "${COMMIT_HASH}" ] || [ "${BRANCH_RAW}" = "HEAD" ]; then
        echo "[ERROR] could not determine git branch/commit for IMAGE_TAG"
        exit 1
    fi

    # Docker tags cannot include '/' or spaces; normalize any unsafe chars.
    BRANCH_SAFE="$(printf '%s' "${BRANCH_RAW}" | tr '[:space:]/' '-' | sed 's/[^A-Za-z0-9_.-]/-/g')"
    IMAGE_TAG="${BRANCH_SAFE}-${COMMIT_HASH}"
fi

DOCKERFILE="${3:-Dockerfile}"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Using Dockerfile: ${DOCKERFILE}"

docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f "${DOCKERFILE}" .

if [ $? -eq 0 ]; then
    echo "[OK] Docker image built successfully: ${IMAGE_NAME}:${IMAGE_TAG}"
else
    echo "[ERROR] Docker image build failed"
    exit 1
fi
