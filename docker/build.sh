#!/bin/bash
#
# Build the Docker image for the PEAR tree-pruning pipeline
# (tree reconstruction + pruning point detection).
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="pear/pear-tree-pruning"
IMAGE_TAG="latest"

echo "============================================="
echo "  Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "============================================="

docker build \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${PROJECT_DIR}"

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================="
    echo "  Build successful: ${IMAGE_NAME}:${IMAGE_TAG}"
    echo "============================================="
else
    echo ""
    echo "============================================="
    echo "  Build FAILED!"
    echo "============================================="
    exit 1
fi
