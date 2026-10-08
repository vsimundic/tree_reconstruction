#!/bin/bash
#
# Run the Docker container for the PEAR tree-pruning pipeline
# (tree reconstruction + pruning point detection).
# Mounts the project directory into /workspace inside the container.
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

IMAGE_NAME="pear/pear-tree-pruning"
IMAGE_TAG="latest"
CONTAINER_NAME="pear-tree-pruning"

echo "============================================="
echo "  Running container: ${CONTAINER_NAME}"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Mounting: ${PROJECT_DIR} -> /workspace"
echo "============================================="

# Allow the local user to connect to the X server
xhost +local:docker 2>/dev/null

# Remove existing container with the same name if it exists
docker rm -f "${CONTAINER_NAME}" 2>/dev/null

docker run -it \
    --name "${CONTAINER_NAME}" \
    --add-host=host.docker.internal:host-gateway \
    --volume "${PROJECT_DIR}:/workspace" \
    --workdir /workspace \
    --env DISPLAY="${DISPLAY}" \
    --env QT_X11_NO_MITSHM=1 \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    /bin/bash
