#!/bin/bash
# Start the project container with one GPU and this repository mounted.
# Usage: docker/run_container.sh [GPU_INDEX] [MEMORY_LIMIT]
set -euo pipefail
GPU=${1:-0}
MEM=${2:-128g}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CONTAINER=${SUPERMIMIC_CONTAINER:-supermimic_isaaclab}
IMAGE=${SUPERMIMIC_IMAGE:-supermimic_isaaclab:latest}
WORKSPACE=/workspace/supermimic_isaaclab
if [ "$GPU" = "--help" ] || [ "$GPU" = "-h" ]; then
  echo "Usage: $0 [GPU_INDEX] [MEMORY_LIMIT]"
  echo "Defaults: GPU 0, memory 128g, container $CONTAINER, image $IMAGE."
  exit 0
fi
if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [GPU_INDEX] [MEMORY_LIMIT]" >&2
  exit 2
fi
if docker container inspect "$CONTAINER" >/dev/null 2>&1; then
  echo "Container $CONTAINER already exists. Inspect its repository mount before reusing it." >&2
  exit 1
fi
docker run -d --init --name "$CONTAINER" --gpus "\"device=${GPU}\"" \
  --memory="$MEM" --shm-size=16g --user root \
  -e WARP_CACHE_PATH="$WORKSPACE/.cache/warp" -e MPLCONFIGDIR=/tmp/mpl \
  -e TORCHINDUCTOR_CACHE_DIR="$WORKSPACE/.cache/inductor" \
  -v "$REPO:$WORKSPACE" -w "$WORKSPACE" "$IMAGE" sleep infinity
