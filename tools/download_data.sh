#!/bin/bash
# Download verified standard G1/LAFAN1 assets, then convert them in the container.
set -euo pipefail
cd "$(dirname "$0")/.."
CONTAINER=${SUPERMIMIC_CONTAINER:-supermimic_isaaclab}
WORKSPACE=/workspace/supermimic_isaaclab
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo "Usage: $0 (requires the running $CONTAINER container)"
  echo "Without Docker: python3 tools/download_data_native.py, then run both converters."
  exit 0
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: $0" >&2
  exit 2
fi
docker exec -w "$WORKSPACE" "$CONTAINER" python -c 'import mujoco; from pxr import Usd'
python3 tools/download_data_native.py
docker exec -w "$WORKSPACE" "$CONTAINER" python tools/urdf_to_mjcf.py
docker exec -w "$WORKSPACE" "$CONTAINER" python tools/mjcf_to_usd.py
