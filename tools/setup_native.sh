#!/bin/bash
# Optional native environment: use the same upstream lock and extras as Docker.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  echo "Usage: $0"
  echo "Requires uv 0.12.9 and Linux system libraries described in README.md."
  echo "Installs a managed Python 3.12 environment into $REPO/.venv."
  exit 0
fi
if [ "$#" -ne 0 ]; then
  echo "Usage: $0" >&2
  exit 2
fi
if ! command -v uv >/dev/null || [[ "$(uv --version)" != "uv 0.12.9"* ]]; then
  echo "Install uv 0.12.9 first: pipx install uv==0.12.9 (see docs/ENVIRONMENT.md)." >&2
  exit 1
fi
"$REPO/tools/setup.sh" --checkout-only
# Avoid modifying a currently activated conda or virtual environment. The pinned
# upstream lock also carries the CUDA wheel index and importer dependency overrides.
UV_PROJECT_ENVIRONMENT="$REPO/.venv" UV_PYTHON_PREFERENCE=only-managed \
  uv sync --project "$REPO/IsaacLab" --python 3.12 --frozen --inexact \
  --extra all --extra importers --extra test
echo "Ready. Activate with: source '$REPO/.venv/bin/activate'"
