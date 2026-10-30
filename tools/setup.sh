#!/bin/bash
# Fetch the pinned Isaac Lab checkout, then build the kit-less runtime images.
set -euo pipefail
cd "$(dirname "$0")/.."
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
case "${1:-}" in
  --help|-h)
    echo "Usage: $0 [--checkout-only]"
    echo "Default: fetch pinned Isaac Lab and build Docker images."
    exit 0 ;;
  --checkout-only|"") ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [--checkout-only]" >&2
  exit 2
fi
if [ ! -e IsaacLab ]; then
  git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git IsaacLab
fi
if [ ! -d IsaacLab/.git ]; then
  echo "IsaacLab must be a Git checkout; move the existing directory before setup." >&2
  exit 1
fi
if [ -n "$(git -C IsaacLab status --porcelain)" ]; then
  echo "IsaacLab has local changes; preserve or commit them before setup." >&2
  exit 1
fi
if ! git -C IsaacLab cat-file -e "${ISAACLAB_COMMIT}^{commit}" 2>/dev/null; then
  git -C IsaacLab fetch origin "$ISAACLAB_COMMIT"
fi
git -C IsaacLab checkout --detach "$ISAACLAB_COMMIT"
if [ "${1:-}" = "--checkout-only" ]; then
  exit 0
fi
(cd IsaacLab && docker build -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker build -t "${SUPERMIMIC_IMAGE:-supermimic_isaaclab:latest}" docker/
