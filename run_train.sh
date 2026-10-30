#!/bin/bash
# Launch training in a tmux session inside the project container.
# Usage: [CPUS=0-23] ./run_train.sh <run_name> [train.py args...]
# CPUS optionally pins the run with taskset (rootless docker ignores --cpuset-cpus).
set -euo pipefail
RUN=${1:-lafan1_main}
if [ "$RUN" = "--help" ] || [ "$RUN" = "-h" ]; then
  echo "Usage: [CPUS=0-23] $0 [run_name] [train.py args...]"
  echo "Uses logs/<run_name> exactly; stdout is saved to logs/<run_name>.stdout.log."
  exit 0
fi
if [ "$#" -gt 0 ]; then shift; fi
if [[ ! "$RUN" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "run_name must contain only letters, digits, underscores and hyphens." >&2
  exit 2
fi
CONTAINER=${SUPERMIMIC_CONTAINER:-supermimic_isaaclab}
TRAIN=(python supermimic/train.py --log_dir "logs/$RUN" --no-timestamp "$@")
if [ -n "${CPUS:-}" ]; then TRAIN=(taskset -c "$CPUS" "${TRAIN[@]}"); fi
# tmux invokes a shell: quote each argument rather than joining user input with $*.
printf -v TRAIN_COMMAND '%q ' "${TRAIN[@]}"
printf -v LOG_PATH '%q' "logs/$RUN.stdout.log"
docker exec "$CONTAINER" tmux new-session -d -s "$RUN" \
  -c /workspace/supermimic_isaaclab bash -o pipefail -c \
  "mkdir -p logs; ${TRAIN_COMMAND} 2>&1 | tee -a ${LOG_PATH}"
echo "Started tmux session $RUN in $CONTAINER."
