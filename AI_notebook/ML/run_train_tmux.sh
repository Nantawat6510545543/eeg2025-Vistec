#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/mount/NAS-workspace-portal/eeg2025-Vistec"
CONFIG_PATH="${1:-$REPO_ROOT/AI_notebook/ML/train_config.sample.json}"
SESSION_NAME="${2:-eeg_train_cfg}"
PYTHON_EXEC="${3:-python}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config file not found: $CONFIG_PATH"
  exit 1
fi

RUNS_DIR="$REPO_ROOT/jobs/notebook_binary_run"
mkdir -p "$RUNS_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="$RUNS_DIR/${SESSION_NAME}_${TS}.log"

CMD="cd '$REPO_ROOT' && PYTHONUNBUFFERED=1 '$PYTHON_EXEC' AI_notebook/ML/train_from_config.py --config '$CONFIG_PATH' 2>&1 | stdbuf -oL -eL tee '$LOG_PATH'"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Use another name or close it with: tmux kill-session -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "$CMD"

echo "Started tmux session: $SESSION_NAME"
echo "Log: $LOG_PATH"
echo "Attach: tmux attach -t $SESSION_NAME"
echo "Check: tmux ls"
