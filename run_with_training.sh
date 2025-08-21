#!/usr/bin/env bash
set -euo pipefail

# Create session directory if it doesn't exist
mkdir -p session

# Start the FastAI training script in the background with logging
python -u -m src.ml.fastai_training > session/fastai_training.logs 2>&1 &
TRAIN_PID=$!

cleanup() {
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Stopping training process $TRAIN_PID"
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

# Run the web application via gunicorn with logging
# Bind to port 8001 to match project defaults
exec gunicorn --reload -w 1 -b 0.0.0.0:8001 src.backend.main:app > session/main.logs 2>&1
