#!/usr/bin/env bash
set -euo pipefail

# Start the FastAI training script in the background
python -m src.ml.fastai_training &
TRAIN_PID=$!

cleanup() {
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Stopping training process $TRAIN_PID"
    kill "$TRAIN_PID" 2>/dev/null || true
    wait "$TRAIN_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

# Run the web application via gunicorn
# Bind to port 8001 to match project defaults
exec gunicorn -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001 src.backend.main:app
