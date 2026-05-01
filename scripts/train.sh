#!/bin/bash

# Propagator Training Script (Single Entry Point)
# Usage: ./scripts/train.sh [--foreground]

FOREGROUND=false
if [[ "$1" == "--foreground" ]]; then
    FOREGROUND=true
fi

# Project Root Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

mkdir -p logs

# NVIDIA/JAX Library Path Setup
NVIDIA_LIBS=""
for dir in $PROJECT_ROOT/.venv/lib/python3.11/site-packages/nvidia/*/lib; do
    if [ -d "$dir" ]; then
        NVIDIA_LIBS="$NVIDIA_LIBS:$dir"
    fi
done

export LD_LIBRARY_PATH="$NVIDIA_LIBS:$LD_LIBRARY_PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

TRAIN_ARGS=(
    --dataset-name xinrongzhang2022/Duplex-UltraChat
    --dataset-mode duplex_chat
    --max-train-rows 100000
    --max-val-rows 10000
    --streaming
    --eval-every 100
    --checkpoint-every 1000
    --batch-size 8
    --epochs 4
    --learning-rate 3e-4
    --precision float16
)

PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"

if [ "$FOREGROUND" = true ]; then
    echo "Starting training in foreground..."
    "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}"
else
    echo "Starting training in background..."
    # Use absolute path for train.py and ensure logs are written correctly
    nohup "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}" > "$PROJECT_ROOT/logs/train.log" 2>&1 &
    PID=$!
    echo "PID: $PID"
    echo "Follow logs with: tail -f logs/train.log"
fi
