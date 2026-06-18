#!/bin/bash
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG_PATH="logs/train_${TIMESTAMP}.log"
ln -sfn "train_${TIMESTAMP}.log" logs/train.latest.log
exec ./scripts/train.sh --foreground > "$LOG_PATH" 2>&1
