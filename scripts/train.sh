#!/usr/bin/env bash
set -euo pipefail

# Propagator Training Script
# Usage: ./scripts/train.sh [--foreground]

FOREGROUND=false
if [[ "${1:-}" == "--foreground" ]]; then
    FOREGROUND=true
fi

# Project Root Setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

mkdir -p logs

CPU_COUNT="$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export SLACK_LOG_ENABLED="${SLACK_LOG_ENABLED:-0}"
export SLACK_LOG_PREFIX="${SLACK_LOG_PREFIX:-propagator-train}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$CPU_COUNT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_COUNT}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_COUNT}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_COUNT}"
export TPU_RUNTIME_METRICS_PORTS="${TPU_RUNTIME_METRICS_PORTS:-8431,8432,8433,8434}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

if [[ -d /gcs ]]; then
    DEFAULT_GCS_BACKUP="/gcs/propagator-backups/propagator-duplex"
else
    DEFAULT_GCS_BACKUP=""
fi
GCS_BACKUP_DIR="${GCS_BACKUP_DIR:-$DEFAULT_GCS_BACKUP}"
TEXT_PREPROCESSING_WORKERS="${TEXT_PREPROCESSING_WORKERS:-$(( CPU_COUNT < 64 ? CPU_COUNT : 64 ))}"
AUDIO_PREPROCESSING_WORKERS="${AUDIO_PREPROCESSING_WORKERS:-$(( CPU_COUNT < 64 ? CPU_COUNT : 64 ))}"
TEXT_PREPROCESSING_CHUNK_SIZE="${TEXT_PREPROCESSING_CHUNK_SIZE:-16}"
AUDIO_PREPROCESSING_CHUNK_SIZE="${AUDIO_PREPROCESSING_CHUNK_SIZE:-1}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_AUDIO_EVERY="${EVAL_AUDIO_EVERY:-5000}"
EVAL_AUDIO_SAMPLES="${EVAL_AUDIO_SAMPLES:-2}"
EVAL_AUDIO_SECONDS="${EVAL_AUDIO_SECONDS:-10.0}"
EVAL_AUDIO_INPUT_AUDIO_SECONDS="${EVAL_AUDIO_INPUT_AUDIO_SECONDS:-5.0}"
AUDIO_MIN_GENERATION_SECONDS="${AUDIO_MIN_GENERATION_SECONDS:-1.0}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-16}"
LOCAL_CHECKPOINT_KEEP="${LOCAL_CHECKPOINT_KEEP:-1}"
AUDIO_TASK_MIX="${AUDIO_TASK_MIX:-{\"asr\":0.25,\"tts\":0.35,\"audio\":0.20,\"hybrid\":0.20}}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-4.0}"
MAX_AUDIO_TOKENS_PER_ROW="${MAX_AUDIO_TOKENS_PER_ROW:-2400}"
REMAT_SCAN_STEP="${REMAT_SCAN_STEP:-true}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20000}"
GCS_SYNC_EVERY="${GCS_SYNC_EVERY:-20000}"
GCS_BACKUP_KEEP="${GCS_BACKUP_KEEP:-5}"
DATASET_MIX_FILE="${DATASET_MIX_FILE:-data/propagator_dataset_mix_v3.json}"
PROPAGATOR_DISK_ROOT="${PROPAGATOR_DISK_ROOT:-/mnt/disks/propagator-cache}"
if [[ -z "${CACHE_ROOT:-}" && -d "$PROPAGATOR_DISK_ROOT" && -w "$PROPAGATOR_DISK_ROOT" ]]; then
    CACHE_ROOT="$PROPAGATOR_DISK_ROOT/cache"
elif [[ -z "${CACHE_ROOT:-}" && -d /dev/shm && -w /dev/shm ]]; then
    CACHE_ROOT="/dev/shm/propagator-cache"
else
    CACHE_ROOT="${CACHE_ROOT:-outputs/cache}"
fi
if [[ -d "$PROPAGATOR_DISK_ROOT" && -w "$PROPAGATOR_DISK_ROOT" ]]; then
    export CACHE_STORAGE="${CACHE_STORAGE:-disk}"
    export CACHE_READ_MODE="${CACHE_READ_MODE:-mmap}"
    export HF_HOME="${HF_HOME:-$PROPAGATOR_DISK_ROOT/hf}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
    export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
    export TMPDIR="${TMPDIR:-$PROPAGATOR_DISK_ROOT/tmp}"
    mkdir -p "$CACHE_ROOT" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TMPDIR"
fi
DEFAULT_OUTPUT_ROOT="outputs/propagator-multimodal-v2"
if [[ -d "$PROPAGATOR_DISK_ROOT" && -w "$PROPAGATOR_DISK_ROOT" ]]; then
    DEFAULT_OUTPUT_ROOT="$PROPAGATOR_DISK_ROOT/outputs/propagator-multimodal-v2"
    mkdir -p "$(dirname "$DEFAULT_OUTPUT_ROOT")"
fi
ECHOX_CACHE_RAW_SHARDS="${ECHOX_CACHE_RAW_SHARDS:-auto}"
ECHOX_RAW_CACHE_MIN_FREE_GB="${ECHOX_RAW_CACHE_MIN_FREE_GB:-96}"

if [[ -z "${DATASET_MIX:-}" && -f "$DATASET_MIX_FILE" ]]; then
    DATASET_MIX="$(python3 - "$DATASET_MIX_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
print(json.dumps(json.loads(path.read_text(encoding="utf-8")), separators=(",", ":")))
PY
)"
fi

TRAIN_ARGS=(
    --hidden-size "${HIDDEN_SIZE:-1536}"
    --num-layers "${NUM_LAYERS:-24}"
    --memory-key-size "${MEMORY_KEY_SIZE:-384}"
    --memory-value-size "${MEMORY_VALUE_SIZE:-768}"
    --train-unroll-len "${TRAIN_UNROLL_LEN:-32}"
    --batch-size "${BATCH_SIZE:-0}"
    --output-root "${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
    --tokenizer-vocab-size 16000
    --tokenizer-train-rows "${TOKENIZER_TRAIN_ROWS:-200000}"
    --max-train-chunks "${MAX_TRAIN_CHUNKS:-0}"
    --max-val-chunks "${MAX_VAL_CHUNKS:-50000}"
    --streaming
    --text-preprocessing-workers "$TEXT_PREPROCESSING_WORKERS"
    --audio-preprocessing-workers "$AUDIO_PREPROCESSING_WORKERS"
    --text-preprocessing-chunk-size "$TEXT_PREPROCESSING_CHUNK_SIZE"
    --audio-preprocessing-chunk-size "$AUDIO_PREPROCESSING_CHUNK_SIZE"
    --cache-storage "${CACHE_STORAGE:-auto}"
    --cache-root "$CACHE_ROOT"
    --cache-read-mode "${CACHE_READ_MODE:-auto}"
    --cache-read-memory-fraction "${CACHE_READ_MEMORY_FRACTION:-0.50}"
    --same-split-validation-stride "${SAME_SPLIT_VALIDATION_STRIDE:-10}"
    --same-split-validation-offset "${SAME_SPLIT_VALIDATION_OFFSET:-0}"
    --validation-control-batches "${VALIDATION_CONTROL_BATCHES:-4}"
    --synthetic-control-train-examples "${SYNTHETIC_CONTROL_TRAIN_EXAMPLES:-2048}"
    --synthetic-control-val-examples "${SYNTHETIC_CONTROL_VAL_EXAMPLES:-512}"
    --synthetic-control-train-rate "${SYNTHETIC_CONTROL_TRAIN_RATE:-0.10}"
    --synthetic-interrupt-fraction "${SYNTHETIC_INTERRUPT_FRACTION:-0.60}"
    --eval-every "$EVAL_EVERY"
    --eval-audio-every "$EVAL_AUDIO_EVERY"
    --eval-audio-samples "$EVAL_AUDIO_SAMPLES"
    --eval-audio-seconds "$EVAL_AUDIO_SECONDS"
    --eval-audio-input-samples "${EVAL_AUDIO_INPUT_SAMPLES:-2}"
    --eval-audio-input-audio-seconds "$EVAL_AUDIO_INPUT_AUDIO_SECONDS"
    --validation-batches "$VALIDATION_BATCHES"
    --audio-task-mix "$AUDIO_TASK_MIX"
    --max-audio-seconds "$MAX_AUDIO_SECONDS"
    --max-audio-tokens-per-row "$MAX_AUDIO_TOKENS_PER_ROW"
    --audio-token-loss-weight "${AUDIO_TOKEN_LOSS_WEIGHT:-2.0}"
    --audio-codebook-loss-weight "${AUDIO_CODEBOOK_LOSS_WEIGHT:-3.0}"
    --audio-out-loss-weight "${AUDIO_OUT_LOSS_WEIGHT:-8.0}"
    --audio-end-loss-weight "${AUDIO_END_LOSS_WEIGHT:-8.0}"
    --audio-min-generation-seconds "$AUDIO_MIN_GENERATION_SECONDS"
    --audio-eval-normalize-rms "${AUDIO_EVAL_NORMALIZE_RMS:-0.06}"
    --audio-low-rms-threshold "${AUDIO_LOW_RMS_THRESHOLD:-0.005}"
    --train-log-every "${TRAIN_LOG_EVERY:-1000}"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --local-checkpoint-keep "$LOCAL_CHECKPOINT_KEEP"
    --gcs-sync-every "$GCS_SYNC_EVERY"
    --gcs-backup-keep "$GCS_BACKUP_KEEP"
    --auto-batch-max-per-device "${AUTO_BATCH_MAX_PER_DEVICE:-16}"
    --auto-batch-multiple-per-device "${AUTO_BATCH_MULTIPLE_PER_DEVICE:-8}"
    --auto-batch-memory-util "${AUTO_BATCH_MEMORY_UTIL:-0.78}"
    --epochs "${EPOCHS:-30}"
    --learning-rate 3e-4
    --interrupt-input-loss-weight 1.0
    --precision bfloat16
    --optimizer adamw
    --grad-clip-norm 1.0
    --edge-vram-mb 2048
    --edge-vram-util-target 0.70
    --quantization-bits 4
)

if [[ -n "${DATASET_MIX:-}" ]]; then
    TRAIN_ARGS+=(--dataset-mix "$DATASET_MIX")
fi

case "${ECHOX_CACHE_RAW_SHARDS,,}" in
    auto)
        if [[ "$CACHE_ROOT" == /dev/shm/* || "$CACHE_ROOT" == "$PROPAGATOR_DISK_ROOT"/* ]]; then
            TRAIN_ARGS+=(
                --echox-cache-raw-shards
                --echox-raw-cache-dir "$CACHE_ROOT/echox_raw_shards"
                --echox-raw-cache-min-free-gb "$ECHOX_RAW_CACHE_MIN_FREE_GB"
            )
        fi
        ;;
    1|true|yes|on)
        TRAIN_ARGS+=(
            --echox-cache-raw-shards
            --echox-raw-cache-dir "${ECHOX_RAW_CACHE_DIR:-$CACHE_ROOT/echox_raw_shards}"
            --echox-raw-cache-min-free-gb "$ECHOX_RAW_CACHE_MIN_FREE_GB"
        )
        ;;
    0|false|no|off)
        TRAIN_ARGS+=(--no-echox-cache-raw-shards)
        ;;
    *)
        echo "ECHOX_CACHE_RAW_SHARDS must be auto, true, or false; got: $ECHOX_CACHE_RAW_SHARDS" >&2
        exit 2
        ;;
esac

case "${REMAT_SCAN_STEP,,}" in
    1|true|yes|on)
        TRAIN_ARGS+=(--remat-scan-step)
        ;;
    0|false|no|off)
        TRAIN_ARGS+=(--no-remat-scan-step)
        ;;
    *)
        echo "REMAT_SCAN_STEP must be true or false, got: $REMAT_SCAN_STEP" >&2
        exit 2
        ;;
esac

if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
    TRAIN_ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS")
fi

if [[ -n "${EVAL_AUDIO_TOKENS:-}" ]]; then
    TRAIN_ARGS+=(--eval-audio-tokens "$EVAL_AUDIO_TOKENS")
fi

if [[ -n "${EVAL_AUDIO_INPUT_AUDIO_TOKENS:-}" ]]; then
    TRAIN_ARGS+=(--eval-audio-input-audio-tokens "$EVAL_AUDIO_INPUT_AUDIO_TOKENS")
fi

if [[ -n "${AUDIO_MIN_GENERATION_TOKENS:-}" ]]; then
    TRAIN_ARGS+=(--audio-min-generation-tokens "$AUDIO_MIN_GENERATION_TOKENS")
fi

if [[ -n "$GCS_BACKUP_DIR" ]]; then
    TRAIN_ARGS+=(--gcs-backup-dir "$GCS_BACKUP_DIR")
fi

if [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

if [ "$FOREGROUND" = true ]; then
    echo "Starting training in foreground..."
    "$PYTHON_BIN" "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}"
else
    echo "Starting training in background..."
    LOG_PATH="$PROJECT_ROOT/logs/train_$(date -u +%Y%m%dT%H%M%SZ).log"
    ln -sfn "$(basename "$LOG_PATH")" "$PROJECT_ROOT/logs/train.latest.log"
    if command -v setsid >/dev/null 2>&1; then
        setsid "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}" > "$LOG_PATH" 2>&1 < /dev/null &
    else
        nohup "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}" > "$LOG_PATH" 2>&1 < /dev/null &
    fi
    PID=$!
    echo "PID: $PID"
    echo "$PID" > "$PROJECT_ROOT/logs/train.pid"
    echo "Follow logs with: tail -f logs/train.latest.log"
fi
