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

declare -A USER_ENV_OVERRIDES=()
for key in \
    MODEL_PRESET OUTPUT_ROOT CACHE_ROOT DATASET_MIX_FILE DATASET_MIX FORCE_DATASET_MIX_FILE \
    MAX_STEPS EPOCHS GCS_BACKUP_DIR TRAIN_UNROLL_LEN \
    MAX_TRAIN_CHUNKS MAX_VAL_CHUNKS MAX_TRAIN_ROWS MAX_VAL_ROWS DATA_PACK_COUNT DATA_PACK_INDEX \
    MAX_AUDIO_SECONDS MAX_AUDIO_TOKENS_PER_ROW \
    CHECKPOINT_RESUME FORCE_TRAIN_TOKENIZER TOKENIZER_TRAIN_ROWS \
    EVAL_EVERY EVAL_DATASET_CASES CHECKPOINT_EVERY GCS_SYNC_EVERY LOCAL_CHECKPOINT_KEEP \
    TEXT_PREPROCESSING_WORKERS AUDIO_PREPROCESSING_WORKERS \
    TEXT_PREPROCESSING_CHUNK_SIZE AUDIO_PREPROCESSING_CHUNK_SIZE \
    TEXT_PREPROCESSING_BATCH_ROWS AUDIO_PREPROCESSING_BATCH_ROWS \
    TOKENIZE_START_METHOD TOKENIZE_IMAP_CHUNK_SIZE TOKENIZE_MAXTASKS_PER_CHILD \
    DATASET_STREAMING_RETRIES DATASET_STREAMING_RETRY_INITIAL_DELAY DATASET_STREAMING_RETRY_MAX_DELAY \
    AUTO_BATCH_MAX_PER_DEVICE AUTO_BATCH_MULTIPLE_PER_DEVICE AUTO_BATCH_MEMORY_UTIL \
    TPU_PREFLIGHT TPU_EXPECTED_DEVICE_COUNT TPU_ENABLE_VFIO_BIND TPU_VFIO_DEVICE_IDS TPU_PREFLIGHT_TIMEOUT \
    ECHOX_RAW_CACHE_MIN_FREE_GB ASR_EVAL_CASE_FOLD NO_ASR_EVAL_CASE_FOLD \
    EARLY_STOPPING_PATIENCE EARLY_STOPPING_MIN_DELTA \
    IMAGE_INPUT_RESOLUTION IMAGE_MAX_INPUT_RESOLUTION IMAGE_PATCH_SIZE IMAGE_PATCH_VOCAB_SIZE IMAGE_TOKENS_PER_SAMPLE
do
    if [[ ${!key+x} ]]; then
        USER_ENV_OVERRIDES["$key"]="${!key}"
    fi
done

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi
for key in "${!USER_ENV_OVERRIDES[@]}"; do
    printf -v "$key" '%s' "${USER_ENV_OVERRIDES[$key]}"
    export "$key"
done

mkdir -p logs

enable_transparent_hugepages() {
    local thp_path="/sys/kernel/mm/transparent_hugepage/enabled"
    if [[ ! -e "$thp_path" ]]; then
        return 0
    fi
    if grep -q '^\[always\]' "$thp_path" 2>/dev/null; then
        return 0
    fi
    if [[ -w "$thp_path" ]]; then
        printf always >"$thp_path" || true
    elif command -v sudo >/dev/null 2>&1; then
        sudo -n sh -c "echo always > '$thp_path'" 2>/dev/null || true
    fi
    if ! grep -q '^\[always\]' "$thp_path" 2>/dev/null; then
        echo "Transparent hugepages are not set to always; enable with: sudo sh -c 'echo always > $thp_path'" >&2
        exit 2
    fi
}

enable_transparent_hugepages

CPU_COUNT="$(python3 - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HUB_TOKEN="${HF_HUB_TOKEN:-${HF_TOKEN}}"
export HF_DATASETS_DISABLE_PROGRESS_BARS="${HF_DATASETS_DISABLE_PROGRESS_BARS:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"
export SLACK_LOG_ENABLED="${SLACK_LOG_ENABLED:-0}"
export SLACK_LOG_PREFIX="${SLACK_LOG_PREFIX:-propagator-train}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-$CPU_COUNT}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CPU_COUNT}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CPU_COUNT}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CPU_COUNT}"
export TPU_RUNTIME_METRICS_PORTS="${TPU_RUNTIME_METRICS_PORTS:-8431,8432,8433,8434}"
if [[ -z "${LIBTPU_INIT_ARGS:-}" ]]; then
    export LIBTPU_INIT_ARGS="--enable_tpunetd_client=false"
elif [[ "$LIBTPU_INIT_ARGS" != *"--enable_tpunetd_client"* ]]; then
    export LIBTPU_INIT_ARGS="$LIBTPU_INIT_ARGS --enable_tpunetd_client=false"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
TPU_PREFLIGHT="${TPU_PREFLIGHT:-1}"
TPU_EXPECTED_DEVICE_COUNT="${TPU_EXPECTED_DEVICE_COUNT:-8}"
TPU_ENABLE_VFIO_BIND="${TPU_ENABLE_VFIO_BIND:-1}"
TPU_VFIO_DEVICE_IDS="${TPU_VFIO_DEVICE_IDS:-1ae0:006f}"
TPU_PREFLIGHT_TIMEOUT="${TPU_PREFLIGHT_TIMEOUT:-90}"

if [[ -d /gcs ]]; then
    DEFAULT_GCS_BACKUP="/gcs/propagator-backups/propagator-duplex"
else
    DEFAULT_GCS_BACKUP=""
fi
GCS_BACKUP_DIR="${GCS_BACKUP_DIR:-$DEFAULT_GCS_BACKUP}"
TEXT_PREPROCESSING_WORKERS="${TEXT_PREPROCESSING_WORKERS:-$(( CPU_COUNT < 160 ? CPU_COUNT : 160 ))}"
AUDIO_BACKEND="${AUDIO_BACKEND:-mimi}"
AUDIO_PREPROCESSING_WORKERS="${AUDIO_PREPROCESSING_WORKERS:-$(( CPU_COUNT < 64 ? CPU_COUNT : 64 ))}"
TEXT_PREPROCESSING_CHUNK_SIZE="${TEXT_PREPROCESSING_CHUNK_SIZE:-64}"
AUDIO_PREPROCESSING_CHUNK_SIZE="${AUDIO_PREPROCESSING_CHUNK_SIZE:-8}"
TEXT_PREPROCESSING_BATCH_ROWS="${TEXT_PREPROCESSING_BATCH_ROWS:-128}"
AUDIO_PREPROCESSING_BATCH_ROWS="${AUDIO_PREPROCESSING_BATCH_ROWS:-128}"
TOKENIZE_START_METHOD="${TOKENIZE_START_METHOD:-fork}"
TOKENIZE_IMAP_CHUNK_SIZE="${TOKENIZE_IMAP_CHUNK_SIZE:-0}"
TOKENIZE_MAXTASKS_PER_CHILD="${TOKENIZE_MAXTASKS_PER_CHILD:-0}"
DATASET_STREAMING_RETRIES="${DATASET_STREAMING_RETRIES:-100}"
DATASET_STREAMING_RETRY_INITIAL_DELAY="${DATASET_STREAMING_RETRY_INITIAL_DELAY:-5.0}"
DATASET_STREAMING_RETRY_MAX_DELAY="${DATASET_STREAMING_RETRY_MAX_DELAY:-120.0}"
EVAL_EVERY="${EVAL_EVERY:-20000}"
EVAL_AUDIO_SECONDS="${EVAL_AUDIO_SECONDS:-10.0}"
EVAL_AUDIO_INPUT_AUDIO_SECONDS="${EVAL_AUDIO_INPUT_AUDIO_SECONDS:-5.0}"
AUDIO_MIN_GENERATION_SECONDS="${AUDIO_MIN_GENERATION_SECONDS:-1.0}"
IMAGE_INPUT_RESOLUTION="${IMAGE_INPUT_RESOLUTION:-160}"
IMAGE_MAX_INPUT_RESOLUTION="${IMAGE_MAX_INPUT_RESOLUTION:-192}"
IMAGE_PATCH_SIZE="${IMAGE_PATCH_SIZE:-16}"
IMAGE_PATCH_VOCAB_SIZE="${IMAGE_PATCH_VOCAB_SIZE:-1024}"
IMAGE_TOKENS_PER_SAMPLE="${IMAGE_TOKENS_PER_SAMPLE:-64}"
ASR_EVAL_CASE_FOLD="${ASR_EVAL_CASE_FOLD:-0}"
NO_ASR_EVAL_CASE_FOLD="${NO_ASR_EVAL_CASE_FOLD:-0}"
VALIDATION_BATCHES="${VALIDATION_BATCHES:-16}"
LOCAL_CHECKPOINT_KEEP="${LOCAL_CHECKPOINT_KEEP:-1}"
CHECKPOINT_RESUME="${CHECKPOINT_RESUME:-1}"
FORCE_TRAIN_TOKENIZER="${FORCE_TRAIN_TOKENIZER:-0}"
TOKENIZER_TRAIN_ROWS="${TOKENIZER_TRAIN_ROWS:-0}"
DATA_PACK_COUNT="${DATA_PACK_COUNT:-0}"
DATA_PACK_INDEX="${DATA_PACK_INDEX:-0}"
MODEL_PRESET="${MODEL_PRESET:-full}"
case "${MODEL_PRESET,,}" in
    full)
        ;;
    sl2610|edge|2gb)
        : "${HIDDEN_SIZE:=768}"
        : "${NUM_LAYERS:=16}"
        : "${MEMORY_KEY_SIZE:=192}"
        : "${MEMORY_VALUE_SIZE:=384}"
        : "${ASSOCIATIVE_GROUPS:=4}"
        : "${MLP_MULTIPLIER:=3}"
        : "${MOE_NUM_EXPERTS:=1}"
        : "${MOE_TOP_K:=1}"
        : "${BATCH_SIZE:=0}"
        : "${AUTO_BATCH_MAX_PER_DEVICE:=8}"
        : "${AUTO_BATCH_MULTIPLE_PER_DEVICE:=4}"
        : "${TOKENIZER_PATH:=assets/tokenizer-byte-bpe-16000.json}"
        : "${TOKENIZER_VOCAB_SIZE:=16000}"
        ;;
    *)
        echo "MODEL_PRESET must be full or sl2610; got: $MODEL_PRESET" >&2
        exit 2
        ;;
esac

AUDIO_TASK_MIX="${AUDIO_TASK_MIX:-{\"asr\":0.25,\"tts\":0.35,\"audio\":0.20,\"hybrid\":0.20}}"
MAX_AUDIO_SECONDS="${MAX_AUDIO_SECONDS:-0}"
case "${AUDIO_BACKEND,,}" in
    mimi)
        AUDIO_CODEBOOK_SIZE="${AUDIO_CODEBOOK_SIZE:-2048}"
        AUDIO_FRAMES_PER_SECOND="${AUDIO_FRAMES_PER_SECOND:-12.5}"
        MAX_AUDIO_TOKENS_PER_ROW="${MAX_AUDIO_TOKENS_PER_ROW:-0}"
        ;;
    encodec)
        AUDIO_CODEBOOK_SIZE="${AUDIO_CODEBOOK_SIZE:-1024}"
        AUDIO_FRAMES_PER_SECOND="${AUDIO_FRAMES_PER_SECOND:-75.0}"
        MAX_AUDIO_TOKENS_PER_ROW="${MAX_AUDIO_TOKENS_PER_ROW:-0}"
        ;;
    *)
        echo "AUDIO_BACKEND must be mimi or encodec; got: $AUDIO_BACKEND" >&2
        exit 2
        ;;
esac
REMAT_SCAN_STEP="${REMAT_SCAN_STEP:-true}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50000}"
GCS_SYNC_EVERY="${GCS_SYNC_EVERY:-50000}"
GCS_BACKUP_KEEP="${GCS_BACKUP_KEEP:-5}"
if [[ "${POST_TRAIN:-0}" =~ ^(1|true|yes|on)$ ]]; then
    DATASET_MIX_FILE="${DATASET_MIX_FILE:-data/mixes/propagator_posttrain_mix.json}"
    LEARNING_RATE="${LEARNING_RATE:-2e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-1000}"
else
    DATASET_MIX_FILE="${DATASET_MIX_FILE:-data/mixes/propagator_dataset_mix.json}"
fi
if [[ ! ${USER_ENV_OVERRIDES[DATASET_MIX]+x} || "${FORCE_DATASET_MIX_FILE:-0}" =~ ^(1|true|yes|on)$ ]]; then
    unset DATASET_MIX
fi
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
DEFAULT_OUTPUT_ROOT="outputs/propagator-multimodal_1b"
if [[ -d "$PROPAGATOR_DISK_ROOT" && -w "$PROPAGATOR_DISK_ROOT" ]]; then
    DEFAULT_OUTPUT_ROOT="$PROPAGATOR_DISK_ROOT/outputs/propagator-multimodal_1b"
    mkdir -p "$(dirname "$DEFAULT_OUTPUT_ROOT")"
fi
EFFECTIVE_OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
if [[ "${EFFECTIVE_OUTPUT_ROOT}" == *"_1b"* && "${MODEL_PRESET,,}" != "full" ]]; then
    echo "Refusing to launch non-full MODEL_PRESET=$MODEL_PRESET into 1B output root: $EFFECTIVE_OUTPUT_ROOT" >&2
    echo "Use MODEL_PRESET=full for 1B training, or choose a non-_1b OUTPUT_ROOT for smaller presets." >&2
    exit 2
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
    --hidden-size "${HIDDEN_SIZE:-1920}"
    --num-layers "${NUM_LAYERS:-24}"
    --memory-key-size "${MEMORY_KEY_SIZE:-416}"
    --memory-value-size "${MEMORY_VALUE_SIZE:-832}"
    --associative-groups "${ASSOCIATIVE_GROUPS:-4}"
    --mlp-multiplier "${MLP_MULTIPLIER:-3}"
    --moe-num-experts "${MOE_NUM_EXPERTS:-1}"
    --moe-top-k "${MOE_TOP_K:-1}"
    --rope-base "${ROPE_BASE:-10000}"
    --rope-position-scale "${ROPE_POSITION_SCALE:-16}"
    --rope-max-position "${ROPE_MAX_POSITION:-1048576}"
    --use-swiglu
    --train-unroll-len "${TRAIN_UNROLL_LEN:-64}"
    --batch-size "${BATCH_SIZE:-0}"
    --output-root "$EFFECTIVE_OUTPUT_ROOT"
    --tokenizer-path "${TOKENIZER_PATH:-assets/tokenizer-byte-bpe-16000.json}"
    --tokenizer-vocab-size "${TOKENIZER_VOCAB_SIZE:-16000}"
    --tokenizer-train-rows "$TOKENIZER_TRAIN_ROWS"
    --max-train-chunks "${MAX_TRAIN_CHUNKS:-0}"
    --max-val-chunks "${MAX_VAL_CHUNKS:-0}"
    --data-pack-count "$DATA_PACK_COUNT"
    --data-pack-index "$DATA_PACK_INDEX"
    --streaming
    --text-preprocessing-workers "$TEXT_PREPROCESSING_WORKERS"
    --audio-preprocessing-workers "$AUDIO_PREPROCESSING_WORKERS"
    --text-preprocessing-chunk-size "$TEXT_PREPROCESSING_CHUNK_SIZE"
    --audio-preprocessing-chunk-size "$AUDIO_PREPROCESSING_CHUNK_SIZE"
    --text-preprocessing-batch-rows "$TEXT_PREPROCESSING_BATCH_ROWS"
    --audio-preprocessing-batch-rows "$AUDIO_PREPROCESSING_BATCH_ROWS"
    --tokenize-start-method "$TOKENIZE_START_METHOD"
    --tokenize-imap-chunk-size "$TOKENIZE_IMAP_CHUNK_SIZE"
    --tokenize-maxtasks-per-child "$TOKENIZE_MAXTASKS_PER_CHILD"
    --dataset-streaming-retries "$DATASET_STREAMING_RETRIES"
    --dataset-streaming-retry-initial-delay "$DATASET_STREAMING_RETRY_INITIAL_DELAY"
    --dataset-streaming-retry-max-delay "$DATASET_STREAMING_RETRY_MAX_DELAY"
    --cache-storage "${CACHE_STORAGE:-auto}"
    --cache-root "$CACHE_ROOT"
    --cache-read-mode "${CACHE_READ_MODE:-auto}"
    --cache-read-memory-fraction "${CACHE_READ_MEMORY_FRACTION:-0.50}"
    --same-split-validation-stride "${SAME_SPLIT_VALIDATION_STRIDE:-10}"
    --same-split-validation-offset "${SAME_SPLIT_VALIDATION_OFFSET:-0}"
    --eval-every "$EVAL_EVERY"
    --eval-audio-seconds "$EVAL_AUDIO_SECONDS"
    --eval-audio-input-samples "${EVAL_AUDIO_INPUT_SAMPLES:-2}"
    --eval-audio-input-audio-seconds "$EVAL_AUDIO_INPUT_AUDIO_SECONDS"
    --validation-batches "$VALIDATION_BATCHES"
    --audio-task-mix "$AUDIO_TASK_MIX"
    --audio-backend "$AUDIO_BACKEND"
    --audio-codebook-size "$AUDIO_CODEBOOK_SIZE"
    --audio-frames-per-second "$AUDIO_FRAMES_PER_SECOND"
    --max-audio-seconds "$MAX_AUDIO_SECONDS"
    --max-audio-tokens-per-row "$MAX_AUDIO_TOKENS_PER_ROW"
    --audio-token-loss-weight "${AUDIO_TOKEN_LOSS_WEIGHT:-1.0}"
    --audio-codebook-loss-weight "${AUDIO_CODEBOOK_LOSS_WEIGHT:-1.0}"
    --audio-out-loss-weight "${AUDIO_OUT_LOSS_WEIGHT:-2.0}"
    --output-modality-loss-weight "${OUTPUT_MODALITY_LOSS_WEIGHT:-2.0}"
    --audio-min-generation-seconds "$AUDIO_MIN_GENERATION_SECONDS"
    --audio-eval-normalize-rms "${AUDIO_EVAL_NORMALIZE_RMS:-0.06}"
    --audio-low-rms-threshold "${AUDIO_LOW_RMS_THRESHOLD:-0.005}"
    --image-input-resolution "$IMAGE_INPUT_RESOLUTION"
    --image-max-input-resolution "$IMAGE_MAX_INPUT_RESOLUTION"
    --image-patch-size "$IMAGE_PATCH_SIZE"
    --image-patch-vocab-size "$IMAGE_PATCH_VOCAB_SIZE"
    --image-tokens-per-sample "$IMAGE_TOKENS_PER_SAMPLE"
    --train-log-every "${TRAIN_LOG_EVERY:-1000}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE:-0}"
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA:-0.01}"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --local-checkpoint-keep "$LOCAL_CHECKPOINT_KEEP"
    --gcs-sync-every "$GCS_SYNC_EVERY"
    --gcs-backup-keep "$GCS_BACKUP_KEEP"
    --auto-batch-max-per-device "${AUTO_BATCH_MAX_PER_DEVICE:-16}"
    --auto-batch-multiple-per-device "${AUTO_BATCH_MULTIPLE_PER_DEVICE:-8}"
    --auto-batch-memory-util "${AUTO_BATCH_MEMORY_UTIL:-0.78}"
    --epochs "${EPOCHS:-1}"
    --max-steps "${MAX_STEPS:-0}"
    --learning-rate "${LEARNING_RATE:-1e-4}"
    --warmup-steps "${WARMUP_STEPS:-5000}"
    --write-rate "${WRITE_RATE:-0.02}"
    --forget-rate "${FORGET_RATE:-0.002}"
    --interrupt-input-loss-weight 1.0
    --precision bfloat16
    --optimizer adamw
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --grad-clip-norm 1.0
    --edge-vram-mb 2048
    --edge-vram-util-target 0.70
    --quantization-bits 4
)

if [[ -n "${DATASET_MIX:-}" ]]; then
    TRAIN_ARGS+=(--dataset-mix "$DATASET_MIX")
fi

if [[ -n "${EVAL_DATASET_CASES:-}" ]]; then
    TRAIN_ARGS+=(--eval-dataset-cases "$EVAL_DATASET_CASES")
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

if [[ -n "${EVAL_AUDIO_TOKENS:-}" ]]; then
    TRAIN_ARGS+=(--eval-audio-tokens "$EVAL_AUDIO_TOKENS")
fi

if [[ -n "${EVAL_AUDIO_INPUT_AUDIO_TOKENS:-}" ]]; then
    TRAIN_ARGS+=(--eval-audio-input-audio-tokens "$EVAL_AUDIO_INPUT_AUDIO_TOKENS")
fi

if [[ -n "${AUDIO_MIN_GENERATION_TOKENS:-}" ]]; then
    TRAIN_ARGS+=(--audio-min-generation-tokens "$AUDIO_MIN_GENERATION_TOKENS")
fi

case "${FORCE_TRAIN_TOKENIZER,,}" in
    1|true|yes|on)
        TRAIN_ARGS+=(--force-train-tokenizer)
        ;;
esac

case "${CHECKPOINT_RESUME,,}" in
    0|false|no|off)
        TRAIN_ARGS+=(--no-checkpoint-resume)
        ;;
esac

if [[ -n "$GCS_BACKUP_DIR" ]]; then
    TRAIN_ARGS+=(--gcs-backup-dir "$GCS_BACKUP_DIR")
fi

case "${ASR_EVAL_CASE_FOLD,,}" in
    1|true|yes|on)
        TRAIN_ARGS+=(--asr-eval-case-fold)
        ;;
esac

case "${NO_ASR_EVAL_CASE_FOLD,,}" in
    1|true|yes|on)
        TRAIN_ARGS+=(--no-asr-eval-case-fold)
        ;;
esac

if [[ -x "$PROJECT_ROOT/.venv/bin/python3" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python3"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

run_tpu_preflight() {
    case "${TPU_PREFLIGHT,,}" in
        0|false|no|off)
            return 0
            ;;
    esac

    local expected="$TPU_EXPECTED_DEVICE_COUNT"
    local found=0
    local needs_bind=()
    local id vendor_hex device_hex device_path driver_path driver_name
    for id in $TPU_VFIO_DEVICE_IDS; do
        vendor_hex="${id%%:*}"
        device_hex="${id##*:}"
        for device_path in /sys/bus/pci/devices/*; do
            [[ -f "$device_path/vendor" && -f "$device_path/device" ]] || continue
            [[ "$(cat "$device_path/vendor")" == "0x$vendor_hex" ]] || continue
            [[ "$(cat "$device_path/device")" == "0x$device_hex" ]] || continue
            found=$((found + 1))
            driver_path="$device_path/driver"
            driver_name=""
            if [[ -e "$driver_path" ]]; then
                driver_name="$(basename "$(readlink -f "$driver_path")")"
            fi
            if [[ "$driver_name" != "vfio-pci" ]]; then
                needs_bind+=("$(basename "$device_path")")
            fi
        done
    done

    if (( found == 0 )); then
        echo "TPU preflight: no TPU PCI devices matching TPU_VFIO_DEVICE_IDS=$TPU_VFIO_DEVICE_IDS; skipping vfio bind check."
    elif (( found < expected )); then
        echo "TPU preflight: found $found TPU PCI devices, expected $expected." >&2
        exit 2
    elif (( ${#needs_bind[@]} > 0 )); then
        case "${TPU_ENABLE_VFIO_BIND,,}" in
            0|false|no|off)
                echo "TPU preflight: ${#needs_bind[@]} TPU devices are not bound to vfio-pci: ${needs_bind[*]}" >&2
                exit 2
                ;;
        esac
        if [[ ! -x /usr/lib/udev/bind_to_vfio_pci.sh ]]; then
            echo "TPU preflight: /usr/lib/udev/bind_to_vfio_pci.sh is missing; cannot bind ${needs_bind[*]} to vfio-pci." >&2
            exit 2
        fi
        if ! sudo -n true 2>/dev/null; then
            echo "TPU preflight: sudo without a password is required to bind TPU devices to vfio-pci." >&2
            exit 2
        fi
        if systemctl list-unit-files tpu-runtime.service >/dev/null 2>&1; then
            sudo -n systemctl stop tpu-runtime.service >/dev/null 2>&1 || true
        fi
        for device_path in "${needs_bind[@]}"; do
            echo "TPU preflight: binding $device_path to vfio-pci"
            sudo -n /usr/lib/udev/bind_to_vfio_pci.sh "$device_path"
        done
    fi

    if systemctl list-unit-files tpu-runtime.service >/dev/null 2>&1; then
        sudo -n systemctl start tpu-runtime.service >/dev/null 2>&1 || true
    fi

    echo "TPU preflight: checking JAX TPU backend"
    if ! timeout "$TPU_PREFLIGHT_TIMEOUT" "$PYTHON_BIN" - <<'PY'
import os
import jax

devices = jax.devices()
print(f"JAX backend={jax.default_backend()} device_count={len(devices)}", flush=True)
if jax.default_backend() != "tpu" or not devices:
    raise SystemExit("JAX did not initialize a TPU backend")
os._exit(0)
PY
    then
        echo "TPU preflight failed; check tpu-runtime.service, vfio-pci bindings, and LIBTPU_INIT_ARGS." >&2
        exit 2
    fi
}

run_tpu_preflight

if [ "$FOREGROUND" = true ]; then
    LOG_PATH="$PROJECT_ROOT/logs/train_$(date -u +%Y%m%dT%H%M%SZ).log"
    ln -sfn "$(basename "$LOG_PATH")" "$PROJECT_ROOT/logs/train.latest.log"
    echo "Starting training in foreground. Writing logs to $LOG_PATH"
    "$PYTHON_BIN" -u "$PROJECT_ROOT/train.py" "${TRAIN_ARGS[@]}" \
        2>&1 | tee "$LOG_PATH"
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
