#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-once}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs/propagator-multimodal_1b}"
GCS_BASE="${GCS_BASE:-${GCS_BACKUP_DIR:-gs://propagator-gde-project-aicloud/propagator-duplex}}"
INTERVAL_SECONDS="${GCS_SYNC_INTERVAL_SECONDS:-21600}"
CHECKPOINT_POLL_SECONDS="${GCS_CHECKPOINT_POLL_SECONDS:-300}"
CHECKPOINT_KEEP="${GCS_CHECKPOINT_KEEP:-5}"
INITIAL_FULL_SYNC="${GCS_INITIAL_FULL_SYNC:-true}"
LOG_DIR="$PROJECT_ROOT/logs"
LOCK_FILE="${GCS_SYNC_LOCK_FILE:-$LOG_DIR/gcs_periodic_sync.lock}"
PID_FILE="${GCS_SYNC_PID_FILE:-$LOG_DIR/gcs_periodic_sync.pid}"
PROJECT_EXCLUDE='(^|/)(\.venv|__pycache__|\.git)(/|$)|(^|/)outputs/cache(/|$)'
OUTPUT_EXCLUDE='(^|/)checkpoint(/|$)|(^|/)checkpoint\.orbax-checkpoint-tmp[^/]*(/|$)'

mkdir -p "$LOG_DIR"

log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

require_gcloud() {
    if ! command -v gcloud >/dev/null 2>&1; then
        log "gcloud is required for periodic GCS sync"
        exit 1
    fi
}

gcs_child() {
    local child="${1#/}"
    printf '%s/%s' "${GCS_BASE%/}" "$child"
}

latest_checkpoints() {
    if [[ ! -d "$OUTPUT_ROOT" ]]; then
        return 0
    fi
    find "$OUTPUT_ROOT" -maxdepth 2 -mindepth 2 -type d -name checkpoint 2>/dev/null \
        | sed -E 's#.*/step_([0-9]+)/checkpoint#\1 & #' \
        | sort -n \
        | tail -n "$CHECKPOINT_KEEP"
}

sync_dir() {
    local source="$1"
    local target="$2"
    if [[ ! -d "$source" ]]; then
        log "missing source dir, skipped: $source"
        return 0
    fi
    log "rsync $source -> $target"
    gcloud storage rsync "$source" "$target" --recursive --delete-unmatched-destination-objects
}

sync_project() {
    local target
    target="$(gcs_child project)"
    log "rsync project $PROJECT_ROOT -> $target"
    gcloud storage rsync "$PROJECT_ROOT" "$target" \
        --recursive \
        --delete-unmatched-destination-objects \
        --exclude "$PROJECT_EXCLUDE"
}

sync_output_without_checkpoints() {
    local target
    target="$(gcs_child output)"
    if [[ ! -d "$OUTPUT_ROOT" ]]; then
        log "missing output dir, skipped: $OUTPUT_ROOT"
        return 0
    fi
    log "rsync output without checkpoints $OUTPUT_ROOT -> $target"
    gcloud storage rsync "$OUTPUT_ROOT" "$target" \
        --recursive \
        --delete-unmatched-destination-objects \
        --exclude "$OUTPUT_EXCLUDE"
    prune_remote_output_checkpoints
}

prune_remote_output_checkpoints() {
    log "pruning checkpoint objects from output mirror"
    gcloud storage rm -r "$(gcs_child 'output/step_*/checkpoint')" 2>/dev/null || true
}

remote_checkpoint_steps() {
    gcloud storage ls -r "$(gcs_child 'sync_step_*/checkpoint/**')" 2>/dev/null \
        | sed -nE 's#.*/sync_step_([0-9]+)/checkpoint/.*#\1#p' \
        | sort -n \
        | uniq
}

prune_remote_checkpoints() {
    local remote_step keep_start idx
    local -a remote_steps
    mapfile -t remote_steps < <(remote_checkpoint_steps)
    if [[ "${#remote_steps[@]}" -le "$CHECKPOINT_KEEP" ]]; then
        return 0
    fi

    keep_start=$((${#remote_steps[@]} - CHECKPOINT_KEEP))
    for ((idx = 0; idx < keep_start; idx++)); do
        remote_step="${remote_steps[$idx]}"
        [[ -n "$remote_step" ]] || continue
        log "pruning remote checkpoint for step $remote_step"
        gcloud storage rm -r "$(gcs_child "sync_step_${remote_step}/checkpoint")" 2>/dev/null || true
    done
}

sync_latest_checkpoints() {
    local latest step checkpoint newest_step tmp
    local -a checkpoints
    mapfile -t checkpoints < <(latest_checkpoints)
    if [[ "${#checkpoints[@]}" -eq 0 ]]; then
        log "no checkpoint found under $OUTPUT_ROOT"
        return 0
    fi

    newest_step=""
    for latest in "${checkpoints[@]}"; do
        step="${latest%% *}"
        checkpoint="${latest#* }"
        checkpoint="${checkpoint% }"
        if [[ ! -d "$checkpoint" ]]; then
            log "checkpoint path missing, skipped: $checkpoint"
            continue
        fi
        newest_step="$step"
        sync_dir "$checkpoint" "$(gcs_child "sync_step_${step}/checkpoint")"
    done

    if [[ -z "$newest_step" ]]; then
        log "no valid checkpoint path found under $OUTPUT_ROOT"
        return 0
    fi

    latest="${checkpoints[$((${#checkpoints[@]} - 1))]}"
    checkpoint="${latest#* }"
    checkpoint="${checkpoint% }"
    sync_dir "$checkpoint" "$(gcs_child latest_checkpoint)"

    tmp="$(mktemp)"
    printf '%s\n' "$newest_step" > "$tmp"
    gcloud storage cp "$tmp" "$(gcs_child latest_checkpoint_step.txt)"
    rm -f "$tmp"
    log "latest checkpoint synced: step $newest_step"
    prune_remote_checkpoints
}

sync_checkpoints_once() {
    require_gcloud
    (
        if ! flock -n 9; then
            log "another GCS sync is already running; skipped"
            return 0
        fi

        log "checkpoint sync started: base=$GCS_BASE output=$OUTPUT_ROOT keep=$CHECKPOINT_KEEP"
        sync_latest_checkpoints
        log "checkpoint sync finished"
    ) 9>"$LOCK_FILE"
}

sync_once() {
    require_gcloud
    (
        if ! flock -n 9; then
            log "another GCS sync is already running; skipped"
            return 0
        fi

        log "sync started: base=$GCS_BASE output=$OUTPUT_ROOT"
        sync_project
        sync_output_without_checkpoints
        sync_latest_checkpoints
        log "sync finished"
    ) 9>"$LOCK_FILE"
}

run_loop() {
    echo "$$" > "$PID_FILE"
    log "periodic GCS sync loop started: full_interval=${INTERVAL_SECONDS}s checkpoint_poll=${CHECKPOINT_POLL_SECONDS}s pid=$$"
    local last_full_sync=0
    local now=0
    if [[ "${INITIAL_FULL_SYNC,,}" == "0" || "${INITIAL_FULL_SYNC,,}" == "false" || "${INITIAL_FULL_SYNC,,}" == "no" ]]; then
        last_full_sync="$(date +%s)"
    fi
    while true; do
        now="$(date +%s)"
        if (( now - last_full_sync >= INTERVAL_SECONDS )); then
            sync_once && last_full_sync="$(date +%s)" || log "sync failed with exit code $?"
        else
            sync_checkpoints_once || log "checkpoint sync failed with exit code $?"
        fi
        sleep "$CHECKPOINT_POLL_SECONDS"
    done
}

case "$MODE" in
    once)
        sync_once
        ;;
    loop)
        run_loop
        ;;
    *)
        echo "Usage: $0 [once|loop]" >&2
        exit 2
        ;;
esac
