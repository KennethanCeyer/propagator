#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PID_FILE="${PID_FILE:-$PROJECT_ROOT/logs/train.pid}"
LOG_FILE="${LOG_FILE:-$PROJECT_ROOT/logs/memory_guardrail.log}"
WARN_PCT="${WARN_PCT:-75}"
TERM_PCT="${TERM_PCT:-79.5}"
KILL_PCT="${KILL_PCT:-80}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"

mkdir -p "$(dirname "$LOG_FILE")"

log() {
    printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$LOG_FILE"
}

read_mem_pct() {
    awk '
        /^MemTotal:/ { total=$2 }
        /^MemAvailable:/ { avail=$2 }
        END {
            if (total <= 0) {
                print 0
            } else {
                printf "%.2f", 100.0 * (total - avail) / total
            }
        }
    ' /proc/meminfo
}

process_group_for_pid() {
    local pid="$1"
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]' || true)"
    if [[ -n "$pgid" ]]; then
        printf '%s\n' "$pgid"
    else
        printf '%s\n' "$pid"
    fi
}

log "Memory guardrail started: warn=${WARN_PCT}% term=${TERM_PCT}% kill=${KILL_PCT}% interval=${INTERVAL_SECONDS}s grace=${TERM_GRACE_SECONDS}s pid_file=$PID_FILE"
warned=0

while true; do
    if [[ ! -s "$PID_FILE" ]]; then
        log "PID file missing or empty; waiting."
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    pid="$(cat "$PID_FILE" 2>/dev/null | tr -d '[:space:]')"
    if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
        warned=0
        log "Training PID $pid is not running; waiting."
        sleep "$INTERVAL_SECONDS"
        continue
    fi

    used_pct="$(read_mem_pct)"
    pgid="$(process_group_for_pid "$pid")"

    if awk -v used="$used_pct" -v limit="$WARN_PCT" 'BEGIN { exit !(used >= limit) }'; then
        if [[ "$warned" == "0" ]]; then
            log "Memory ${used_pct}% >= ${WARN_PCT}%; warning threshold crossed for PID $pid process group $pgid."
            warned=1
        fi
    else
        warned=0
    fi

    if awk -v used="$used_pct" -v limit="$KILL_PCT" 'BEGIN { exit !(used >= limit) }'; then
        log "Memory ${used_pct}% >= ${KILL_PCT}%; killing process group $pgid."
        kill -KILL "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    elif awk -v used="$used_pct" -v limit="$TERM_PCT" 'BEGIN { exit !(used >= limit) }'; then
        log "Memory ${used_pct}% >= ${TERM_PCT}%; gracefully terminating process group $pgid."
        kill -TERM "-$pgid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        sleep "$TERM_GRACE_SECONDS"
        if kill -0 "$pid" 2>/dev/null; then
            log "PID $pid still alive after ${TERM_GRACE_SECONDS}s grace; killing process group $pgid."
            kill -KILL "-$pgid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    fi

    sleep "$INTERVAL_SECONDS"
done
