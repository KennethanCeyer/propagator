#!/bin/bash
LOG_FILE="/mnt/disks/propagator-cache/propagator/logs/cache_cleaner.log"
echo "=== Cache Cleaner Started at $(date) ===" > "$LOG_FILE"

while true; do
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Syncing filesystem and clearing page cache..." >> "$LOG_FILE"
    sync
    echo 3 > /proc/sys/vm/drop_caches
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Available memory after clear:" >> "$LOG_FILE"
    free -h >> "$LOG_FILE"
    sleep 3600
done
