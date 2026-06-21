#!/bin/bash
LOG_FILE="/mnt/disks/propagator-cache/propagator/logs/memory_monitor.log"
echo "=== Memory Monitor Started at $(date) ===" > "$LOG_FILE"

PID=$(cat /mnt/disks/propagator-cache/propagator/logs/train.pid 2>/dev/null)
if [ -z "$PID" ]; then
    PID=$(pgrep -f "train.py" | head -n 1)
fi

echo "Monitoring PID: $PID" >> "$LOG_FILE"

while true; do
    echo "=========================================" >> "$LOG_FILE"
    echo "Timestamp: $(date -u +'%Y-%m-%dT%H:%M:%SZ')" >> "$LOG_FILE"
    echo "--- System Memory ---" >> "$LOG_FILE"
    free -h >> "$LOG_FILE"
    
    echo "--- /proc/meminfo Cache Details ---" >> "$LOG_FILE"
    grep -E '^(Active|Inactive|MemFree|Cached|Dirty|Writeback|Mapped|Shmem|Active\(anon\)|Inactive\(anon\)|Active\(file\)|Inactive\(file\)):' /proc/meminfo >> "$LOG_FILE"
    
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "--- Process $PID Memory ---" >> "$LOG_FILE"
        ps -p "$PID" -o pid,ppid,rss,vsz,%mem,%cpu,time,comm >> "$LOG_FILE"
        
        echo "--- Process $PID Child Processes ---" >> "$LOG_FILE"
        ps --ppid "$PID" -o pid,rss,vsz,%mem,%cpu,comm >> "$LOG_FILE"
    else
        echo "Process $PID is not running." >> "$LOG_FILE"
        PID=$(pgrep -f "train.py" | head -n 1)
        if [ -n "$PID" ]; then
            echo "Found new PID: $PID" >> "$LOG_FILE"
        fi
    fi
    
    echo "--- Top 5 Memory Consuming Processes ---" >> "$LOG_FILE"
    ps -eo pid,ppid,rss,vsz,%mem,comm --sort=-rss | head -n 6 >> "$LOG_FILE"
    
    sleep 10
done
