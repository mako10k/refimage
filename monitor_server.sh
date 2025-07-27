#!/bin/bash

# RefImage Server Monitor Script
# Purpose: Monitor server status and log file changes in real-time

LOG_DIR="/home/mako10k/imagestore/backend/logs"
LATEST_LOG_PATTERN="refimage_combined_*.log"

echo "================================================"
echo "RefImage Server Monitor"
echo "Timestamp: $(date)"
echo "Monitoring logs in: ${LOG_DIR}"
echo "================================================"

# Function to get latest log file
get_latest_log() {
    find "${LOG_DIR}" -name "${LATEST_LOG_PATTERN}" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-
}

# Function to check server status
check_server_status() {
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        echo "[$(date)] ✅ Server is RUNNING"
        return 0
    else
        echo "[$(date)] ❌ Server is DOWN"
        return 1
    fi
}

# Function to show log tail
show_recent_logs() {
    local log_file=$(get_latest_log)
    if [ -n "$log_file" ] && [ -f "$log_file" ]; then
        echo "Recent logs from: $(basename $log_file)"
        echo "----------------------------------------"
        tail -10 "$log_file"
        echo "----------------------------------------"
    fi
}

# Main monitoring loop
while true; do
    clear
    echo "RefImage Server Monitor - $(date)"
    echo "================================================"
    
    # Check server status
    if check_server_status; then
        # Server is running - show access logs
        show_recent_logs
    else
        # Server is down - show error details
        echo "🔍 Server appears to be down. Checking logs..."
        show_recent_logs
        
        # Check if process exists
        if pgrep -f "uvicorn.*refimage" > /dev/null; then
            echo "🔄 Process found but not responding to HTTP requests"
        else
            echo "💀 No RefImage process found"
        fi
    fi
    
    echo "================================================"
    echo "Press Ctrl+C to stop monitoring"
    sleep 10
done
