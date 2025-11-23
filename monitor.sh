#!/bin/bash

# Monitor reptile-scaling-law experiment progress

echo "=================================="
echo "REPTILE EXPERIMENT MONITOR"
echo "=================================="
echo ""

# Check if process is running
PID_FILE="/home/jihyun/reptile-scaling-law/run_full.pid"
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✓ Process running (PID: $PID)"
        echo ""
        
        # CPU and Memory usage
        ps -p $PID -o pid,%cpu,%mem,rss,etime,cmd | tail -n +2
        echo ""
    else
        echo "✗ Process not running"
        echo ""
    fi
else
    echo "✗ PID file not found"
    echo ""
fi

# GPU Status
echo "GPU STATUS:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
while IFS=, read -r idx name util mem_used mem_total; do
    printf "  GPU %s: %3s%% util, %5s/%5s MB\n" "$idx" "$util" "$mem_used" "$mem_total"
done
echo ""

# Latest log entries
echo "LATEST LOG (last 15 lines):"
echo "----------------------------"
tail -15 /home/jihyun/reptile-scaling-law/experiments_full.log | grep -v "^$"
echo ""

# Check for errors
ERROR_COUNT=$(grep -i "error\|exception\|traceback" /home/jihyun/reptile-scaling-law/experiments_full.log 2>/dev/null | wc -l)
if [ $ERROR_COUNT -gt 0 ]; then
    echo "⚠ WARNING: Found $ERROR_COUNT error messages in log"
    echo ""
fi

# Experiment progress
echo "EXPERIMENT PROGRESS:"
echo "----------------------------"
grep -E "N_tasks=|Meta-step|Experiment|Zero-shot|Baseline" /home/jihyun/reptile-scaling-law/experiments_full.log 2>/dev/null | tail -5
echo ""

echo "=================================="
echo "Monitor script complete"
echo "Run: watch -n 10 ./monitor.sh"
echo "=================================="
