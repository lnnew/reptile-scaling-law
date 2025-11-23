#!/bin/bash

echo "========================================="
echo "QUICK STATUS CHECK"
echo "========================================="
echo ""

# Current experiment
CURR_PID=$(ps aux | grep "run_full_experiments.py" | grep -v grep | awk '{print $2}')
if [ -n "$CURR_PID" ]; then
    echo "✓ Current experiment running (PID: $CURR_PID)"
    RUNTIME=$(ps -p $CURR_PID -o etime= | xargs)
    echo "  Runtime: $RUNTIME"
    
    # Progress
    PROGRESS=$(tail -50 /home/jihyun/reptile-scaling-law/experiments_full.log | grep "Meta-training:" | tail -1 | grep -oP '\d+/\d+' || echo "checking...")
    echo "  Progress: $PROGRESS"
else
    echo "✗ No experiment running"
fi

echo ""

# Monitor script
MON_PID=$(ps aux | grep "wait_and_restart.sh" | grep -v grep | awk '{print $2}')
if [ -n "$MON_PID" ]; then
    echo "✓ Auto-restart monitor active (PID: $MON_PID)"
else
    echo "✗ No monitor running"
fi

echo ""

# GPU
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 0)
echo "GPU 0: $GPU_UTIL"

echo ""
echo "========================================="
echo "Commands:"
echo "  Current log:  tail -f experiments_full.log"
echo "  Monitor log:  tail -f restart.log"
echo "  Full status:  ./monitor.sh"
echo "========================================="
