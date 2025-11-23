#!/bin/bash
# Real-time progress monitor for Reptile Scaling Law experiments

LOG_FILE="experiments_full.log"
PID_FILE="run_full.pid"

# Save PID
pgrep -f "run_full_experiments.py" > $PID_FILE

echo "================================================"
echo "Reptile Scaling Law Experiment Monitor"
echo "================================================"
echo ""

# Check if running
if [ -f "$PID_FILE" ]; then
    PID=$(cat $PID_FILE)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Experiment is RUNNING (PID: $PID)"
        echo ""
        
        # Show GPU usage
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
            awk '{printf "  GPU %s: %s%% utilization, %s/%s MB\n", $1, $2, $3, $4}'
        echo ""
        
        # Show last 30 lines of log
        echo "Recent Progress:"
        echo "----------------"
        tail -30 $LOG_FILE | grep -E "(EXPERIMENT|N_tasks|Meta-step|Evaluating|Loss:|Acc:|COMPLETE|Stage)" || \
            tail -30 $LOG_FILE
    else
        echo "❌ Experiment STOPPED (check logs)"
    fi
else
    echo "⚠️  No PID file found"
fi

echo ""
echo "================================================"
echo "Commands:"
echo "  tail -f $LOG_FILE    # Watch live log"
echo "  nvidia-smi -l 5      # Monitor GPU usage"
echo "  cat $PID_FILE        # Get PID"
echo "================================================"
