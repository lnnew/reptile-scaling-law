#!/bin/bash

echo "========================================"
echo "WAIT FOR CURRENT EXPERIMENT TO FINISH"
echo "========================================"
echo ""

PID_FILE="/home/jihyun/reptile-scaling-law/run_full.pid"
if [ ! -f "$PID_FILE" ]; then
    echo "Error: PID file not found"
    exit 1
fi

OLD_PID=$(cat $PID_FILE)
echo "Monitoring PID: $OLD_PID"
echo "Started at: $(date)"
echo ""

# Wait for process to finish
while ps -p $OLD_PID > /dev/null 2>&1; do
    # Show progress every 5 minutes
    PROGRESS=$(tail -1 /home/jihyun/reptile-scaling-law/experiments_full.log 2>/dev/null | grep -oP '\d+%' || echo "running")
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0)
    echo "[$(date +%H:%M:%S)] Process still running - Progress: $PROGRESS, GPU0: ${GPU_UTIL}%"
    sleep 300  # Check every 5 minutes
done

echo ""
echo "========================================"
echo "PREVIOUS EXPERIMENT COMPLETED"
echo "========================================"
echo "Finished at: $(date)"
echo ""

# Backup old log
BACKUP_LOG="/home/jihyun/reptile-scaling-law/experiments_full_$(date +%Y%m%d_%H%M%S).log"
cp /home/jihyun/reptile-scaling-law/experiments_full.log "$BACKUP_LOG"
echo "Old log backed up to: $BACKUP_LOG"
echo ""

# Clean GPU memory
echo "Cleaning GPU memory..."
sleep 10
nvidia-smi

# Check if GPU is free
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
if [ "$GPU_MEM" -gt 1000 ]; then
    echo "Warning: GPU still has $GPU_MEM MB allocated. Waiting 30s..."
    sleep 30
fi

echo ""
echo "========================================"
echo "STARTING OPTIMIZED EXPERIMENT"
echo "========================================"
echo "Optimizations applied:"
echo "  - meta_steps: 10000 → 2000 (5x faster)"
echo "  - k_inner: 5 → 3 (40% faster per step)"
echo "  - meta_batch_size: 1 → 2 (2x throughput)"
echo "  - inner_batch_size: 8 → 16 (better GPU util)"
echo "  - eval tasks: 100 → 50 (2x faster eval)"
echo "  - test tasks: 200 → 100 (2x faster test)"
echo ""
echo "Expected speedup: 5-8x faster"
echo "Starting at: $(date)"
echo ""

# Start new experiment
cd /home/jihyun/reptile-scaling-law
nohup bash -c 'source /home/jihyun/miniconda3/etc/profile.d/conda.sh && conda activate reptile && python run_full_experiments.py' > experiments_full.log 2>&1 & 
NEW_PID=$!
echo $NEW_PID > run_full.pid

echo "New experiment started with PID: $NEW_PID"
echo ""

# Wait a bit and check if it's running
sleep 30
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✓ New experiment is running successfully"
    echo ""
    nvidia-smi
    echo ""
    echo "Monitor with: watch -n 10 ./monitor.sh"
else
    echo "✗ Error: New experiment failed to start"
    echo "Check log: tail -50 experiments_full.log"
fi

echo ""
echo "========================================"
echo "RESTART COMPLETE"
echo "========================================"
