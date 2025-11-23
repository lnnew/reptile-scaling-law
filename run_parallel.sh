#!/bin/bash

echo "========================================"
echo "PARALLEL EXPERIMENTS LAUNCHER (8 GPUs)"
echo "========================================"
echo ""
echo "Running 8 experiments in parallel with distinct N_tasks:"
echo "  GPU 0: N_tasks=50"
echo "  GPU 1: N_tasks=150"
echo "  GPU 2: N_tasks=100"
echo "  GPU 3: N_tasks=200"
echo "  GPU 4: N_tasks=300"
echo "  GPU 5: N_tasks=500"
echo "  GPU 6: N_tasks=1000"
echo "  GPU 7: N_tasks=700"
echo ""
echo "Expected time: ~4-6 hours (all parallel)"
echo "========================================"
echo ""

cd /home/jihyun/reptile-scaling-law

# Activate conda
source /home/jihyun/miniconda3/etc/profile.d/conda.sh
conda activate reptile

# Clean GPU memory
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"
sleep 2

# Create directories
mkdir -p logs pids results

# Launch 8 parallel experiments
echo "Starting experiments..."
echo ""

# GPU 0: N=50
nohup python run_single_ntask.py 50 0 > logs/ntasks_50_gpu0.log 2>&1 &
PID_0=$!
echo "GPU 0: N_tasks=50   (PID: $PID_0)"
echo $PID_0 > pids/ntasks_50_gpu0.pid
sleep 2

# GPU 1: N=150
nohup python run_single_ntask.py 150 1 > logs/ntasks_150_gpu1.log 2>&1 &
PID_1=$!
echo "GPU 1: N_tasks=150  (PID: $PID_1)"
echo $PID_1 > pids/ntasks_150_gpu1.pid
sleep 2

# GPU 2: N=100
nohup python run_single_ntask.py 100 2 > logs/ntasks_100_gpu2.log 2>&1 &
PID_2=$!
echo "GPU 2: N_tasks=100  (PID: $PID_2)"
echo $PID_2 > pids/ntasks_100_gpu2.pid
sleep 2

# GPU 3: N=200
nohup python run_single_ntask.py 200 3 > logs/ntasks_200_gpu3.log 2>&1 &
PID_3=$!
echo "GPU 3: N_tasks=200  (PID: $PID_3)"
echo $PID_3 > pids/ntasks_200_gpu3.pid
sleep 2

# GPU 4: N=300
nohup python run_single_ntask.py 300 4 > logs/ntasks_300_gpu4.log 2>&1 &
PID_4=$!
echo "GPU 4: N_tasks=300  (PID: $PID_4)"
echo $PID_4 > pids/ntasks_300_gpu4.pid
sleep 2

# GPU 5: N=500
nohup python run_single_ntask.py 500 5 > logs/ntasks_500_gpu5.log 2>&1 &
PID_5=$!
echo "GPU 5: N_tasks=500  (PID: $PID_5)"
echo $PID_5 > pids/ntasks_500_gpu5.pid
sleep 2

# GPU 6: N=1000
nohup python run_single_ntask.py 1000 6 > logs/ntasks_1000_gpu6.log 2>&1 &
PID_6=$!
echo "GPU 6: N_tasks=1000 (PID: $PID_6)"
echo $PID_6 > pids/ntasks_1000_gpu6.pid
sleep 2

# GPU 7: N=700
nohup python run_single_ntask.py 700 7 > logs/ntasks_700_gpu7.log 2>&1 &
PID_7=$!
echo "GPU 7: N_tasks=700  (PID: $PID_7)"
echo $PID_7 > pids/ntasks_700_gpu7.pid

echo ""
echo "All 8 experiments launched!"
echo ""

# Wait and check
sleep 30

echo "========================================"
echo "STATUS CHECK (after 30s)"
echo "========================================"
echo ""

for config in "50_gpu0" "150_gpu1" "100_gpu2" "200_gpu3" "300_gpu4" "500_gpu5" "1000_gpu6" "700_gpu7"; do
    pid=$(cat pids/ntasks_${config}.pid 2>/dev/null)
    ntasks=${config%%_*}
    gpu=${config##*gpu}
    if ps -p $pid > /dev/null 2>&1; then
        echo "✓ N_tasks=$ntasks GPU$gpu running (PID: $pid)"
    else
        echo "✗ N_tasks=$ntasks GPU$gpu failed - check logs/ntasks_${ntasks}_gpu${gpu}.log"
    fi
done

echo ""
echo "========================================"
echo "GPU STATUS"
echo "========================================"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader

echo ""
echo "========================================"
echo "MONITORING COMMANDS"
echo "========================================"
echo "  Check all:     ./monitor_parallel.sh"
echo "  Watch status:  watch -n 10 ./monitor_parallel.sh"
echo "  View log:      tail -f logs/ntasks_<N>_gpu<X>.log"
echo "  Kill all:      pkill -f run_single_ntask.py"
echo "========================================"
