#!/bin/bash
echo "=== 8-GPU EXPERIMENT STATUS ==="
echo ""
echo "ACTIVE EXPERIMENTS:"
ps aux | grep "python run_single_ntask.py" | grep -v grep | awk '{print "  GPU " $NF ": N_tasks=" $(NF-1) " (PID: " $2 ")"}'
echo ""
echo "GPU UTILIZATION:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{printf "  GPU %s: %3s%% util, %5s/%5s MB\n", $1, $2, $3, $4}'
echo ""
echo "CONFIGURATION:"
echo "  GPU 0: N_tasks=50   | GPU 1: N_tasks=150"
echo "  GPU 2: N_tasks=100  | GPU 3: N_tasks=200" 
echo "  GPU 4: N_tasks=300  | GPU 5: N_tasks=500"
echo "  GPU 6: N_tasks=1000 | GPU 7: N_tasks=700"
echo ""
echo "More detailed status: ./monitor_parallel.sh"
