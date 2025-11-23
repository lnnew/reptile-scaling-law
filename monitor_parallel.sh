#!/bin/bash

echo "========================================"
echo "PARALLEL EXPERIMENTS MONITOR"
echo "========================================"
date
echo ""

cd /home/jihyun/reptile-scaling-law

echo "RUNNING PROCESSES:"
echo "----------------------------------------"
for config in "50_gpu0" "50_gpu1" "100_gpu2" "100_gpu3" "300_gpu4" "300_gpu5" "1000_gpu6" "1000_gpu7"; do
    pid_file="pids/ntasks_${config}.pid"
    ntasks=${config%%_*}
    gpu=${config##*gpu}
    if [ -f "$pid_file" ]; then
        pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            runtime=$(ps -p $pid -o etime= | xargs)
            cpu=$(ps -p $pid -o %cpu= | xargs)
            mem=$(ps -p $pid -o rss= | awk '{printf "%.1fGB", $1/1024/1024}')
            echo "✓ N=$ntasks GPU$gpu | PID: $pid | Time: $runtime | CPU: ${cpu}% | Mem: $mem"
        else
            echo "✗ N=$ntasks GPU$gpu | Process stopped"
        fi
    else
        echo "- N=$ntasks GPU$gpu | Not started"
    fi
done

echo ""
echo "GPU STATUS:"
echo "----------------------------------------"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
while IFS=, read -r idx name util mem_used mem_total; do
    printf "GPU %s: %3s%% util | %5s/%5s MB\n" "$idx" "$util" "$mem_used" "$mem_total"
done

echo ""
echo "PROGRESS (last line from each log):"
echo "----------------------------------------"
for config in "50_gpu0" "50_gpu1" "100_gpu2" "100_gpu3" "300_gpu4" "300_gpu5" "1000_gpu6" "1000_gpu7"; do
    log_file="logs/ntasks_${config}.log"
    ntasks=${config%%_*}
    gpu=${config##*gpu}
    if [ -f "$log_file" ]; then
        # Get progress from log
        progress=$(tail -20 "$log_file" | grep -E "(Meta-training|Baseline|Zero-shot)" | tail -1 | head -c 60)
        if [ -n "$progress" ]; then
            echo "N=$ntasks G$gpu: $progress"
        else
            echo "N=$ntasks G$gpu: Starting..."
        fi
    fi
done

echo ""
echo "========================================"
echo "Commands:"
echo "  Logs:       tail -f logs/ntasks_<N>_gpu<X>.log"
echo "  Kill all:   pkill -f run_single_ntask.py"
echo "  Kill one:   kill \$(cat pids/ntasks_<N>_gpu<X>.pid)"
echo "========================================"
