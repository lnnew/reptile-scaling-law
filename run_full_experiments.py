#!/usr/bin/env python3
"""
Full Scaling Law Experiments
N_tasks = [50, 100, 300, 1000]
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
sys.path.append('/root/ssd/reptile-scaling-law')

from experiment_runner import ScalingLawExperimentRunner

    # Optimized configuration (5-8x faster)
    config = {
        'lora_r': 8,
        'lora_alpha': 16,
        'inner_lr': 1e-4,  # 5e-4 → 1e-4 (more stable)
        'meta_lr': 0.05,  # 0.1 → 0.05 (prevent NaN)
        'k_inner': 3,  # 5 → 3 (40% faster per step)
        'meta_batch_size': 2,  # 1 → 2 (2x throughput)
        'inner_batch_size': 16,  # 8 → 16 (better GPU utilization)
        'num_meta_steps': 10000,  # 2000 → 10000 (as requested)
        'eval_interval': 500,
        'num_eval_tasks': 50,  # 100 → 50 (2x faster eval)
        'num_meta_test_tasks': 100,  # 200 → 100 (2x faster test)
        'load_in_8bit': False  # Disable 8-bit (causes NaN)
    }N_TASKS_LIST = [50, 100, 300, 1000]

print("="*80)
print("REPTILE SCALING LAW EXPERIMENTS")
print("="*80)
print(f"N_tasks to test: {N_TASKS_LIST}")
print(f"GPUs: 0-7 (via CUDA_VISIBLE_DEVICES)")
print(f"Estimated time: ~15-20 hours (with 8 GPUs)")
print("="*80)

# Initialize runner
runner = ScalingLawExperimentRunner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks_list=N_TASKS_LIST,
    base_config=CONFIG,
    seed_meta_test=0,
    seed_train_base=100,
    save_root='./experiments_scaling_law',
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]  # All 8 GPUs
)

# Run all experiments
print("\nStarting experiments...")
runner.run_all_experiments()

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE!")
print("="*80)
print("\nResults saved to: ./experiments_scaling_law/")
print("\nKey files:")
print("  - scaling_law_results.csv")
print("  - power_law_fit.json")
print("  - scaling_law_plot.png")
print("  - summary_report.json")
