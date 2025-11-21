#!/usr/bin/env python3
"""
Full Scaling Law Experiments
N_tasks = [50, 100, 300, 1000]
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
sys.path.append('/root/ssd/reptile-scaling-law')

from experiment_runner import ScalingLawExperimentRunner

# Configuration
CONFIG = {
    'n_way': 5,
    'k_support': 5,
    'k_query': 15,
    'max_length': 64,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'inner_lr': 5e-4,
    'meta_lr': 0.1,
    'k_inner': 5,
    'meta_batch_size': 2,  # Reduced for memory
    'inner_batch_size': 10,  # Reduced for memory
    'num_meta_steps': 10000,
    'eval_interval': 500,
    'num_eval_tasks': 100,
    'num_meta_test_tasks': 200,
    'load_in_8bit': True  # Enable 8-bit quantization
}

N_TASKS_LIST = [50, 100, 300, 1000]

print("="*80)
print("REPTILE SCALING LAW EXPERIMENTS")
print("="*80)
print(f"N_tasks to test: {N_TASKS_LIST}")
print(f"GPUs: 4, 5, 6, 7 (via CUDA_VISIBLE_DEVICES)")
print(f"Estimated time: ~20-25 hours")
print("="*80)

# Initialize runner
runner = ScalingLawExperimentRunner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks_list=N_TASKS_LIST,
    base_config=CONFIG,
    seed_meta_test=0,
    seed_train_base=100,
    save_root='./experiments_scaling_law',
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # Maps to actual 4,5,6,7
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
