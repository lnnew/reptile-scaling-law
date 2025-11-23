#!/usr/bin/env python3
"""
Run a single N_tasks experiment on a specific GPU
Usage: python run_single_ntask.py <n_tasks> <gpu_id>
"""
import sys
import os

if len(sys.argv) != 3:
    print("Usage: python run_single_ntask.py <n_tasks> <gpu_id>")
    sys.exit(1)

n_tasks = int(sys.argv[1])
gpu_id = sys.argv[2]

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

sys.path.append('/home/jihyun/reptile-scaling-law')

from experiment_runner import ScalingLawExperimentRunner

# Configuration optimized for stability and speed
CONFIG = {
    'n_way': 5,
    'k_support': 5,
    'k_query': 15,
    'max_length': 64,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'inner_lr': 1e-4,      # Lowered for stability
    'meta_lr': 0.05,       # Lowered for stability
    'k_inner': 3,
    'meta_batch_size': 2,
    'inner_batch_size': 16,
    'num_meta_steps': 10000, # Increased to 10000
    'eval_interval': 500,
    'num_eval_tasks': 50,
    'num_meta_test_tasks': 100,
    'load_in_8bit': False,  # Disabled to prevent NaN
    'gradient_checkpointing': False # Disabled for speed (20-30% faster) if VRAM allows
}

print("="*80)
print(f"SINGLE N_TASKS EXPERIMENT: N_tasks={n_tasks}")
print("="*80)
print(f"GPU: {gpu_id}")
print(f"Config: meta_steps={CONFIG['num_meta_steps']}, k_inner={CONFIG['k_inner']}")
print("="*80)
print()

# Initialize runner with single N_tasks value
runner = ScalingLawExperimentRunner(
    model_name='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    base_config=CONFIG,
    n_tasks_list=[n_tasks],  # Only this N_tasks
    devices=['cuda:0'],  # Will be GPU specified by CUDA_VISIBLE_DEVICES
    save_root=f'results/ntasks_{n_tasks}_gpu{gpu_id}/'
)

# Run experiment
runner.run_all_experiments()

print()
print("="*80)
print(f"COMPLETED: N_tasks={n_tasks} on GPU {gpu_id}")
print("="*80)
