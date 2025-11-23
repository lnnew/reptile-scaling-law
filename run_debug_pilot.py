#!/usr/bin/env python3
"""
Debug Pilot Experiment
Runs a single N_tasks=50 experiment on GPU 0 with verbose logging and safe config.
"""
import os
import sys
import torch

# Force GPU 0
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from experiment_runner import ScalingLawExperimentRunner

# Safe configuration for debugging
# Lower learning rates, gradient clipping is now in trainer
DEBUG_CONFIG = {
    'n_way': 5,
    'k_support': 5,
    'k_query': 15,
    'max_length': 64,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'inner_lr': 1e-5,      # Very conservative inner LR
    'meta_lr': 0.001,      # Very conservative meta LR
    'k_inner': 3,
    'meta_batch_size': 2,
    'inner_batch_size': 8, # Smaller batch size
    'num_meta_steps': 50,  # Short run for debugging
    'eval_interval': 10,   # Frequent eval
    'num_eval_tasks': 10,
    'num_meta_test_tasks': 20,
    'load_in_8bit': False,  # Disable 8-bit (unnecessary for 1.1B LoRA and causes NaN)
    'use_amp': True         # Re-enable AMP for speed
}

print("="*80)
print("DEBUG PILOT EXPERIMENT")
print("="*80)
print(f"Config: {DEBUG_CONFIG}")
print("="*80)

runner = ScalingLawExperimentRunner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks_list=[50],
    base_config=DEBUG_CONFIG,
    seed_meta_test=0,
    seed_train_base=100,
    save_root='./results_debug',
    devices=["cuda:0"]
)

print("\nStarting debug run...")
runner.run_all_experiments()
print("\nDebug run complete!")
