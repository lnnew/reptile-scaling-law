#!/usr/bin/env python3
"""
Simple pilot test with minimal configuration
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from experiment_runner import ScalingLawExperimentRunner

# Very conservative config - NO 8-bit quantization
config = {
    'n_way': 5,
    'k_support': 5,
    'k_query': 15,
    'max_length': 64,
    'lora_r': 8,
    'lora_alpha': 16,
    'lora_dropout': 0.05,
    'inner_lr': 1e-5,  # Very low to test stability
    'meta_lr': 0.01,
    'k_inner': 3,
    'meta_batch_size': 1,  # Single task at a time
    'inner_batch_size': 8,
    'num_meta_steps': 100,  # Just 100 steps to test
    'eval_interval': 50,
    'num_eval_tasks': 20,
    'num_meta_test_tasks': 20,
    'load_in_8bit': False  # DISABLE 8-bit quantization
}

print("="*80)
print("PILOT TEST - NO 8-BIT QUANTIZATION")
print("="*80)

runner = ScalingLawExperimentRunner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks_list=[50],
    base_config=config,
    seed_meta_test=0,
    seed_train_base=100,
    save_root='./results_pilot',
    devices=["cuda:0"]
)

runner.run_all_experiments()
print("\nPILOT TEST COMPLETE!")
