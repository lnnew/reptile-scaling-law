#!/usr/bin/env python3
"""
Pilot experiment: N_tasks=100, 2000 meta-steps
Quick test before full scaling law experiment
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
sys.path.append('/root/ssd/reptile-scaling-law')

from experiment_runner import run_pilot_experiment

print("="*80)
print("PILOT EXPERIMENT")
print("="*80)
print("N_tasks: 100")
print("Meta-steps: 2000")
print("GPUs: 4,5,6,7")
print("="*80)

run_pilot_experiment(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks=100,
    num_meta_steps=2000,
    save_dir="./pilot_n100",
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # Maps to actual 4,5,6,7
)

print("\n✓ Pilot experiment complete!")
