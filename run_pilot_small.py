#!/usr/bin/env python3
"""
Small pilot experiment: N_tasks=50, 1000 steps
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import sys
sys.path.append('/root/ssd/reptile-scaling-law')

import torch
import numpy as np
import random

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("Importing modules...")
from llm_model import LLMLoRAClassifier
from banking77_data import Banking77TaskSampler
from reptile_trainer import ReptileLLMTrainer

print("\n" + "="*80)
print("PILOT EXPERIMENT: N_tasks=50, 1000 steps")
print("="*80)

# Clear GPU cache
for i in range(torch.cuda.device_count()):
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()

# Initialize model with memory optimization for 10GB VRAM
print("\nLoading model...")
model_wrapper = LLMLoRAClassifier(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_labels=5,
    lora_r=4,  # Reduced rank for 10GB VRAM
    lora_alpha=8,  # Reduced alpha for 10GB VRAM
    lora_dropout=0.05,
    device="cuda:0",
    load_in_8bit=True  # Enable 8-bit for memory efficiency
)

# Clear cache after model loading
torch.cuda.empty_cache()

print("\nLoading dataset...")
task_sampler = Banking77TaskSampler(
    tokenizer=model_wrapper.tokenizer,
    max_length=64,
    n_way=5,
    k_support=5,
    k_query=15
)

print("\nGenerating task pools...")
train_pool = task_sampler.generate_task_pool(50, split='train', seed=100)
test_pool = task_sampler.generate_task_pool(100, split='test', seed=0)

print(f"Train tasks: {len(train_pool)}")
print(f"Test tasks: {len(test_pool)}")

# Initialize trainer
print("\nInitializing trainer...")
trainer = ReptileLLMTrainer(
    model_wrapper=model_wrapper,
    task_sampler=task_sampler,
    train_task_pool=train_pool,
    test_task_pool=test_pool,
    inner_lr=5e-4,
    meta_lr=0.1,
    k_inner=5,
    meta_batch_size=1,  # Reduced for 10GB VRAM
    inner_batch_size=15,  # Reduced for 10GB VRAM
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]  # All GPUs
)

# Train
print("\nStarting training...")
try:
    trainer.train(
        num_meta_steps=1000,
        eval_interval=200,
        num_eval_tasks=50,
        save_dir='./pilot_small',
        experiment_name='pilot_n50'
    )
    print("\n✓ Pilot experiment completed successfully!")
except Exception as e:
    print(f"\n✗ Error during training: {e}")
    import traceback
    traceback.print_exc()
