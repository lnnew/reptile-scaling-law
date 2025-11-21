#!/usr/bin/env python3
"""
Quick test script to verify all components work on GPUs 4,5,6,7
"""
import sys
sys.path.append('/root/ssd/reptile-scaling-law')

import torch
import os

# Set environment to use only GPUs 4,5,6,7
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

print("="*60)
print("QUICK TEST: Banking77 + Reptile on GPUs 4,5,6,7")
print("="*60)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from llm_model import LLMLoRAClassifier
    from banking77_data import Banking77TaskSampler
    from reptile_trainer import ReptileLLMTrainer
    print("✓ All modules imported successfully")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Initialize model
print("\n[2/5] Initializing TinyLlama model with LoRA...")
try:
    model = LLMLoRAClassifier(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=5,
        lora_r=8,  # Smaller for quick test
        lora_alpha=16,
        device='cuda:0',  # Will map to actual GPU 4
        load_in_8bit=False
    )
    print(f"✓ Model loaded on device: {model.device}")
except Exception as e:
    print(f"✗ Model initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Initialize data sampler
print("\n[3/5] Initializing Banking77 data sampler...")
try:
    sampler = Banking77TaskSampler(
        tokenizer=model.tokenizer,
        max_length=64,
        n_way=5,
        k_support=5,
        k_query=15
    )
    print(f"✓ Data sampler initialized")
    print(f"  Total classes: {sampler.num_total_classes}")
except Exception as e:
    print(f"✗ Data sampler error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generate and sample a task
print("\n[4/5] Testing task sampling...")
try:
    tasks = sampler.generate_task_pool(n_tasks=3, split='train', seed=42)
    print(f"✓ Generated {len(tasks)} tasks")
    
    support, query, s_labels, q_labels = sampler.sample_episode(tasks[0], split='train')
    print(f"  Support: {support['input_ids'].shape}, Query: {query['input_ids'].shape}")
except Exception as e:
    print(f"✗ Task sampling error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Forward pass
print("\n[5/5] Testing forward pass...")
try:
    model.train()
    with torch.no_grad():
        outputs = model.forward(
            support['input_ids'][:5].to('cuda:0'),
            support['attention_mask'][:5].to('cuda:0'),
            s_labels[:5].to('cuda:0')
        )
        print(f"✓ Forward pass successful")
        print(f"  Loss: {outputs.loss.item():.4f}")
        print(f"  Logits shape: {outputs.logits.shape}")
except Exception as e:
    print(f"✗ Forward pass error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nReady to run full experiments!")
