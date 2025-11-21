#!/usr/bin/env python3
"""
Minimal test with available GPU memory (~7GB free on GPU 4)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
sys.path.append('/root/ssd/reptile-scaling-law')

import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("="*60)
print("MINIMAL TEST - Using ~7GB available memory")
print("="*60)

try:
    from llm_model import LLMLoRAClassifier
    from banking77_data import Banking77TaskSampler
    
    # Ultra minimal config
    print("\nLoading model with 8-bit quantization...")
    model_wrapper = LLMLoRAClassifier(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels=5,
        lora_r=4,  # Minimal rank
        lora_alpha=8,
        lora_dropout=0.05,
        device="cuda:0",
        load_in_8bit=True  # Use 8-bit
    )
    
    print(f"\nGPU memory allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    print(f"GPU memory reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    
    print("\n✓ Model loaded successfully with available memory!")
    
except Exception as e:
    print(f"\n✗ Failed: {e}")
    import traceback
    traceback.print_exc()
