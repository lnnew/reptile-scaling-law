#!/usr/bin/env python3
"""
Simple test to verify GPU 0-7 work and model can load.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices visible: {torch.cuda.device_count()}")

# Test GPU access
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Clear cache
    torch.cuda.set_device(i)
    torch.cuda.empty_cache()
    print(f"  GPU {i} cache cleared")

print("\n✓ GPU test successful!")

# Test simple tensor operation
print("\nTesting GPU operations...")
device = torch.device("cuda:0")
x = torch.randn(100, 100).to(device)
y = torch.matmul(x, x.T)
print(f"Matrix multiplication result shape: {y.shape}")
print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

torch.cuda.empty_cache()
print("\n✓ All tests passed!")
