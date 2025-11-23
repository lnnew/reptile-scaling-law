#!/usr/bin/env python3
"""
Clear GPU 0-7 memory cache
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import torch
import gc

print("Clearing GPU caches...")
for i in range(4):
    try:
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print(f"  GPU {i} (physical {i+4}): cleared")
    except Exception as e:
        print(f"  GPU {i} (physical {i+4}): {e}")

gc.collect()
print("\n✓ Cache clearing complete")

# Check memory
print("\nCurrent GPU memory usage:")
for i in range(4):
    torch.cuda.set_device(i)
    allocated = torch.cuda.memory_allocated(i) / 1e9
    reserved = torch.cuda.memory_reserved(i) / 1e9
    print(f"  GPU {i} (physical {i+4}): Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")
