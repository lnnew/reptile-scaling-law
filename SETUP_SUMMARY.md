# Reptile Scaling Law - GPU Setup Summary

## Changes Made (November 21, 2025)

### Environment Setup
- Created new conda environment: `reptile` with Python 3.10
- Installed all dependencies from requirements.txt
- Path: `/home/jihyun/miniconda3/envs/reptile`

### GPU Configuration Changes
**From:** GPU 4,5,6,7 (4 GPUs) with 48GB VRAM assumption
**To:** GPU 0-7 (8 GPUs) with 10GB VRAM each

### Files Modified

#### 1. GPU Device Settings (CUDA_VISIBLE_DEVICES)
- `test_gpu.py` - Updated to use GPUs 0-7
- `test_quick.py` - Updated to use GPUs 0-7
- `test_basic.py` - Updated to use GPUs 0-7
- `clear_gpu.py` - Updated to use GPUs 0-7
- `run_pilot.py` - Updated to use GPUs 0-7
- `run_pilot_small.py` - Updated to use GPUs 0-7
- `run_full_experiments.py` - Updated to use GPUs 0-7

#### 2. Device List Parameters (devices=[...])
- `experiment_runner.py` - Default devices changed to all 8 GPUs
- `reptile_trainer.py` - Default devices changed to all 8 GPUs
- `run_experiments.py` - Updated device list
- `run_pilot.py` - Updated device list
- `run_pilot_small.py` - Updated device list
- `run_full_experiments.py` - Updated device list

#### 3. Memory Optimization for 10GB VRAM
Updated configuration in multiple files:

**run_pilot_small.py:**
- `lora_r`: 8 → 4 (reduced rank)
- `lora_alpha`: 16 → 8 (reduced alpha)
- `load_in_8bit`: False → True
- `meta_batch_size`: 2 → 1
- `inner_batch_size`: 25 → 15

**run_full_experiments.py:**
- `lora_r`: 16 → 8
- `lora_alpha`: 32 → 16
- `meta_batch_size`: 2 → 1
- `inner_batch_size`: 10 → 8
- `load_in_8bit`: True (kept enabled)

**experiment_runner.py:**
- Updated default config with same optimizations as above

## GPU Status
```
8x NVIDIA GeForce RTX 2080 Ti (11GB each)
Available memory per GPU: ~10.8GB
```

## How to Run

### Activate Environment
```bash
conda activate reptile
```

### Quick Test
```bash
python test_quick.py
```

### Small Pilot (Fast)
```bash
python run_pilot_small.py
```

### Full Pilot
```bash
python run_pilot.py
```

### Full Experiments
```bash
python run_full_experiments.py
```

## Performance Notes
- With 8 GPUs instead of 4, training should be ~2x faster
- Reduced batch sizes and LoRA rank to fit 10GB VRAM
- 8-bit quantization enabled for memory efficiency
- Estimated time for full experiments: 15-20 hours (down from 20-25 hours)

## Verification
✅ All tests pass successfully
✅ Model loads without OOM errors
✅ Forward/backward passes work correctly
✅ Multi-GPU parallelism functional
