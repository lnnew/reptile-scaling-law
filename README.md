# Investigating Scaling Laws in Meta-Learning with LLMs

**Research Question**: Does meta-learning performance follow a power law with respect to the number of training tasks?

$$L_{\text{meta}} \propto N_{\text{tasks}}^{-\beta}$$

## Project Overview

- **Dataset**: Banking77 (77 intent classification)
- **Base Model**: TinyLlama-1.1B with LoRA
- **Meta-Learning**: Reptile (First-order MAML)
- **Task**: 5-way 5-shot few-shot learning
- **Multi-GPU Support**: GPUs 4, 5, 6, 7

## Repository Structure

```
reptile-scaling-law/
├── banking77_data.py          # Banking77 dataset loader and task sampler
├── llm_model.py               # LLM + LoRA model wrapper
├── reptile_trainer.py         # Reptile training loop with multi-GPU
├── evaluation.py              # Evaluation utilities and baselines
├── experiment_runner.py       # Scaling law experiment orchestration
├── reptile_scaling_law_experiments.ipynb  # Main Jupyter notebook
└── README.md                  # This file
```

## Quick Start

### 1. Setup Environment

```bash
pip install transformers datasets peft torch accelerate bitsandbytes
pip install scipy matplotlib seaborn pandas tqdm tensorboardX
```

### 2. Run Experiments

Open the Jupyter notebook:

```bash
jupyter notebook reptile_scaling_law_experiments.ipynb
```

Or run from Python:

```python
from experiment_runner import ScalingLawExperimentRunner

runner = ScalingLawExperimentRunner(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks_list=[50, 100, 300, 1000],
    devices=["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
)

runner.run_all_experiments()
```

### 3. Run Pilot Experiment (Optional)

Test hyperparameters with a smaller experiment:

```python
from experiment_runner import run_pilot_experiment

run_pilot_experiment(
    n_tasks=200,
    num_meta_steps=3000,
    devices=["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
)
```

## Task Sampling Strategy

We implement **Option B** task sampling:
- **N_tasks** unique class combinations (5-way) are fixed at the start
- Within each task, support/query samples are randomly drawn each time
- This controls "task diversity" while maintaining episode variance

## Configuration

Key hyperparameters (from clarification):

```python
CONFIG = {
    # Task
    'n_way': 5,
    'k_support': 5,
    'k_query': 15,
    
    # LoRA
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    
    # Training
    'inner_lr': 5e-4,
    'meta_lr': 0.1,
    'k_inner': 5,
    'meta_batch_size': 4,
    
    # Experiment
    'num_meta_steps': 10000,
    'eval_interval': 500,
    'num_eval_tasks': 100,
    'num_meta_test_tasks': 200  # Fixed across all N_tasks
}
```

## Reptile Meta-Update

Meta-batch processing follows **Option A**:
1. Clone θ_old (meta-parameters)
2. For each task in meta-batch:
   - Reset to θ_old
   - Run K_inner step inner loop → φ_i
3. Average: Δ = mean(φ_i - θ_old)
4. Update: θ ← θ_old + ε · Δ

## Evaluation

### Meta-Test Protocol
- **K_eval_inner**: 5 steps (same as training)
- **Meta-test pool**: 200 tasks, fixed across all N_tasks experiments
- **Metrics**: Query loss (for scaling law) + accuracy

### Baselines
1. **Zero-shot**: Direct inference without adaptation
2. **No meta-learning**: Per-task fine-tuning from pre-trained init

## Multi-GPU Support

The implementation distributes tasks across GPUs 4, 5, 6, 7:
- Tasks are round-robin assigned to devices
- Each GPU processes its assigned tasks sequentially
- Results are aggregated for meta-update

## Expected Runtime

Approximate times per experiment (4 GPUs):
- N_tasks=50: ~2-3 hours
- N_tasks=100: ~3-4 hours
- N_tasks=300: ~5-6 hours
- N_tasks=1000: ~8-10 hours

Total for all experiments: **~20-25 hours**

## Outputs

```
experiments_scaling_law/
├── shared_meta_test_pool.json
├── baseline_results.json
├── ntasks_50_seed_100/
│   ├── config.json
│   ├── train_task_pool.json
│   ├── *_train_stats.csv
│   ├── *_eval_stats.csv
│   └── final_result.json
├── ntasks_100_seed_101/
│   └── ...
├── scaling_law_results.csv
├── all_results.json
├── power_law_fit.json
├── scaling_law_plot.png
└── summary_report.json
```

## Analysis

The experiment runner automatically:
1. Fits power law: L = A × N^(-β)
2. Calculates R² goodness-of-fit
3. Generates log-log plots
4. Compares with baselines

## Key Design Decisions

Based on clarification discussions:

1. **Task Sampling**: Fixed class combinations, variable support/query samples
2. **Meta-Update**: Batch averaging (Option A)
3. **Evaluation**: K_inner=5, fixed test pool
4. **LoRA Scope**: Backbone only, classifier head full fine-tune
5. **Scaling Metric**: Cross-entropy loss (primary)
6. **Seeds**: Separate for meta-test (0) and training (100+)
7. **Baselines**: Zero-shot + no meta-learning

## Citation

Based on:
- **Reptile**: Nichol et al., "On First-Order Meta-Learning Algorithms", 2018
- **Neural Scaling Laws**: Kaplan et al., "Scaling Laws for Neural Language Models", 2020
- **Banking77**: Casanueva et al., "Efficient Intent Detection with Dual Sentence Encoders", 2020

## Hardware Requirements

- **Minimum**: 4 GPUs with 10GB+ VRAM each
- **Recommended**: 4 GPUs with 16GB+ VRAM
- **Alternative**: Use `load_in_8bit=True` for lower memory

## Troubleshooting

### OOM Errors
```python
CONFIG['load_in_8bit'] = True
CONFIG['meta_batch_size'] = 2
CONFIG['inner_batch_size'] = 10
```

### Slow Training
```python
CONFIG['num_meta_steps'] = 5000  # Reduce steps
CONFIG['eval_interval'] = 1000   # Evaluate less frequently
```

### Multi-GPU Issues
Ensure GPUs 4-7 are available:
```python
import torch
for i in [4, 5, 6, 7]:
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

## License

MIT License

## Contact

For questions or issues, please open a GitHub issue.
