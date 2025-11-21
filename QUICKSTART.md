# Reptile Scaling Law Experiments - Quick Start Guide

## 📋 Overview

This project investigates whether meta-learning performance follows a **power law** with respect to the number of training tasks:

$$L_{\text{meta}} \propto N_{\text{tasks}}^{-\beta}$$

Using:
- **Reptile** meta-learning on **Banking77** dataset
- **TinyLlama-1.1B** with **LoRA** fine-tuning
- **Multi-GPU** training (GPUs 4, 5, 6, 7)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
cd /root/ssd/reptile-scaling-law
pip install -r requirements.txt
```

### Step 2: Choose Your Method

**Option A: Jupyter Notebook** (Interactive, Recommended)
```bash
jupyter notebook reptile_scaling_law_experiments.ipynb
```

**Option B: Python Script** (Non-interactive)
```bash
python3 run_experiments.py
```

### Step 3: Wait & Analyze

- Experiments take **~20-25 hours** on 4 GPUs
- Results saved to `./experiments_scaling_law/`
- Power law fit, plots, and summary automatically generated

---

## 📁 Project Structure

```
reptile-scaling-law/
├── banking77_data.py          # Data loading & task sampling
├── llm_model.py               # LLM + LoRA wrapper
├── reptile_trainer.py         # Reptile training loop
├── evaluation.py              # Baselines & evaluation
├── experiment_runner.py       # Orchestrates N_tasks sweep
├── reptile_scaling_law_experiments.ipynb  # Main notebook
├── run_experiments.py         # CLI script
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
└── README.md                  # Full documentation
```

---

## 🎯 What Gets Tested

### N_tasks Values
- **50** tasks
- **100** tasks
- **300** tasks
- **1000** tasks

### For Each N_tasks
1. Meta-train for 10,000 steps
2. Evaluate every 500 steps
3. Final evaluation on 200 shared test tasks
4. Baseline comparisons (zero-shot, no meta-learning)

---

## 📊 Key Outputs

### Main Results
- `scaling_law_results.csv` - Summary table
- `power_law_fit.json` - Fitted β and R²
- `scaling_law_plot.png` - Log-log visualization
- `summary_report.json` - Complete experiment record

### Per-Experiment
Each `ntasks_XX_seed_YY/` folder contains:
- `config.json` - Experiment configuration
- `*_train_stats.csv` - Training metrics
- `*_eval_stats.csv` - Evaluation metrics
- `final_result.json` - Final results

---

## ⚙️ Configuration

Key hyperparameters (can modify in notebook/script):

```python
CONFIG = {
    'model_name': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'n_way': 5,           # 5-way classification
    'k_support': 5,       # 5-shot support
    'k_query': 15,        # 15 query samples
    'inner_lr': 5e-4,     # Inner loop learning rate
    'meta_lr': 0.1,       # Meta-step size
    'k_inner': 5,         # Inner adaptation steps
    'meta_batch_size': 4, # Tasks per meta-update
    'num_meta_steps': 10000,
    'devices': ['cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']
}
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)

```python
# Enable 8-bit quantization
CONFIG['load_in_8bit'] = True

# Reduce batch sizes
CONFIG['meta_batch_size'] = 2
CONFIG['inner_batch_size'] = 10
```

### Slow Training

```python
# Reduce training steps
CONFIG['num_meta_steps'] = 5000

# Evaluate less frequently
CONFIG['eval_interval'] = 1000
```

### GPU Not Available

Check GPUs:
```python
import torch
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
```

Adjust devices:
```python
CONFIG['devices'] = ['cuda:0', 'cuda:1']  # Use different GPUs
```

---

## 📈 Expected Results

### Power Law Fit
- β (scaling exponent): **0.2 - 0.4** (typical range)
- R²: **> 0.9** indicates strong power law

### Performance Improvement
- N=50 → N=1000: **5-15% accuracy gain**
- Diminishing returns as N increases

### Baseline Comparison
- Meta-learning should **outperform** no-meta baseline
- Larger gains with **fewer adaptation steps**

---

## 🔬 Research Questions Answered

1. ✅ **Does meta-learning follow power law?**
   - Check R² from `power_law_fit.json`

2. ✅ **What is the scaling exponent β?**
   - See β value in fit results

3. ✅ **How much gain from more tasks?**
   - Compare N=50 vs N=1000 performance

4. ✅ **Is meta-learning better than baselines?**
   - Check `baseline_comparison.png`

---

## 📝 Next Steps

### After Experiments Complete

1. **Analyze Results**
   ```python
   # In notebook cell 4
   results_df = pd.read_csv('./experiments_scaling_law/scaling_law_results.csv')
   ```

2. **Generate Custom Plots**
   ```python
   # Plot loss vs N_tasks with confidence intervals
   plt.errorbar(results_df['n_tasks'], results_df['meta_test_loss'], 
                yerr=results_df['meta_test_loss_std'])
   ```

3. **Try Different Models**
   ```python
   CONFIG['model_name'] = 'google/gemma-2b'  # Use Gemma instead
   ```

4. **Extend to More N_tasks**
   ```python
   N_TASKS_LIST = [50, 100, 300, 1000, 3000]  # Add more points
   ```

---

## 🤝 Contributing

To extend this work:

1. **Add new baselines** in `evaluation.py`
2. **Try different meta-algorithms** (modify `reptile_trainer.py`)
3. **Test other datasets** (extend `banking77_data.py`)
4. **Implement MAML** (add second-order gradients)

---

## 📚 References

- **Reptile**: Nichol et al. (2018) - *On First-Order Meta-Learning Algorithms*
- **Scaling Laws**: Kaplan et al. (2020) - *Scaling Laws for Neural Language Models*
- **LoRA**: Hu et al. (2021) - *LoRA: Low-Rank Adaptation of Large Language Models*
- **Banking77**: Casanueva et al. (2020) - *Efficient Intent Detection*

---

## ✅ Checklist

Before running experiments:

- [ ] GPUs 4, 5, 6, 7 available
- [ ] At least 40GB free disk space
- [ ] Python 3.8+ with CUDA support
- [ ] ~24 hours of uninterrupted runtime
- [ ] Dependencies installed (`requirements.txt`)

---

## 💡 Tips

- **Monitor Progress**: Check `experiments_scaling_law/*/eval_stats.csv` during training
- **Early Stopping**: If one experiment fails, others continue
- **Resume**: Can restart individual experiments by modifying `N_TASKS_LIST`
- **Compare Runs**: Use different seeds to measure variance

---

## 🆘 Support

Issues? Check:
1. GPU memory (`nvidia-smi`)
2. Disk space (`df -h`)
3. Error logs in experiment folders
4. GitHub Issues (if public repo)

**Happy experimenting! 🚀**
