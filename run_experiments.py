#!/usr/bin/env python3
"""
Simple script to run the full scaling law experiments.
Alternative to using the Jupyter notebook.
"""

import sys
sys.path.append('/root/ssd/reptile-scaling-law')

from experiment_runner import ScalingLawExperimentRunner

def main():
    print("="*80)
    print("REPTILE SCALING LAW EXPERIMENTS")
    print("="*80)
    
    # Configuration
    CONFIG = {
        'n_way': 5,
        'k_support': 5,
        'k_query': 15,
        'max_length': 64,
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'inner_lr': 5e-4,
        'meta_lr': 0.1,
        'k_inner': 5,
        'meta_batch_size': 4,
        'inner_batch_size': 25,
        'num_meta_steps': 10000,
        'eval_interval': 500,
        'num_eval_tasks': 100,
        'num_meta_test_tasks': 200,
        'load_in_8bit': False
    }
    
    N_TASKS_LIST = [50, 100, 300, 1000]
    
    # Initialize runner
    runner = ScalingLawExperimentRunner(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_tasks_list=N_TASKS_LIST,
        base_config=CONFIG,
        seed_meta_test=0,
        seed_train_base=100,
        save_root='./experiments_scaling_law',
        devices=["cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    )
    
    # Run all experiments
    print("\nStarting experiments...")
    print(f"N_tasks to test: {N_TASKS_LIST}")
    print(f"Estimated total time: ~20-25 hours\n")
    
    runner.run_all_experiments()
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print("\nResults saved to: ./experiments_scaling_law/")
    print("\nKey files:")
    print("  - scaling_law_results.csv")
    print("  - power_law_fit.json")
    print("  - scaling_law_plot.png")
    print("  - summary_report.json")

if __name__ == "__main__":
    main()
