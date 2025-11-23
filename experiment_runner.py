"""
Experiment Runner for Scaling Law Investigation

Orchestrates multiple experiments with different N_tasks values.
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns

from llm_model import LLMLoRAClassifier
from banking77_data import Banking77TaskSampler
from reptile_trainer import ReptileLLMTrainer
from evaluation import evaluate_baseline_no_meta, evaluate_zero_shot, compare_adaptation_steps


class ScalingLawExperimentRunner:
    """
    Manages multiple Reptile experiments with different N_tasks values.
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        n_tasks_list: List[int] = [50, 100, 300, 1000],
        base_config: Dict = None,
        seed_meta_test: int = 0,
        seed_train_base: int = 100,
        save_root: str = "./experiments",
        devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    ):
        """
        Args:
            model_name: HuggingFace model name
            n_tasks_list: List of N_tasks values to experiment with
            base_config: Base configuration dict
            seed_meta_test: Seed for generating meta-test pool (fixed across all experiments)
            seed_train_base: Base seed for training (will add offset for each N_tasks)
            save_root: Root directory for saving experiments
            devices: GPU devices to use
        """
        self.model_name = model_name
        self.n_tasks_list = sorted(n_tasks_list)
        self.seed_meta_test = seed_meta_test
        self.seed_train_base = seed_train_base
        self.save_root = save_root
        self.devices = devices
        
        # Default configuration optimized for speed (5-8x faster)
        self.base_config = {
            'n_way': 5,
            'k_support': 5,
            'k_query': 15,
            'max_length': 64,
            'lora_r': 8,
            'lora_alpha': 16,
            'lora_dropout': 0.05,
            'inner_lr': 1e-4,  # 5e-4 → 1e-4 (more stable)
            'meta_lr': 0.05,  # 0.1 → 0.05 (prevent NaN)
            'k_inner': 3,  # 5 → 3 (40% faster)
            'meta_batch_size': 2,  # 1 → 2 (2x throughput)
            'inner_batch_size': 16,  # 8 → 16 (better GPU util)
            'num_meta_steps': 10000,  # 2000 → 10000 (as requested)
            'eval_interval': 500,
            'num_eval_tasks': 50,  # 100 → 50 (2x faster)
            'num_meta_test_tasks': 100,  # 200 → 100 (2x faster)
            'load_in_8bit': False,  # Disable 8-bit (causes NaN, use gradient checkpointing instead)
            'gradient_checkpointing': True # Default to True for safety
        }
        
        if base_config:
            self.base_config.update(base_config)
        
        os.makedirs(save_root, exist_ok=True)
        
        # Results tracking
        self.results = []
        
    def run_all_experiments(self):
        """
        Run experiments for all N_tasks values sequentially.
        """
        print(f"\n{'='*80}")
        print(f"SCALING LAW EXPERIMENT SUITE")
        print(f"{'='*80}")
        print(f"Model: {self.model_name}")
        print(f"N_tasks values: {self.n_tasks_list}")
        print(f"Meta-test seed: {self.seed_meta_test}")
        print(f"Devices: {self.devices}")
        print(f"{'='*80}\n")
        
        # Create shared meta-test pool (same for all experiments)
        print("Creating shared meta-test task pool...")
        model_wrapper = self._initialize_model()
        task_sampler = self._initialize_task_sampler(model_wrapper.tokenizer)
        
        shared_test_pool = task_sampler.generate_task_pool(
            n_tasks=self.base_config['num_meta_test_tasks'],
            split='test',
            seed=self.seed_meta_test
        )
        print(f"Meta-test pool: {len(shared_test_pool)} tasks\n")
        
        # Save shared test pool
        with open(os.path.join(self.save_root, 'shared_meta_test_pool.json'), 'w') as f:
            json.dump([list(task) for task in shared_test_pool], f)
        
        # Run baseline evaluations (before any meta-training)
        print("\n" + "="*80)
        print("BASELINE EVALUATIONS")
        print("="*80 + "\n")
        
        baseline_results = self._run_baseline_evaluations(
            model_wrapper, task_sampler, shared_test_pool
        )
        
        # Save baseline results
        with open(os.path.join(self.save_root, 'baseline_results.json'), 'w') as f:
            json.dump(baseline_results, f, indent=2)
        
        # Run meta-learning experiments for each N_tasks
        for idx, n_tasks in enumerate(self.n_tasks_list):
            seed_train = self.seed_train_base + idx
            
            print(f"\n{'='*80}")
            print(f"EXPERIMENT {idx+1}/{len(self.n_tasks_list)}: N_tasks = {n_tasks}")
            print(f"{'='*80}\n")
            
            result = self.run_single_experiment(
                n_tasks=n_tasks,
                seed_train=seed_train,
                shared_test_pool=shared_test_pool
            )
            
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
        
        print(f"\n{'='*80}")
        print("ALL EXPERIMENTS COMPLETE!")
        print(f"{'='*80}\n")
        
        # Generate final analysis
        self._generate_analysis()
        
    def run_single_experiment(
        self,
        n_tasks: int,
        seed_train: int,
        shared_test_pool: List,
        resume_checkpoint: str = None
    ) -> Dict:
        """
        Run one experiment with given N_tasks.
        
        Args:
            n_tasks: Number of training tasks
            seed_train: Random seed for this experiment
            shared_test_pool: Shared meta-test task pool
            resume_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Dictionary with experiment results
        """
        experiment_name = f"ntasks_{n_tasks}_seed_{seed_train}"
        save_dir = os.path.join(self.save_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Set random seeds
        torch.manual_seed(seed_train)
        np.random.seed(seed_train)
        
        # Initialize model and data
        print(f"Initializing model and data for N_tasks={n_tasks}...")
        model_wrapper = self._initialize_model()
        task_sampler = self._initialize_task_sampler(model_wrapper.tokenizer)
        
        # Generate training task pool
        print(f"Generating {n_tasks} training tasks...")
        train_task_pool = task_sampler.generate_task_pool(
            n_tasks=n_tasks,
            split='train',
            seed=seed_train
        )
        
        # Save train task pool
        with open(os.path.join(save_dir, 'train_task_pool.json'), 'w') as f:
            json.dump([list(task) for task in train_task_pool], f)
        
        # Save config
        config = self.base_config.copy()
        config.update({
            'n_tasks': n_tasks,
            'seed_train': seed_train,
            'seed_meta_test': self.seed_meta_test,
            'model_name': self.model_name,
            'experiment_name': experiment_name
        })
        
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Initialize trainer
        trainer = ReptileLLMTrainer(
            model_wrapper=model_wrapper,
            task_sampler=task_sampler,
            train_task_pool=train_task_pool,
            test_task_pool=shared_test_pool,
            inner_lr=config['inner_lr'],
            meta_lr=config['meta_lr'],
            k_inner=config['k_inner'],
            meta_batch_size=config['meta_batch_size'],
            inner_batch_size=config['inner_batch_size'],
            use_amp=config.get('use_amp', True),
            devices=self.devices
        )
        
        # Check for existing checkpoints
        start_step = 1
        if os.path.exists(save_dir):
            checkpoint_files = [f for f in os.listdir(save_dir) if f.startswith(f"{experiment_name}_step_") and f.endswith(".pt")]
            if checkpoint_files:
                # Sort by step number
                checkpoint_files.sort(key=lambda x: int(x.split('_step_')[1].split('.pt')[0]))
                latest_checkpoint = checkpoint_files[-1]
                checkpoint_path = os.path.join(save_dir, latest_checkpoint)
                print(f"Found checkpoint: {latest_checkpoint}")
                start_step = trainer.load_checkpoint(checkpoint_path) + 1
        
        # Train
        print(f"\nStarting meta-training...")
        trainer.train(
            num_meta_steps=config['num_meta_steps'],
            eval_interval=config['eval_interval'],
            num_eval_tasks=config['num_eval_tasks'],
            save_dir=save_dir,
            experiment_name=experiment_name,
            start_step=start_step
        )
        
        # Final evaluation
        print(f"\nFinal evaluation...")
        final_eval = trainer.evaluate_meta_test(num_eval_tasks=len(shared_test_pool))
        
        # Additional analysis: different K steps
        print(f"\nEvaluating different adaptation steps...")
        k_steps_results = compare_adaptation_steps(
            model_wrapper=model_wrapper,
            task_sampler=task_sampler,
            test_task_pool=shared_test_pool,
            k_steps_list=[1, 3, 5, 10, 20],
            inner_lr=config['inner_lr'],
            num_eval_tasks=min(50, len(shared_test_pool)),
            device=self.devices[0]
        )
        
        # Compile results
        result = {
            'n_tasks': n_tasks,
            'seed_train': seed_train,
            'final_meta_test_loss': final_eval['meta_test_loss'],
            'final_meta_test_accuracy': final_eval['meta_test_accuracy'],
            'final_meta_test_loss_std': final_eval['meta_test_loss_std'],
            'final_meta_test_accuracy_std': final_eval['meta_test_accuracy_std'],
            'k_steps_analysis': k_steps_results,
            'save_dir': save_dir
        }
        
        # Save final result
        with open(os.path.join(save_dir, 'final_result.json'), 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
    
    def _initialize_model(self) -> LLMLoRAClassifier:
        """Initialize LLM with LoRA."""
        return LLMLoRAClassifier(
            model_name=self.model_name,
            num_labels=self.base_config['n_way'],
            lora_r=self.base_config['lora_r'],
            lora_alpha=self.base_config['lora_alpha'],
            lora_dropout=self.base_config['lora_dropout'],
            device=self.devices[0],
            load_in_8bit=self.base_config['load_in_8bit'],
            gradient_checkpointing=self.base_config.get('gradient_checkpointing', True)
        )
    
    def _initialize_task_sampler(self, tokenizer) -> Banking77TaskSampler:
        """Initialize task sampler."""
        return Banking77TaskSampler(
            tokenizer=tokenizer,
            max_length=self.base_config['max_length'],
            n_way=self.base_config['n_way'],
            k_support=self.base_config['k_support'],
            k_query=self.base_config['k_query']
        )
    
    def _run_baseline_evaluations(
        self,
        model_wrapper: LLMLoRAClassifier,
        task_sampler: Banking77TaskSampler,
        test_pool: List
    ) -> Dict:
        """Run baseline evaluations."""
        results = {}
        
        # Zero-shot
        print("Evaluating zero-shot baseline...")
        zero_shot_results = evaluate_zero_shot(
            model_wrapper=model_wrapper,
            task_sampler=task_sampler,
            test_task_pool=test_pool,
            num_eval_tasks=min(100, len(test_pool)),
            device=self.devices[0]
        )
        results['zero_shot'] = zero_shot_results
        print(f"Zero-shot: Loss={zero_shot_results['zero_shot_loss']:.4f}, "
              f"Acc={zero_shot_results['zero_shot_accuracy']:.4f}")
        
        # No meta-learning baseline
        print("\nEvaluating no-meta-learning baseline (per-task FT)...")
        no_meta_results = evaluate_baseline_no_meta(
            model_wrapper=model_wrapper,
            task_sampler=task_sampler,
            test_task_pool=test_pool,
            inner_lr=self.base_config['inner_lr'],
            k_inner=self.base_config['k_inner'],
            num_eval_tasks=min(100, len(test_pool)),
            device=self.devices[0]
        )
        results['no_meta_learning'] = no_meta_results
        print(f"No meta-learning: Loss={no_meta_results['baseline_no_meta_loss']:.4f}, "
              f"Acc={no_meta_results['baseline_no_meta_accuracy']:.4f}")
        
        return results
    
    def _save_results(self):
        """Save current results to CSV and JSON."""
        # Create summary DataFrame
        summary_data = []
        for result in self.results:
            summary_data.append({
                'n_tasks': result['n_tasks'],
                'seed_train': result['seed_train'],
                'meta_test_loss': result['final_meta_test_loss'],
                'meta_test_accuracy': result['final_meta_test_accuracy'],
                'meta_test_loss_std': result['final_meta_test_loss_std'],
                'meta_test_accuracy_std': result['final_meta_test_accuracy_std']
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.save_root, 'scaling_law_results.csv'), index=False)
        
        # Save full results as JSON
        with open(os.path.join(self.save_root, 'all_results.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def _generate_analysis(self):
        """Generate analysis plots and power law fitting."""
        print("\n" + "="*80)
        print("GENERATING ANALYSIS")
        print("="*80 + "\n")
        
        # Load results
        df = pd.DataFrame([
            {
                'n_tasks': r['n_tasks'],
                'loss': r['final_meta_test_loss'],
                'accuracy': r['final_meta_test_accuracy'],
                'loss_std': r['final_meta_test_loss_std'],
                'accuracy_std': r['final_meta_test_accuracy_std']
            }
            for r in self.results
        ])
        
        # Plot scaling law
        self._plot_scaling_law(df)
        
        # Fit power law
        self._fit_power_law(df)
    
    def _plot_scaling_law(self, df: pd.DataFrame):
        """Generate scaling law plots."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Log-log plot for loss
        ax = axes[0]
        ax.errorbar(df['n_tasks'], df['loss'], yerr=df['loss_std'],
                    marker='o', capsize=5, capthick=2, label='Meta-test Loss')
        ax.set_xlabel('N_tasks', fontsize=12)
        ax.set_ylabel('Meta-test Loss', fontsize=12)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Scaling Law: Loss vs N_tasks (log-log)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Log-linear plot for accuracy
        ax = axes[1]
        ax.errorbar(df['n_tasks'], df['accuracy'], yerr=df['accuracy_std'],
                    marker='s', capsize=5, capthick=2, label='Meta-test Accuracy', color='green')
        ax.set_xlabel('N_tasks', fontsize=12)
        ax.set_ylabel('Meta-test Accuracy', fontsize=12)
        ax.set_xscale('log')
        ax.set_title('Scaling Law: Accuracy vs N_tasks (log-linear)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_root, 'scaling_law_plot.png'), dpi=300)
        print(f"Saved: {os.path.join(self.save_root, 'scaling_law_plot.png')}")
    
    def _fit_power_law(self, df: pd.DataFrame):
        """Fit power law: L = A * N^(-beta)"""
        from scipy.optimize import curve_fit
        
        # Log-log linear fit
        log_n = np.log(df['n_tasks'].values)
        log_loss = np.log(df['loss'].values)
        
        # Linear regression in log space: log(L) = log(A) - beta * log(N)
        coeffs = np.polyfit(log_n, log_loss, 1)
        beta = -coeffs[0]
        log_A = coeffs[1]
        A = np.exp(log_A)
        
        # Calculate R²
        log_loss_pred = np.polyval(coeffs, log_n)
        ss_res = np.sum((log_loss - log_loss_pred) ** 2)
        ss_tot = np.sum((log_loss - np.mean(log_loss)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        print(f"\nPower Law Fit: L = A * N^(-β)")
        print(f"  A = {A:.6f}")
        print(f"  β = {beta:.6f}")
        print(f"  R² = {r_squared:.6f}")
        
        # Save fit results
        fit_results = {
            'A': float(A),
            'beta': float(beta),
            'R_squared': float(r_squared),
            'formula': f"Loss = {A:.6f} * N_tasks^(-{beta:.6f})"
        }
        
        with open(os.path.join(self.save_root, 'power_law_fit.json'), 'w') as f:
            json.dump(fit_results, f, indent=2)
        
        print(f"\nSaved power law fit to: {os.path.join(self.save_root, 'power_law_fit.json')}")


def run_pilot_experiment(
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    n_tasks: int = 200,
    num_meta_steps: int = 3000,
    save_dir: str = "./pilot_experiment",
    devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
):
    """
    Run a single pilot experiment for hyperparameter tuning.
    
    Args:
        model_name: Model to use
        n_tasks: Number of training tasks
        num_meta_steps: Number of meta-training steps
        save_dir: Where to save results
        devices: GPU devices
    """
    print(f"\n{'='*80}")
    print(f"PILOT EXPERIMENT")
    print(f"{'='*80}")
    print(f"Model: {model_name}")
    print(f"N_tasks: {n_tasks}")
    print(f"Meta-steps: {num_meta_steps}")
    print(f"{'='*80}\n")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize
    model_wrapper = LLMLoRAClassifier(
        model_name=model_name,
        num_labels=5,
        device=devices[0]
    )
    
    task_sampler = Banking77TaskSampler(
        tokenizer=model_wrapper.tokenizer,
        max_length=64,
        n_way=5,
        k_support=5,
        k_query=15
    )
    
    # Generate task pools
    train_pool = task_sampler.generate_task_pool(n_tasks, split='train', seed=42)
    test_pool = task_sampler.generate_task_pool(200, split='test', seed=0)
    
    # Train
    trainer = ReptileLLMTrainer(
        model_wrapper=model_wrapper,
        task_sampler=task_sampler,
        train_task_pool=train_pool,
        test_task_pool=test_pool,
        devices=devices
    )
    
    trainer.train(
        num_meta_steps=num_meta_steps,
        eval_interval=250,
        save_dir=save_dir,
        experiment_name="pilot"
    )
    
    print(f"\nPilot experiment complete! Results in: {save_dir}")
