"""
Reptile Meta-Learning Training Loop for Banking77 + LLM

Implements efficient multi-GPU training with proper meta-updates.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

from llm_model import (
    LLMLoRAClassifier,
    clone_model_state,
    load_model_state,
    interpolate_states,
    average_states
)
from banking77_data import Banking77TaskSampler, MetaTaskBatchSampler


class ReptileLLMTrainer:
    """
    Reptile meta-learning trainer for LLM with multi-GPU support.
    """
    
    def __init__(
        self,
        model_wrapper: LLMLoRAClassifier,
        task_sampler: Banking77TaskSampler,
        train_task_pool: List[Tuple[int, ...]],
        test_task_pool: List[Tuple[int, ...]],
        inner_lr: float = 5e-4,
        meta_lr: float = 0.1,
        k_inner: int = 5,
        meta_batch_size: int = 4,
        inner_batch_size: int = 25,  # support+query per task
        weight_decay: float = 0.01,
        use_amp: bool = True,
        devices: List[str] = ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4", "cuda:5", "cuda:6", "cuda:7"]
    ):
        """
        Args:
            model_wrapper: LLMLoRAClassifier instance
            task_sampler: Banking77TaskSampler instance
            train_task_pool: List of task class combinations for training
            test_task_pool: List of task class combinations for testing
            inner_lr: Learning rate for inner loop optimization
            meta_lr: Meta-learning step size (epsilon in Reptile)
            k_inner: Number of inner loop gradient steps
            meta_batch_size: Number of tasks per meta-batch
            inner_batch_size: Batch size for inner loop (support+query)
            weight_decay: Weight decay for inner optimizer
            use_amp: Use automatic mixed precision
            devices: List of GPU devices to use
        """
        self.model_wrapper = model_wrapper
        self.task_sampler = task_sampler
        self.train_task_pool = train_task_pool
        self.test_task_pool = test_task_pool
        
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.k_inner = k_inner
        self.meta_batch_size = meta_batch_size
        self.inner_batch_size = inner_batch_size
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        self.devices = devices
        
        # Create batch samplers
        self.train_batch_sampler = MetaTaskBatchSampler(
            train_task_pool, task_sampler, split='train'
        )
        
        # Stats tracking
        self.train_stats = []
        self.eval_stats = []
        
        print(f"Initialized ReptileLLMTrainer:")
        print(f"  Train tasks: {len(train_task_pool)}")
        print(f"  Test tasks: {len(test_task_pool)}")
        print(f"  Inner LR: {inner_lr}, Meta LR: {meta_lr}, K_inner: {k_inner}")
        print(f"  Meta batch size: {meta_batch_size}")
        print(f"  Using devices: {devices}")
        print(f"  AMP: {use_amp}")
    
    def _inner_loop(
        self,
        support_encodings: Dict,
        support_labels: torch.Tensor,
        query_encodings: Dict,
        query_labels: torch.Tensor,
        device: str
    ) -> Tuple[Dict, float, float]:
        """
        Perform inner loop adaptation on one task.
        
        Args:
            support_encodings: Support set inputs
            support_labels: Support set labels
            query_encodings: Query set inputs  
            query_labels: Query set labels
            device: Device to run on
            
        Returns:
            adapted_state: State dict after K_inner steps
            final_train_loss: Loss on support after adaptation
            final_query_loss: Loss on query after adaptation
        """
        # Move data to device
        support_input_ids = support_encodings['input_ids'].to(device)
        support_attention_mask = support_encodings['attention_mask'].to(device)
        support_labels = support_labels.to(device)
        
        query_input_ids = query_encodings['input_ids'].to(device)
        query_attention_mask = query_encodings['attention_mask'].to(device)
        query_labels = query_labels.to(device)
        
        # Set model to train mode
        self.model_wrapper.train()
        
        # Create optimizer for inner loop
        optimizer = AdamW(
            self.model_wrapper.get_trainable_params(),
            lr=self.inner_lr,
            weight_decay=self.weight_decay
        )
        
        # Inner loop training
        final_train_loss = 0
        for step in range(self.k_inner):
            optimizer.zero_grad()
            
            # Process support in smaller batches to avoid OOM
            batch_size = min(self.inner_batch_size, len(support_input_ids))
            total_loss = 0
            num_batches = (len(support_input_ids) + batch_size - 1) // batch_size
            
            for i in range(0, len(support_input_ids), batch_size):
                end_idx = min(i + batch_size, len(support_input_ids))
                batch_input_ids = support_input_ids[i:end_idx]
                batch_attention_mask = support_attention_mask[i:end_idx]
                batch_labels = support_labels[i:end_idx]
                
                # Forward on support batch
                if self.use_amp:
                    with autocast():
                        outputs = self.model_wrapper.forward(
                            batch_input_ids,
                            batch_attention_mask,
                            batch_labels
                        )
                        loss = outputs.loss / num_batches
                else:
                    outputs = self.model_wrapper.forward(
                        batch_input_ids,
                        batch_attention_mask,
                        batch_labels
                    )
                    loss = outputs.loss / num_batches
                
                # Backward
                loss.backward()
                total_loss += loss.item() * num_batches
            
            # Gradient clipping to prevent NaN
            torch.nn.utils.clip_grad_norm_(
                self.model_wrapper.get_trainable_params(),
                max_norm=1.0
            )
            
            optimizer.step()
            final_train_loss = total_loss
        
        # Final evaluation on support (train loss) - use final_train_loss from last step
        # (already computed above)
        
        # Evaluation on query set
        with torch.no_grad():
            outputs = self.model_wrapper.forward(
                query_input_ids,
                query_attention_mask,
                query_labels
            )
            final_query_loss = outputs.loss.item()
        
        # Get adapted state
        adapted_state = self.model_wrapper.clone_state()
        
        return adapted_state, final_train_loss, final_query_loss
    
    def meta_train_step(self) -> Dict:
        """
        Perform one Reptile meta-training step.
        
        Returns:
            Dictionary with training statistics
        """
        # Clone current meta-parameters (theta_old)
        theta_old = clone_model_state(self.model_wrapper)
        
        # Sample meta-batch of tasks
        task_batch = self.train_batch_sampler.sample_batch(self.meta_batch_size)
        
        # Process all tasks on primary GPU (simplified single-GPU mode)
        adapted_states = []
        train_losses = []
        query_losses = []
        device = self.devices[0]  # Use only first GPU
        
        for task_data in task_batch:
            support_enc, query_enc, support_labels, query_labels = task_data
            
            # Reset to theta_old before each task
            load_model_state(self.model_wrapper, theta_old)
            
            # Run inner loop
            adapted_state, train_loss, query_loss = self._inner_loop(
                support_enc, support_labels,
                query_enc, query_labels,
                device
            )
            
            adapted_states.append(adapted_state)
            train_losses.append(train_loss)
            query_losses.append(query_loss)
        
        # Reptile meta-update: average adapted states
        avg_adapted_state = average_states(adapted_states)
        
        # Interpolate: theta_new = theta_old + meta_lr * (avg_phi - theta_old)
        theta_new = interpolate_states(theta_old, avg_adapted_state, self.meta_lr)
        
        # Update model with new meta-parameters
        load_model_state(self.model_wrapper, theta_new)
        
        # Return statistics
        return {
            'meta_train_loss': np.mean(train_losses),
            'meta_query_loss': np.mean(query_losses),
            'train_loss_std': np.std(train_losses),
            'query_loss_std': np.std(query_losses)
        }
    
    def _distribute_tasks(
        self,
        task_batch: List,
        devices: List[str]
    ) -> Dict[str, List]:
        """
        Distribute tasks across devices for parallel processing.
        
        For simplicity, we process sequentially but can extend to parallel.
        """
        # Simple round-robin distribution
        tasks_per_gpu = {device: [] for device in devices}
        
        for i, task in enumerate(task_batch):
            device = devices[i % len(devices)]
            tasks_per_gpu[device].append(task)
        
        return tasks_per_gpu
    
    def evaluate_meta_test(
        self,
        num_eval_tasks: int = None,
        k_eval_inner: int = None
    ) -> Dict:
        """
        Evaluate on meta-test tasks.
        
        Args:
            num_eval_tasks: Number of test tasks to evaluate (None = all)
            k_eval_inner: Number of inner steps for adaptation (None = use k_inner)
            
        Returns:
            Dictionary with evaluation statistics
        """
        if k_eval_inner is None:
            k_eval_inner = self.k_inner
        
        if num_eval_tasks is None:
            num_eval_tasks = len(self.test_task_pool)
        else:
            num_eval_tasks = min(num_eval_tasks, len(self.test_task_pool))
        
        # Save current meta-parameters
        theta_meta = clone_model_state(self.model_wrapper)
        
        query_losses = []
        query_accuracies = []
        
        # Evaluate on subset of test tasks
        eval_tasks = np.random.choice(len(self.test_task_pool), num_eval_tasks, replace=False)
        
        for task_idx in eval_tasks:
            task_classes = self.test_task_pool[task_idx]
            
            # Sample episode for this task
            support_enc, query_enc, support_labels, query_labels = \
                self.task_sampler.sample_episode(task_classes, split='test')
            
            # Reset to meta-parameters
            load_model_state(self.model_wrapper, theta_meta)
            
            # Adapt on support set
            device = self.devices[0]  # Use first device for eval
            support_input_ids = support_enc['input_ids'].to(device)
            support_attention_mask = support_enc['attention_mask'].to(device)
            support_labels_t = support_labels.to(device)
            
            # Inner loop adaptation
            optimizer = AdamW(
                self.model_wrapper.get_trainable_params(),
                lr=self.inner_lr,
                weight_decay=self.weight_decay
            )
            
            self.model_wrapper.train()
            for _ in range(k_eval_inner):
                optimizer.zero_grad()
                outputs = self.model_wrapper.forward(
                    support_input_ids,
                    support_attention_mask,
                    support_labels_t
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            # Evaluate on query set
            self.model_wrapper.eval()
            query_input_ids = query_enc['input_ids'].to(device)
            query_attention_mask = query_enc['attention_mask'].to(device)
            query_labels_t = query_labels.to(device)
            
            with torch.no_grad():
                outputs = self.model_wrapper.forward(
                    query_input_ids,
                    query_attention_mask,
                    query_labels_t
                )
                query_loss = outputs.loss.item()
                query_logits = outputs.logits
                query_preds = torch.argmax(query_logits, dim=-1)
                query_acc = (query_preds == query_labels_t).float().mean().item()
            
            query_losses.append(query_loss)
            query_accuracies.append(query_acc)
        
        # Restore meta-parameters
        load_model_state(self.model_wrapper, theta_meta)
        
        return {
            'meta_test_loss': np.mean(query_losses),
            'meta_test_accuracy': np.mean(query_accuracies),
            'meta_test_loss_std': np.std(query_losses),
            'meta_test_accuracy_std': np.std(query_accuracies)
        }
    
    def train(
        self,
        num_meta_steps: int,
        eval_interval: int = 500,
        num_eval_tasks: int = 100,
        save_dir: str = "./checkpoints",
        experiment_name: str = "reptile_experiment",
        start_step: int = 1
    ):
        """
        Main training loop.
        
        Args:
            num_meta_steps: Total number of meta-training steps
            eval_interval: Evaluate every N steps
            num_eval_tasks: Number of test tasks for evaluation
            save_dir: Directory to save checkpoints and logs
            experiment_name: Name for this experiment
            start_step: Step to start from (default: 1)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Starting Meta-Training: {experiment_name}")
        print(f"Total meta-steps: {num_meta_steps}")
        print(f"Start step: {start_step}")
        print(f"Eval interval: {eval_interval}")
        print(f"{'='*60}\n")
        
        for step in tqdm(range(start_step, num_meta_steps + 1), desc="Meta-training"):
            # Meta-training step
            train_stats = self.meta_train_step()
            
            # Check for NaN
            if np.isnan(train_stats['meta_train_loss']) or np.isnan(train_stats['meta_query_loss']):
                msg = f"\n[ERROR] NaN loss detected at step {step}! Train Loss: {train_stats['meta_train_loss']}, Query Loss: {train_stats['meta_query_loss']}"
                tqdm.write(msg)
                sys.stdout.flush()
                raise ValueError(msg)
            
            # Evaluation
            train_stats['step'] = step
            self.train_stats.append(train_stats)
            
            # Verbose logging every step
            log_msg = f"Step {step}: Train Loss={train_stats['meta_train_loss']:.4f}, Query Loss={train_stats['meta_query_loss']:.4f}"
            tqdm.write(log_msg)
            sys.stdout.flush()
            
            # Save checkpoint every 1000 steps
            if step % 1000 == 0:
                tqdm.write(f"\nSaving checkpoint at step {step}...")
                self._save_checkpoint(save_dir, experiment_name, step)
            
            # Evaluation
            if step % eval_interval == 0 or step == num_meta_steps:
                eval_stats = self.evaluate_meta_test(num_eval_tasks=num_eval_tasks)
                eval_stats['step'] = step
                self.eval_stats.append(eval_stats)
                
                # Print progress
                tqdm.write(f"\nStep {step}/{num_meta_steps}:")
                tqdm.write(f"  Train - Support Loss: {train_stats['meta_train_loss']:.4f}, "
                      f"Query Loss: {train_stats['meta_query_loss']:.4f}")
                tqdm.write(f"  Test  - Query Loss: {eval_stats['meta_test_loss']:.4f}, "
                      f"Accuracy: {eval_stats['meta_test_accuracy']:.4f}")
                sys.stdout.flush()
                
                # Save checkpoint (if not already saved by 1000 step rule)
                if step % 1000 != 0:
                    self._save_checkpoint(save_dir, experiment_name, step)
        
        print(f"\n{'='*60}")
        print("Meta-Training Complete!")
        print(f"{'='*60}\n")
        
        # Save final results
        self._save_results(save_dir, experiment_name)
    
    def _save_checkpoint(self, save_dir: str, experiment_name: str, step: int):
        """Save model checkpoint and stats."""
        checkpoint_path = os.path.join(save_dir, f"{experiment_name}_step_{step}.pt")
        
        checkpoint = {
            'step': step,
            'model_state': self.model_wrapper.get_state_dict(),
            'train_stats': self.train_stats,
            'eval_stats': self.eval_stats
        }
        
        torch.save(checkpoint, checkpoint_path)
    
    def _save_results(self, save_dir: str, experiment_name: str):
        """Save training and evaluation results."""
        # Save as CSV
        train_df = pd.DataFrame(self.train_stats)
        eval_df = pd.DataFrame(self.eval_stats)
        
        train_df.to_csv(os.path.join(save_dir, f"{experiment_name}_train_stats.csv"), index=False)
        eval_df.to_csv(os.path.join(save_dir, f"{experiment_name}_eval_stats.csv"), index=False)
        
        # Save final eval stats as JSON
        final_eval = self.eval_stats[-1] if self.eval_stats else {}
        with open(os.path.join(save_dir, f"{experiment_name}_final_results.json"), 'w') as f:
            json.dump(final_eval, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint and stats.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            step: The step number of the checkpoint
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.devices[0])
        
        # Load model state
        self.model_wrapper.set_state_dict(checkpoint['model_state'])
        
        # Load stats
        self.train_stats = checkpoint.get('train_stats', [])
        self.eval_stats = checkpoint.get('eval_stats', [])
        
        step = checkpoint.get('step', 0)
        print(f"Resumed from step {step}")
        
        return step
