"""
Evaluation utilities and baseline experiments.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm

from llm_model import LLMLoRAClassifier, clone_model_state, load_model_state
from banking77_data import Banking77TaskSampler
from torch.optim import AdamW


def evaluate_baseline_no_meta(
    model_wrapper: LLMLoRAClassifier,
    task_sampler: Banking77TaskSampler,
    test_task_pool: List[Tuple[int, ...]],
    inner_lr: float = 5e-4,
    k_inner: int = 5,
    num_eval_tasks: int = 100,
    device: str = "cuda:0"
) -> Dict:
    """
    Baseline: No meta-learning, just per-task fine-tuning from random init.
    
    For each test task:
    1. Reset model to initial random state (or pre-trained but no meta-training)
    2. Fine-tune on support set for K steps
    3. Evaluate on query set
    
    Args:
        model_wrapper: LLMLoRAClassifier instance (will use its initial state)
        task_sampler: Banking77TaskSampler
        test_task_pool: List of test tasks
        inner_lr: Learning rate for fine-tuning
        k_inner: Number of fine-tuning steps
        num_eval_tasks: Number of tasks to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with statistics
    """
    # Get initial model state (before any meta-training)
    initial_state = clone_model_state(model_wrapper)
    
    query_losses = []
    query_accuracies = []
    
    num_eval_tasks = min(num_eval_tasks, len(test_task_pool))
    eval_task_indices = np.random.choice(len(test_task_pool), num_eval_tasks, replace=False)
    
    for task_idx in tqdm(eval_task_indices, desc="Baseline Eval"):
        task_classes = test_task_pool[task_idx]
        
        # Sample episode
        support_enc, query_enc, support_labels, query_labels = \
            task_sampler.sample_episode(task_classes, split='test')
        
        # Reset to initial state
        load_model_state(model_wrapper, initial_state)
        
        # Fine-tune on support
        support_input_ids = support_enc['input_ids'].to(device)
        support_attention_mask = support_enc['attention_mask'].to(device)
        support_labels_t = support_labels.to(device)
        
        optimizer = AdamW(
            model_wrapper.get_trainable_params(),
            lr=inner_lr,
            weight_decay=0.01
        )
        
        model_wrapper.train()
        for _ in range(k_inner):
            optimizer.zero_grad()
            outputs = model_wrapper.forward(
                support_input_ids,
                support_attention_mask,
                support_labels_t
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        # Evaluate on query
        model_wrapper.eval()
        query_input_ids = query_enc['input_ids'].to(device)
        query_attention_mask = query_enc['attention_mask'].to(device)
        query_labels_t = query_labels.to(device)
        
        with torch.no_grad():
            outputs = model_wrapper.forward(
                query_input_ids,
                query_attention_mask,
                query_labels_t
            )
            query_loss = outputs.loss.item()
            query_preds = torch.argmax(outputs.logits, dim=-1)
            query_acc = (query_preds == query_labels_t).float().mean().item()
        
        query_losses.append(query_loss)
        query_accuracies.append(query_acc)
    
    # Restore initial state
    load_model_state(model_wrapper, initial_state)
    
    return {
        'baseline_no_meta_loss': np.mean(query_losses),
        'baseline_no_meta_accuracy': np.mean(query_accuracies),
        'baseline_no_meta_loss_std': np.std(query_losses),
        'baseline_no_meta_accuracy_std': np.std(query_accuracies)
    }


def evaluate_zero_shot(
    model_wrapper: LLMLoRAClassifier,
    task_sampler: Banking77TaskSampler,
    test_task_pool: List[Tuple[int, ...]],
    num_eval_tasks: int = 100,
    device: str = "cuda:0"
) -> Dict:
    """
    Zero-shot evaluation: no adaptation, just evaluate with initial model.
    
    Args:
        model_wrapper: LLMLoRAClassifier instance
        task_sampler: Banking77TaskSampler
        test_task_pool: List of test tasks
        num_eval_tasks: Number of tasks to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with statistics
    """
    model_wrapper.eval()
    
    query_losses = []
    query_accuracies = []
    
    num_eval_tasks = min(num_eval_tasks, len(test_task_pool))
    eval_task_indices = np.random.choice(len(test_task_pool), num_eval_tasks, replace=False)
    
    for task_idx in tqdm(eval_task_indices, desc="Zero-shot Eval"):
        task_classes = test_task_pool[task_idx]
        
        # Sample episode (we only need query for zero-shot)
        _, query_enc, _, query_labels = \
            task_sampler.sample_episode(task_classes, split='test')
        
        query_input_ids = query_enc['input_ids'].to(device)
        query_attention_mask = query_enc['attention_mask'].to(device)
        query_labels_t = query_labels.to(device)
        
        with torch.no_grad():
            outputs = model_wrapper.forward(
                query_input_ids,
                query_attention_mask,
                query_labels_t
            )
            query_loss = outputs.loss.item()
            query_preds = torch.argmax(outputs.logits, dim=-1)
            query_acc = (query_preds == query_labels_t).float().mean().item()
        
        query_losses.append(query_loss)
        query_accuracies.append(query_acc)
    
    return {
        'zero_shot_loss': np.mean(query_losses),
        'zero_shot_accuracy': np.mean(query_accuracies),
        'zero_shot_loss_std': np.std(query_losses),
        'zero_shot_accuracy_std': np.std(query_accuracies)
    }


def compare_adaptation_steps(
    model_wrapper: LLMLoRAClassifier,
    task_sampler: Banking77TaskSampler,
    test_task_pool: List[Tuple[int, ...]],
    k_steps_list: List[int] = [1, 3, 5, 10],
    inner_lr: float = 5e-4,
    num_eval_tasks: int = 50,
    device: str = "cuda:0"
) -> Dict:
    """
    Compare meta-learning performance with different adaptation steps.
    
    Args:
        model_wrapper: Meta-trained LLMLoRAClassifier
        task_sampler: Banking77TaskSampler
        test_task_pool: List of test tasks
        k_steps_list: List of different K values to try
        inner_lr: Learning rate for adaptation
        num_eval_tasks: Number of tasks to evaluate
        device: Device to run on
        
    Returns:
        Dictionary with results for each K
    """
    theta_meta = clone_model_state(model_wrapper)
    
    results = {}
    
    for k_steps in k_steps_list:
        query_losses = []
        query_accuracies = []
        
        num_eval_tasks_actual = min(num_eval_tasks, len(test_task_pool))
        eval_task_indices = np.random.choice(len(test_task_pool), num_eval_tasks_actual, replace=False)
        
        for task_idx in tqdm(eval_task_indices, desc=f"K={k_steps}"):
            task_classes = test_task_pool[task_idx]
            
            # Sample episode
            support_enc, query_enc, support_labels, query_labels = \
                task_sampler.sample_episode(task_classes, split='test')
            
            # Reset to meta-parameters
            load_model_state(model_wrapper, theta_meta)
            
            # Adapt on support
            support_input_ids = support_enc['input_ids'].to(device)
            support_attention_mask = support_enc['attention_mask'].to(device)
            support_labels_t = support_labels.to(device)
            
            optimizer = AdamW(
                model_wrapper.get_trainable_params(),
                lr=inner_lr,
                weight_decay=0.01
            )
            
            model_wrapper.train()
            for _ in range(k_steps):
                optimizer.zero_grad()
                outputs = model_wrapper.forward(
                    support_input_ids,
                    support_attention_mask,
                    support_labels_t
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            # Evaluate on query
            model_wrapper.eval()
            query_input_ids = query_enc['input_ids'].to(device)
            query_attention_mask = query_enc['attention_mask'].to(device)
            query_labels_t = query_labels.to(device)
            
            with torch.no_grad():
                outputs = model_wrapper.forward(
                    query_input_ids,
                    query_attention_mask,
                    query_labels_t
                )
                query_loss = outputs.loss.item()
                query_preds = torch.argmax(outputs.logits, dim=-1)
                query_acc = (query_preds == query_labels_t).float().mean().item()
            
            query_losses.append(query_loss)
            query_accuracies.append(query_acc)
        
        results[f'k_{k_steps}'] = {
            'loss': np.mean(query_losses),
            'accuracy': np.mean(query_accuracies),
            'loss_std': np.std(query_losses),
            'accuracy_std': np.std(query_accuracies)
        }
    
    # Restore meta-parameters
    load_model_state(model_wrapper, theta_meta)
    
    return results
