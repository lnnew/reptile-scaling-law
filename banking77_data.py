"""
Banking77 Dataset Module for Meta-Learning with Reptile

Implements efficient task sampling and episode generation for 5-way K-shot learning.
"""

import random
import numpy as np
import torch
from datasets import load_dataset
from collections import defaultdict
from typing import List, Tuple, Dict
import copy


class Banking77TaskSampler:
    """
    Handles Banking77 dataset loading, preprocessing, and meta-learning task generation.
    
    Implements Option B task sampling:
    - N_tasks unique class combinations are fixed
    - Within each task, support/query samples are randomly drawn each time
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 64,
        n_way: int = 5,
        k_support: int = 5,
        k_query: int = 15,
        seed: int = 42
    ):
        """
        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            n_way: Number of classes per task
            k_support: Number of support samples per class
            k_query: Number of query samples per class
            seed: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_way = n_way
        self.k_support = k_support
        self.k_query = k_query
        self.seed = seed
        
        # Load and preprocess dataset
        print("Loading Banking77 dataset...")
        dataset = load_dataset("banking77")
        self.train_data = dataset['train']
        self.test_data = dataset['test']
        
        # Get total number of classes
        self.num_total_classes = 77
        
        # Preprocess and index by class
        print("Preprocessing and indexing by class...")
        self.train_class_indices = self._index_by_class(self.train_data)
        self.test_class_indices = self._index_by_class(self.test_data)
        
        print(f"Loaded {len(self.train_data)} train samples, {len(self.test_data)} test samples")
        print(f"Classes: {self.num_total_classes}")
        
    def _index_by_class(self, dataset) -> Dict[int, List[int]]:
        """Create mapping from class label to list of indices."""
        class_indices = defaultdict(list)
        for idx, example in enumerate(dataset):
            class_indices[example['label']].append(idx)
        return dict(class_indices)
    
    def _tokenize_examples(self, dataset, indices: List[int]) -> Dict:
        """Tokenize a list of examples by indices."""
        texts = [dataset[idx]['text'] for idx in indices]
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return encodings
    
    def generate_task_pool(self, n_tasks: int, split: str = 'train', seed: int = None) -> List[Tuple[int, ...]]:
        """
        Generate N_tasks unique class combinations (5-way).
        
        Args:
            n_tasks: Number of unique task combinations to generate
            split: 'train' or 'test'
            seed: Random seed for task generation
            
        Returns:
            List of tuples, each containing n_way class IDs
        """
        if seed is None:
            seed = self.seed
        
        rng = random.Random(seed)
        all_classes = list(range(self.num_total_classes))
        
        task_pool = []
        max_possible = self._n_choose_k(self.num_total_classes, self.n_way)
        
        if n_tasks > max_possible:
            print(f"Warning: n_tasks={n_tasks} exceeds total combinations ({max_possible})")
            n_tasks = max_possible
        
        # Generate unique combinations
        seen = set()
        attempts = 0
        max_attempts = n_tasks * 100
        
        while len(task_pool) < n_tasks and attempts < max_attempts:
            classes = tuple(sorted(rng.sample(all_classes, self.n_way)))
            if classes not in seen:
                task_pool.append(classes)
                seen.add(classes)
            attempts += 1
        
        if len(task_pool) < n_tasks:
            print(f"Warning: Only generated {len(task_pool)} unique tasks (requested {n_tasks})")
        
        return task_pool
    
    def _n_choose_k(self, n: int, k: int) -> int:
        """Calculate binomial coefficient."""
        from math import factorial
        return factorial(n) // (factorial(k) * factorial(n - k))
    
    def sample_episode(
        self,
        task_classes: Tuple[int, ...],
        split: str = 'train',
        support_only: bool = False
    ) -> Tuple[Dict, Dict, torch.Tensor, torch.Tensor]:
        """
        Sample one episode (support + query) from given class combination.
        
        Args:
            task_classes: Tuple of class IDs for this task
            split: 'train' or 'test'
            support_only: If True, only return support set (for evaluation adaptation)
            
        Returns:
            support_encodings: Dict with input_ids, attention_mask
            query_encodings: Dict with input_ids, attention_mask (None if support_only)
            support_labels: Tensor of remapped labels (0 to n_way-1)
            query_labels: Tensor of remapped labels (None if support_only)
        """
        dataset = self.train_data if split == 'train' else self.test_data
        class_indices = self.train_class_indices if split == 'train' else self.test_class_indices
        
        support_indices = []
        query_indices = []
        support_labels = []
        query_labels = []
        
        # For each class in the task
        for new_label, original_class in enumerate(task_classes):
            available_indices = class_indices[original_class]
            
            # Check if enough samples
            required = self.k_support if support_only else (self.k_support + self.k_query)
            if len(available_indices) < required:
                # Sample with replacement if not enough
                sampled = random.choices(available_indices, k=required)
            else:
                # Sample without replacement
                sampled = random.sample(available_indices, required)
            
            # Split into support and query
            support_indices.extend(sampled[:self.k_support])
            support_labels.extend([new_label] * self.k_support)
            
            if not support_only:
                query_indices.extend(sampled[self.k_support:self.k_support + self.k_query])
                query_labels.extend([new_label] * self.k_query)
        
        # Shuffle support and query separately
        support_perm = list(range(len(support_indices)))
        random.shuffle(support_perm)
        support_indices = [support_indices[i] for i in support_perm]
        support_labels = [support_labels[i] for i in support_perm]
        
        # Tokenize
        support_encodings = self._tokenize_examples(dataset, support_indices)
        support_labels_tensor = torch.tensor(support_labels, dtype=torch.long)
        
        if support_only:
            return support_encodings, None, support_labels_tensor, None
        
        query_perm = list(range(len(query_indices)))
        random.shuffle(query_perm)
        query_indices = [query_indices[i] for i in query_perm]
        query_labels = [query_labels[i] for i in query_perm]
        
        query_encodings = self._tokenize_examples(dataset, query_indices)
        query_labels_tensor = torch.tensor(query_labels, dtype=torch.long)
        
        return support_encodings, query_encodings, support_labels_tensor, query_labels_tensor


class MetaTaskBatchSampler:
    """
    Efficient batch sampler for meta-training that cycles through task pool.
    """
    
    def __init__(
        self,
        task_pool: List[Tuple[int, ...]],
        data_sampler: Banking77TaskSampler,
        split: str = 'train'
    ):
        """
        Args:
            task_pool: List of class combinations (each is a tuple of n_way class IDs)
            data_sampler: Banking77TaskSampler instance
            split: 'train' or 'test'
        """
        self.task_pool = task_pool
        self.data_sampler = data_sampler
        self.split = split
        self.task_indices = list(range(len(task_pool)))
        random.shuffle(self.task_indices)
        self.current_idx = 0
    
    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """
        Sample a meta-batch of tasks.
        
        Returns:
            List of (support_enc, query_enc, support_labels, query_labels) tuples
        """
        batch = []
        for _ in range(batch_size):
            # Cycle through task pool
            if self.current_idx >= len(self.task_indices):
                random.shuffle(self.task_indices)
                self.current_idx = 0
            
            task_idx = self.task_indices[self.current_idx]
            task_classes = self.task_pool[task_idx]
            self.current_idx += 1
            
            # Sample episode for this task
            episode = self.data_sampler.sample_episode(task_classes, split=self.split)
            batch.append(episode)
        
        return batch
    
    def reset(self):
        """Reset the sampler."""
        random.shuffle(self.task_indices)
        self.current_idx = 0
