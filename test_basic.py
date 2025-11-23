#!/usr/bin/env python3
"""
Simple test: Load model and data, run one inner loop
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible devices: {torch.cuda.device_count()}")

sys.path.append('/root/ssd/reptile-scaling-law')

print("\n1. Loading model...")
from llm_model import LLMLoRAClassifier

model = LLMLoRAClassifier(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_labels=5,
    lora_r=8,  # Smaller rank
    lora_alpha=16,
    device='cuda:0',  # Will map to GPU 4
    load_in_8bit=False
)
print("✓ Model loaded")

print("\n2. Loading data...")
from banking77_data import Banking77TaskSampler

sampler = Banking77TaskSampler(
    tokenizer=model.tokenizer,
    max_length=64,
    n_way=5,
    k_support=5,
    k_query=15
)
print("✓ Data loaded")

print("\n3. Generating test tasks...")
test_tasks = sampler.generate_task_pool(n_tasks=10, split='train', seed=42)
print(f"✓ Generated {len(test_tasks)} tasks")

print("\n4. Sampling one episode...")
support_enc, query_enc, support_labels, query_labels = sampler.sample_episode(
    test_tasks[0], split='train'
)
print(f"Support shape: {support_enc['input_ids'].shape}")
print(f"Query shape: {query_enc['input_ids'].shape}")

print("\n5. Testing forward pass...")
model.train()
support_input = support_enc['input_ids'][:5].to('cuda:0')
support_mask = support_enc['attention_mask'][:5].to('cuda:0')
support_lab = support_labels[:5].to('cuda:0')

with torch.no_grad():
    outputs = model.forward(support_input, support_mask, support_lab)
    print(f"Loss: {outputs.loss.item():.4f}")

print("\n6. Testing one gradient step...")
from torch.optim import AdamW
optimizer = AdamW(model.get_trainable_params(), lr=5e-4)

outputs = model.forward(support_input, support_mask, support_lab)
loss = outputs.loss
print(f"Before: {loss.item():.4f}")

optimizer.zero_grad()
loss.backward()
optimizer.step()

with torch.no_grad():
    outputs = model.forward(support_input, support_mask, support_lab)
    print(f"After: {outputs.loss.item():.4f}")

print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
