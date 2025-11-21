"""
LLM Model Wrapper with LoRA for Meta-Learning

Supports TinyLlama and Gemma models with PEFT LoRA adaptation.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import copy
from typing import Dict, List, Optional


class LLMLoRAClassifier:
    """
    Wrapper for LLM + LoRA classifier for 5-way classification.
    """
    
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        num_labels: int = 5,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device: str = "cuda:0",
        load_in_8bit: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name
            num_labels: Number of classes (fixed at 5 for 5-way)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            device: Device to load model
            load_in_8bit: Whether to use 8-bit quantization
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device
        self.load_in_8bit = load_in_8bit
        
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading base model: {model_name}")
        # Load base model
        model_kwargs = {
            "num_labels": num_labels,
            "trust_remote_code": True
        }
        
        if load_in_8bit:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = {"": device}
        
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # Configure pad token
        if self.base_model.config.pad_token_id is None:
            self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Determine target modules based on model architecture
        target_modules = self._get_target_modules()
        
        print(f"Applying LoRA with r={lora_r}, alpha={lora_alpha}, target_modules={target_modules}")
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            inference_mode=False
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.base_model, lora_config)
        
        if not load_in_8bit:
            self.model = self.model.to(device)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
    def _get_target_modules(self) -> List[str]:
        """
        Determine LoRA target modules based on model architecture.
        """
        # Check model architecture
        config = self.base_model.config
        
        # Common patterns for different architectures
        if "llama" in self.model_name.lower() or "tinyllama" in self.model_name.lower():
            # LLaMA/TinyLlama architecture
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gemma" in self.model_name.lower():
            # Gemma architecture (similar to LLaMA)
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in self.model_name.lower():
            # GPT-2 architecture
            return ["c_attn", "c_proj", "c_fc"]
        else:
            # Default: attention modules
            print("Warning: Using default target modules. May need adjustment.")
            return ["q_proj", "v_proj"]
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Get list of trainable parameters (LoRA + classifier head).
        """
        return [p for p in self.model.parameters() if p.requires_grad]
    
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass.
        
        Returns:
            outputs: ModelOutput with loss, logits, etc.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def get_state_dict(self) -> Dict:
        """
        Get state dict of trainable parameters only.
        """
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def set_state_dict(self, state_dict: Dict):
        """
        Set state dict of trainable parameters.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in state_dict:
                param.data.copy_(state_dict[name])
    
    def clone_state(self) -> Dict:
        """
        Create a deep copy of trainable parameter state.
        """
        return {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
    
    def to(self, device):
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def train(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()


def clone_model_state(model_wrapper: LLMLoRAClassifier) -> Dict:
    """
    Clone the current state of trainable parameters.
    
    Args:
        model_wrapper: LLMLoRAClassifier instance
        
    Returns:
        Dictionary of cloned parameters
    """
    return model_wrapper.clone_state()


def load_model_state(model_wrapper: LLMLoRAClassifier, state_dict: Dict):
    """
    Load state dict into model.
    
    Args:
        model_wrapper: LLMLoRAClassifier instance
        state_dict: Dictionary of parameters
    """
    model_wrapper.set_state_dict(state_dict)


def interpolate_states(old_state: Dict, new_state: Dict, epsilon: float) -> Dict:
    """
    Interpolate between two model states (Reptile meta-update).
    
    new_state_interpolated = old_state + epsilon * (new_state - old_state)
    
    Args:
        old_state: Original state dict
        new_state: Target state dict
        epsilon: Interpolation coefficient (meta_step_size)
        
    Returns:
        Interpolated state dict
    """
    interpolated = {}
    for key in new_state.keys():
        if key in old_state:
            interpolated[key] = old_state[key] + epsilon * (new_state[key] - old_state[key])
        else:
            interpolated[key] = new_state[key]
    return interpolated


def average_states(state_list: List[Dict]) -> Dict:
    """
    Average a list of state dicts.
    
    Args:
        state_list: List of state dictionaries
        
    Returns:
        Averaged state dict
    """
    if len(state_list) == 0:
        raise ValueError("Cannot average empty list of states")
    
    if len(state_list) == 1:
        return state_list[0]
    
    # Initialize with first state
    averaged = {}
    for key in state_list[0].keys():
        averaged[key] = state_list[0][key].clone()
    
    # Sum remaining states
    for state in state_list[1:]:
        for key in state.keys():
            if key in averaged:
                averaged[key] += state[key]
    
    # Divide by count
    for key in averaged.keys():
        averaged[key] /= len(state_list)
    
    return averaged
