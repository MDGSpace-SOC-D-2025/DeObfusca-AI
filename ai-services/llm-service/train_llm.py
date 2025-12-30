"""
QLoRA Fine-tuning Script for LLM Decompilation

This script fine-tunes a CodeLlama or similar model for decompilation
using QLoRA (Quantized Low-Rank Adaptation) for memory efficiency.

Features:
- 4-bit quantization with bitsandbytes
- LoRA adapters for efficient training
- Gradient checkpointing
- Mixed precision training
- Kaggle T4 GPU compatible (~15GB VRAM)

Usage:
    python train_llm.py --data_dir /path/to/data --output_dir /path/to/output
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from tqdm import tqdm


@dataclass
class TrainingConfig:
    """Configuration for LLM fine-tuning."""
    # Model
    model_name: str = "codellama/CodeLlama-7b-Instruct-hf"
    
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Quantization
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    use_double_quant: bool = True
    
    # Training
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Efficiency
    gradient_checkpointing: bool = True
    fp16: bool = True
    
    # Output
    output_dir: str = "./checkpoints/llm"
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


class DecompilationDataset(Dataset):
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        split: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self) -> List[Dict]:
        samples = []
        
        split_dir = self.data_dir / self.split
        if not split_dir.exists():
            split_dir = self.data_dir
        
        for json_file in split_dir.glob('*.json'):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    else:
                        samples.append(data)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Format input (P-Code)
        pcode = sample.get('pcode', sample.get('instructions', []))
        pcode_str = self._format_pcode(pcode)
        
        # Format target (C code)
        code = sample.get('code', sample.get('source', ''))
        
        # Create instruction-following prompt
        prompt = self._create_prompt(pcode_str, code)
        
        # Tokenize
        encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # For causal LM, labels are the same as input_ids
        # But we mask the prompt part
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Find where the response starts
        response_start = prompt.find('[/INST]') + len('[/INST]')
        response_tokens = self.tokenizer(
            prompt[:response_start],
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )['input_ids'].size(1)
        
        # Create labels (mask prompt, only train on response)
        labels = input_ids.clone()
        labels[:response_tokens] = -100  # Ignore prompt in loss
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _format_pcode(self, pcode: List) -> str:
        lines = []
        for i, op in enumerate(pcode[:100]):  # Limit
            if isinstance(op, dict):
                mnemonic = op.get('mnemonic', 'UNKNOWN')
                address = op.get('address', f'0x{i:04x}')
                lines.append(f"{address}: {mnemonic}")
            else:
                lines.append(f"0x{i:04x}: {str(op)}")
        return '\n'.join(lines)
    
    def _create_prompt(self, pcode: str, code: str) -> str:
        return f"""<s>[INST] You are a decompilation expert. Convert the following binary representation to clean, readable C code.

Binary representation (P-Code):
{pcode}

Generate clean, well-formatted C code that implements this functionality. Include appropriate variable names and comments.

[/INST]
{code}</s>"""


def create_model_and_tokenizer(config: TrainingConfig):
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_use_double_quant=config.use_double_quant,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype)
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def create_synthetic_data(output_dir: str, num_samples: int = 500):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    # Template patterns
    patterns = [
        {
            'pcode': ['LOAD', 'INT_ADD', 'STORE', 'RETURN'],
            'code': '''int add(int a, int b) {
    // Add two integers
    int result = a + b;
    return result;
}'''
        },
        {
            'pcode': ['LOAD', 'INT_SUB', 'STORE', 'RETURN'],
            'code': '''int subtract(int a, int b) {
    // Subtract b from a
    int result = a - b;
    return result;
}'''
        },
        {
            'pcode': ['LOAD', 'INT_MULT', 'STORE', 'RETURN'],
            'code': '''int multiply(int a, int b) {
    // Multiply two integers
    int result = a * b;
    return result;
}'''
        },
        {
            'pcode': ['LOAD', 'CBRANCH', 'INT_ADD', 'BRANCH', 'INT_SUB', 'RETURN'],
            'code': '''int conditional(int a, int b, int cond) {
    // Return sum if cond is true, difference otherwise
    if (cond) {
        return a + b;
    }
    return a - b;
}'''
        },
        {
            'pcode': ['LOAD', 'INT_LESS', 'CBRANCH', 'INT_ADD', 'BRANCH', 'RETURN'],
            'code': '''int loop_sum(int* arr, int n) {
    // Sum all elements in array
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}'''
        },
        {
            'pcode': ['LOAD', 'INT_EQUAL', 'CBRANCH', 'CALL', 'RETURN'],
            'code': '''int find_value(int* arr, int n, int target) {
    // Find target value in array
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}'''
        },
        {
            'pcode': ['LOAD', 'INT_LESS', 'CBRANCH', 'STORE', 'BRANCH', 'RETURN'],
            'code': '''void swap(int* a, int* b) {
    // Swap two values
    int temp = *a;
    *a = *b;
    *b = temp;
}'''
        },
    ]
    
    for i in range(num_samples):
        pattern = random.choice(patterns)
        
        # Add some variation
        pcode = pattern['pcode'].copy()
        if random.random() < 0.3:
            pcode.insert(random.randint(0, len(pcode)), 'NOP')
        
        samples.append({
            'pcode': pcode,
            'code': pattern['code'],
            'function_name': f'func_{i:04d}'
        })
    
    # Split
    random.shuffle(samples)
    n_train = int(0.85 * len(samples))
    n_val = int(0.1 * len(samples))
    
    for split, split_samples in [
        ('train', samples[:n_train]),
        ('val', samples[n_train:n_train + n_val]),
        ('test', samples[n_train + n_val:])
    ]:
        split_dir = output_dir / split
        split_dir.mkdir(exist_ok=True)
        with open(split_dir / 'data.json', 'w') as f:
            json.dump(split_samples, f, indent=2)
    
    print(f"Created: {n_train} train, {n_val} val, {len(samples) - n_train - n_val} test")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM for decompilation')
    
    parser.add_argument('--data_dir', type=str, default='./data/llm')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/llm')
    parser.add_argument('--model_name', type=str, default='codellama/CodeLlama-7b-Instruct-hf')
    
    # LoRA settings
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--max_seq_length', type=int, default=2048)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--create_synthetic', action='store_true')
    parser.add_argument('--num_synthetic', type=int, default=500)
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create config
    config = TrainingConfig(
        model_name=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir
    )
    
    # Create synthetic data if requested
    if args.create_synthetic:
        args.data_dir = str(create_synthetic_data(args.data_dir, args.num_synthetic))
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(Path(config.output_dir) / 'config.json', 'w') as f:
        json.dump({
            'model_name': config.model_name,
            'lora_r': config.lora_r,
            'lora_alpha': config.lora_alpha,
            'lora_dropout': config.lora_dropout,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'num_epochs': config.num_epochs
        }, f, indent=2)
    
    print("Creating model and tokenizer...")
    model, tokenizer = create_model_and_tokenizer(config)
    
    print("Loading datasets...")
    train_dataset = DecompilationDataset(
        args.data_dir, tokenizer, config.max_seq_length, 'train'
    )
    val_dataset = DecompilationDataset(
        args.data_dir, tokenizer, config.max_seq_length, 'val'
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=config.gradient_checkpointing,
        max_grad_norm=1.0,
        dataloader_pin_memory=True,
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Saving model...")
    trainer.save_model(config.output_dir)
    
    # Save adapter separately
    model.save_pretrained(Path(config.output_dir) / 'adapter')
    
    print(f"\nTraining complete! Model saved to {config.output_dir}")


if __name__ == '__main__':
    main()
