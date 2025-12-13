"""
Fine-tune CodeLlama with QLoRA for decompilation.

Dataset: Pairs of (binary, C source code) from AnghaBench, Exampler, etc.
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os

def format_instruction(sample):
    """Format training sample as instruction-following prompt."""
    return f"""<s>[INST] Decompile the following binary code to readable C code.

Binary representation:
{sample['assembly']}

Generate clean, readable C code: [/INST]
{sample['source_code']}
</s>"""

def main():
    # Config
    MODEL_NAME = "codellama/CodeLlama-7b-hf"
    DATASET_PATH = "/data/training/decompilation_pairs"
    OUTPUT_DIR = "/app/models/qlora_adapter"
    
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    NUM_EPOCHS = 3
    MAX_LENGTH = 2048
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    dataset = load_dataset('json', data_files=f'{DATASET_PATH}/train.jsonl', split='train')
    
    # Tokenize
    def tokenize_function(examples):
        texts = [format_instruction(ex) for ex in examples]
        return tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='pt'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    model.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
