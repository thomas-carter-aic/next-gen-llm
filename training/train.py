#!/usr/bin/env python3
"""
Custom LLM Training Script for AWS SageMaker
Document ID: training_script_20250705_070000
Created: July 5, 2025 07:00:00 UTC

This script handles the fine-tuning of LLaMA 3.1 models using DeepSpeed optimization
and supports both single-node and distributed training on AWS SageMaker.
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_from_disk, load_dataset
import deepspeed
from accelerate import Accelerator
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/opt/ml/output/training.log')
    ]
)
logger = logging.getLogger(__name__)

class LLMTrainer:
    def __init__(self, args):
        """Initialize the LLM trainer with configuration."""
        self.args = args
        self.model = None
        self.tokenizer = None
        self.train_dataset = None
        self.eval_dataset = None
        
        # SageMaker environment variables
        self.model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
        self.training_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
        self.output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output')
        self.checkpoint_dir = os.environ.get('SM_CHECKPOINT_DIR', '/opt/ml/checkpoints')
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator()
        
        logger.info(f"Training configuration: {vars(args)}")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Training data directory: {self.training_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def setup_model_and_tokenizer(self):
        """Load and configure the model and tokenizer."""
        logger.info(f"Loading model: {self.args.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if self.args.fp16 else torch.float32,
            "device_map": None,  # Let DeepSpeed handle device placement
            "low_cpu_mem_usage": True,
        }
        
        # Add quantization if specified
        if self.args.use_8bit:
            model_kwargs["load_in_8bit"] = True
        elif self.args.use_4bit:
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
            model_kwargs["bnb_4bit_use_double_quant"] = True
            model_kwargs["bnb_4bit_quant_type"] = "nf4"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing to save memory
        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Resize token embeddings if needed
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Model parameters: {self.model.num_parameters():,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
    def load_and_preprocess_data(self):
        """Load and preprocess training data."""
        logger.info("Loading training data...")
        
        try:
            # Try to load preprocessed dataset first
            if os.path.exists(os.path.join(self.training_dir, "dataset_info.json")):
                logger.info("Loading preprocessed dataset from disk")
                dataset = load_from_disk(self.training_dir)
            else:
                # Load raw data and preprocess
                logger.info("Loading raw data for preprocessing")
                dataset = self.load_raw_data()
                dataset = self.preprocess_dataset(dataset)
            
            # Split dataset if needed
            if "train" in dataset:
                self.train_dataset = dataset["train"]
                self.eval_dataset = dataset.get("validation", None)
            else:
                # Split the dataset
                split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
                self.train_dataset = split_dataset["train"]
                self.eval_dataset = split_dataset["test"]
            
            logger.info(f"Training samples: {len(self.train_dataset)}")
            if self.eval_dataset:
                logger.info(f"Evaluation samples: {len(self.eval_dataset)}")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def load_raw_data(self):
        """Load raw data from various sources."""
        # This would be customized based on your data format
        # For now, assume JSON lines format
        data_files = []
        for file_path in Path(self.training_dir).glob("*.jsonl"):
            data_files.append(str(file_path))
        
        if not data_files:
            raise ValueError("No training data files found")
        
        dataset = load_dataset("json", data_files=data_files, split="train")
        return dataset
    
    def preprocess_dataset(self, dataset):
        """Preprocess the dataset for training."""
        logger.info("Preprocessing dataset...")
        
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.args.max_seq_length,
                return_overflowing_tokens=False,
            )
            
            # For causal language modeling, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Filter out sequences that are too short
        def filter_short_sequences(example):
            return len(example["input_ids"]) >= 50
        
        tokenized_dataset = tokenized_dataset.filter(filter_short_sequences)
        
        logger.info(f"Dataset preprocessed. Final size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def setup_training_arguments(self):
        """Configure training arguments."""
        training_args = TrainingArguments(
            output_dir=self.checkpoint_dir,
            overwrite_output_dir=True,
            
            # Training parameters
            num_train_epochs=self.args.epochs,
            per_device_train_batch_size=self.args.batch_size,
            per_device_eval_batch_size=self.args.eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            
            # Optimization
            learning_rate=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            
            # Learning rate scheduling
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            
            # Mixed precision
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            
            # Logging and evaluation
            logging_dir=f"{self.output_dir}/logs",
            logging_steps=self.args.logging_steps,
            eval_steps=self.args.eval_steps,
            evaluation_strategy="steps" if self.eval_dataset else "no",
            save_steps=self.args.save_steps,
            save_total_limit=3,
            
            # Other settings
            dataloader_pin_memory=False,
            dataloader_num_workers=4,
            remove_unused_columns=False,
            label_smoothing_factor=0.1,
            
            # DeepSpeed configuration
            deepspeed=self.args.deepspeed_config,
            
            # Reporting
            report_to=["wandb"] if self.args.use_wandb else [],
            run_name=f"{self.args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # Early stopping
            load_best_model_at_end=True if self.eval_dataset else False,
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
        )
        
        return training_args
    
    def setup_data_collator(self):
        """Setup data collator for language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
            pad_to_multiple_of=8 if self.args.fp16 else None,
        )
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        shift_logits = predictions[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        perplexity = torch.exp(loss)
        
        return {"perplexity": perplexity.item()}
    
    def train(self):
        """Execute the training process."""
        logger.info("Starting training...")
        
        # Setup model and data
        self.setup_model_and_tokenizer()
        self.load_and_preprocess_data()
        
        # Setup training components
        training_args = self.setup_training_arguments()
        data_collator = self.setup_data_collator()
        
        # Initialize wandb if enabled
        if self.args.use_wandb:
            wandb.init(
                project="nexus-llm-training",
                name=training_args.run_name,
                config=vars(self.args)
            )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics if self.eval_dataset else None,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if self.eval_dataset else None,
        )
        
        # Start training
        try:
            train_result = trainer.train(
                resume_from_checkpoint=self.args.resume_from_checkpoint
            )
            
            # Log training results
            logger.info(f"Training completed successfully")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            logger.info(f"Training steps: {train_result.global_step}")
            
            # Save the model
            self.save_model(trainer)
            
            # Save training metrics
            self.save_training_metrics(train_result, trainer)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, trainer):
        """Save the trained model and tokenizer."""
        logger.info("Saving model...")
        
        # Save model and tokenizer
        trainer.save_model(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        
        # Save model configuration
        config = {
            "model_name": self.args.model_name,
            "training_args": vars(self.args),
            "model_parameters": self.model.num_parameters(),
            "training_timestamp": datetime.now().isoformat(),
        }
        
        with open(os.path.join(self.model_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {self.model_dir}")
    
    def save_training_metrics(self, train_result, trainer):
        """Save training metrics and logs."""
        metrics = {
            "training_loss": train_result.training_loss,
            "global_step": train_result.global_step,
            "training_runtime": train_result.metrics.get("train_runtime", 0),
            "training_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
        }
        
        # Add evaluation metrics if available
        if hasattr(trainer.state, "log_history"):
            eval_metrics = [log for log in trainer.state.log_history if "eval_loss" in log]
            if eval_metrics:
                best_eval = min(eval_metrics, key=lambda x: x["eval_loss"])
                metrics["best_eval_loss"] = best_eval["eval_loss"]
                metrics["best_eval_perplexity"] = best_eval.get("eval_perplexity", None)
        
        # Save metrics
        with open(os.path.join(self.output_dir, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Training metrics saved")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train custom LLM")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf",
                       help="Base model name or path")
    parser.add_argument("--max_seq_length", type=int, default=2048,
                       help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                       help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    
    # Optimization arguments
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use mixed precision training")
    parser.add_argument("--bf16", action="store_true", default=False,
                       help="Use bfloat16 precision")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True,
                       help="Enable gradient checkpointing")
    parser.add_argument("--use_8bit", action="store_true", default=False,
                       help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", default=False,
                       help="Use 4-bit quantization")
    
    # Logging and saving
    parser.add_argument("--logging_steps", type=int, default=100,
                       help="Logging frequency")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluation frequency")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Model saving frequency")
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Use Weights & Biases for logging")
    
    # DeepSpeed
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json",
                       help="DeepSpeed configuration file")
    
    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Initialize trainer
    trainer = LLMTrainer(args)
    
    # Start training
    success = trainer.train()
    
    if success:
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error("Training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
