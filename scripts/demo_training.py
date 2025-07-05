#!/usr/bin/env python3
"""
Demo Training Script for Custom LLM
Document ID: demo_training_20250705_080000
Created: July 5, 2025 08:00:00 UTC

This script demonstrates the training setup and process with a smaller model.
"""

import os
import json
import logging
import torch
from datetime import datetime
from typing import Dict, Any

import boto3
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoTrainer:
    def __init__(self, config_file='aws_config.json'):
        """Initialize the demo trainer."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.s3_client = boto3.client('s3', region_name=self.config['region'])
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Demo trainer initialized - Device: {self.device}")
    
    def setup_model_and_tokenizer(self, model_name="microsoft/DialoGPT-small"):
        """Setup model and tokenizer for training."""
        logger.info(f"Loading model and tokenizer: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU training
            trust_remote_code=True
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Model info
        param_count = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Total parameters: {param_count:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return param_count, trainable_params
    
    def load_processed_data(self, data_path="./data/processed"):
        """Load processed training data."""
        logger.info("Loading processed training data...")
        
        try:
            # Load dataset from disk
            dataset = load_from_disk(data_path)
            
            logger.info(f"Loaded dataset with {len(dataset['train'])} training samples")
            logger.info(f"Validation samples: {len(dataset['validation'])}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise
    
    def setup_training_arguments(self, output_dir="./training_output"):
        """Setup training arguments."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            
            # Training parameters (reduced for demo)
            num_train_epochs=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            
            # Optimization
            learning_rate=5e-5,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.95,
            
            # Learning rate scheduling
            lr_scheduler_type="linear",
            warmup_steps=10,
            
            # Logging and evaluation
            logging_dir=f"{output_dir}/logs",
            logging_steps=5,
            eval_steps=10,
            evaluation_strategy="steps",
            save_steps=20,
            save_total_limit=2,
            
            # Other settings
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            
            # Reporting
            report_to=[],  # Disable wandb for demo
            run_name=f"demo-training-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            
            # Early stopping
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        
        return training_args
    
    def setup_data_collator(self):
        """Setup data collator for language modeling."""
        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal language modeling
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
    
    def run_demo_training(self):
        """Run the demo training process."""
        logger.info("Starting demo training...")
        
        try:
            # Setup model and tokenizer
            param_count, trainable_params = self.setup_model_and_tokenizer()
            
            # Load processed data
            dataset = self.load_processed_data()
            
            # Setup training components
            training_args = self.setup_training_arguments()
            data_collator = self.setup_data_collator()
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset['train'],
                eval_dataset=dataset['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )
            
            # Start training
            logger.info("Starting training process...")
            train_result = trainer.train()
            
            # Log training results
            logger.info("Training completed successfully")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            logger.info(f"Training steps: {train_result.global_step}")
            
            # Save the model
            self.save_model(trainer, training_args.output_dir)
            
            # Generate training report
            report = self.generate_training_report(train_result, trainer, param_count)
            
            return report
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def save_model(self, trainer, output_dir):
        """Save the trained model."""
        logger.info("Saving trained model...")
        
        # Save model and tokenizer
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        config = {
            "model_type": "demo_training",
            "training_timestamp": datetime.now().isoformat(),
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }
        
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Upload to S3
        self.upload_model_to_s3(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
    
    def upload_model_to_s3(self, local_dir):
        """Upload trained model to S3."""
        logger.info("Uploading model to S3...")
        
        models_bucket = self.config['buckets']['models']
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"demo-model/{relative_path}"
                
                try:
                    self.s3_client.upload_file(local_path, models_bucket, s3_key)
                    logger.info(f"Uploaded {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {s3_key}: {e}")
    
    def generate_training_report(self, train_result, trainer, param_count):
        """Generate training report."""
        report = {
            "training_summary": {
                "status": "completed",
                "training_loss": train_result.training_loss,
                "global_step": train_result.global_step,
                "training_runtime": train_result.metrics.get("train_runtime", 0),
                "samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            },
            "model_info": {
                "total_parameters": param_count,
                "device": str(self.device),
                "model_type": "demo_model"
            },
            "evaluation_metrics": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Add evaluation metrics if available
        if hasattr(trainer.state, "log_history"):
            eval_metrics = [log for log in trainer.state.log_history if "eval_loss" in log]
            if eval_metrics:
                best_eval = min(eval_metrics, key=lambda x: x["eval_loss"])
                report["evaluation_metrics"] = {
                    "best_eval_loss": best_eval["eval_loss"],
                    "best_eval_perplexity": best_eval.get("eval_perplexity", None)
                }
        
        # Save report
        with open("demo_training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def test_model_generation(self, output_dir="./training_output"):
        """Test the trained model with text generation."""
        logger.info("Testing model generation...")
        
        try:
            # Load the trained model
            model = AutoModelForCausalLM.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
            
            # Test prompts
            test_prompts = [
                "Artificial intelligence is",
                "Machine learning helps",
                "The future of technology"
            ]
            
            results = []
            
            for prompt in test_prompts:
                # Tokenize input
                inputs = tokenizer(prompt, return_tensors="pt")
                
                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # Decode response
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                results.append({
                    "prompt": prompt,
                    "generated": generated_text,
                    "new_text": generated_text[len(prompt):].strip()
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Model generation test failed: {e}")
            return []

def main():
    trainer = DemoTrainer()
    
    try:
        # Run demo training
        report = trainer.run_demo_training()
        
        # Test model generation
        generation_results = trainer.test_model_generation()
        
        # Print summary
        print("\n" + "="*80)
        print("DEMO TRAINING COMPLETED")
        print("="*80)
        print(f"Training Loss: {report['training_summary']['training_loss']:.4f}")
        print(f"Training Steps: {report['training_summary']['global_step']}")
        print(f"Model Parameters: {report['model_info']['total_parameters']:,}")
        print(f"Device: {report['model_info']['device']}")
        
        if report['evaluation_metrics']:
            print(f"Best Eval Loss: {report['evaluation_metrics']['best_eval_loss']:.4f}")
            if report['evaluation_metrics']['best_eval_perplexity']:
                print(f"Best Perplexity: {report['evaluation_metrics']['best_eval_perplexity']:.2f}")
        
        print("\nGeneration Test Results:")
        for i, result in enumerate(generation_results, 1):
            print(f"{i}. Prompt: '{result['prompt']}'")
            print(f"   Generated: '{result['new_text']}'")
        
        print("="*80)
        print("Training report saved to: demo_training_report.json")
        print("Model saved to: ./training_output")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Demo training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
