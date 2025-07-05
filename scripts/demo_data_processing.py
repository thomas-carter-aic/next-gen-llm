#!/usr/bin/env python3
"""
Demo Data Processing for Custom LLM
Document ID: demo_data_processing_20250705_080000
Created: July 5, 2025 08:00:00 UTC

This script demonstrates the data processing pipeline with sample data.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import boto3
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DemoDataProcessor:
    def __init__(self, config_file='aws_config.json'):
        """Initialize the demo data processor."""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.s3_client = boto3.client('s3', region_name=self.config['region'])
        self.tokenizer = None
        
        logger.info("Demo data processor initialized")
    
    def setup_tokenizer(self, model_name="microsoft/DialoGPT-medium"):
        """Setup tokenizer for text processing."""
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample training data."""
        logger.info("Creating sample training data...")
        
        sample_texts = [
            "Artificial intelligence is a branch of computer science that aims to create intelligent machines.",
            "Machine learning is a subset of AI that enables computers to learn without being explicitly programmed.",
            "Deep learning uses neural networks with multiple layers to model and understand complex patterns.",
            "Natural language processing helps computers understand and generate human language.",
            "Computer vision enables machines to interpret and understand visual information from the world.",
            "Robotics combines AI with mechanical engineering to create autonomous machines.",
            "Data science involves extracting insights and knowledge from structured and unstructured data.",
            "Cloud computing provides on-demand access to computing resources over the internet.",
            "Cybersecurity protects digital systems, networks, and data from digital attacks.",
            "Blockchain is a distributed ledger technology that maintains a continuously growing list of records.",
            "The Internet of Things connects everyday objects to the internet, enabling data collection and exchange.",
            "Quantum computing uses quantum-mechanical phenomena to perform operations on data.",
            "Virtual reality creates immersive, computer-generated environments that users can interact with.",
            "Augmented reality overlays digital information onto the real world through devices like smartphones.",
            "Big data refers to extremely large datasets that require special tools and techniques to process.",
            "Software engineering is the systematic approach to designing, developing, and maintaining software systems.",
            "Database management systems organize, store, and retrieve data efficiently for applications.",
            "Web development involves creating websites and web applications using various programming languages.",
            "Mobile app development focuses on creating applications for smartphones and tablets.",
            "DevOps combines software development and IT operations to improve collaboration and productivity."
        ]
        
        # Create dataset entries
        data = []
        for i, text in enumerate(sample_texts):
            data.append({
                'id': i,
                'text': text,
                'source': 'demo_data',
                'length': len(text),
                'word_count': len(text.split())
            })
        
        logger.info(f"Created {len(data)} sample entries")
        return data
    
    def preprocess_data(self, data: List[Dict[str, Any]]) -> Dataset:
        """Preprocess the sample data."""
        logger.info("Preprocessing sample data...")
        
        # Convert to Hugging Face dataset
        dataset = Dataset.from_list(data)
        
        # Tokenize the data
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=512,
                return_overflowing_tokens=False,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            desc="Tokenizing dataset"
        )
        
        # Add labels for causal language modeling
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        
        logger.info(f"Preprocessed dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def create_train_val_split(self, dataset: Dataset) -> DatasetDict:
        """Create train/validation split."""
        logger.info("Creating train/validation split...")
        
        # Split dataset (80% train, 20% validation)
        split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        dataset_dict = DatasetDict({
            'train': split_dataset['train'],
            'validation': split_dataset['test']
        })
        
        logger.info(f"Train samples: {len(dataset_dict['train'])}")
        logger.info(f"Validation samples: {len(dataset_dict['validation'])}")
        
        return dataset_dict
    
    def save_processed_data(self, dataset_dict: DatasetDict, output_dir="./data/processed"):
        """Save processed data locally and to S3."""
        logger.info("Saving processed data...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save dataset locally
        dataset_dict.save_to_disk(output_dir)
        
        # Generate statistics
        stats = self.generate_statistics(dataset_dict)
        
        # Save statistics
        stats_file = os.path.join(output_dir, "dataset_statistics.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Upload to S3
        self.upload_to_s3(output_dir)
        
        logger.info(f"Processed data saved to {output_dir}")
        return stats
    
    def generate_statistics(self, dataset_dict: DatasetDict) -> Dict[str, Any]:
        """Generate dataset statistics."""
        logger.info("Generating dataset statistics...")
        
        train_lengths = [len(example['input_ids']) for example in dataset_dict['train']]
        val_lengths = [len(example['input_ids']) for example in dataset_dict['validation']]
        
        stats = {
            'total_samples': len(dataset_dict['train']) + len(dataset_dict['validation']),
            'train_samples': len(dataset_dict['train']),
            'validation_samples': len(dataset_dict['validation']),
            'train_statistics': {
                'mean_length': sum(train_lengths) / len(train_lengths),
                'min_length': min(train_lengths),
                'max_length': max(train_lengths),
                'total_tokens': sum(train_lengths)
            },
            'validation_statistics': {
                'mean_length': sum(val_lengths) / len(val_lengths),
                'min_length': min(val_lengths),
                'max_length': max(val_lengths),
                'total_tokens': sum(val_lengths)
            },
            'processing_timestamp': datetime.now().isoformat()
        }
        
        return stats
    
    def upload_to_s3(self, local_dir: str):
        """Upload processed data to S3."""
        logger.info("Uploading processed data to S3...")
        
        data_bucket = self.config['buckets']['data']
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"processed/{relative_path}"
                
                try:
                    self.s3_client.upload_file(local_path, data_bucket, s3_key)
                    logger.info(f"Uploaded {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {s3_key}: {e}")
    
    def run_demo_processing(self):
        """Run the complete demo processing pipeline."""
        logger.info("Starting demo data processing pipeline...")
        
        try:
            # Setup tokenizer
            self.setup_tokenizer()
            
            # Create sample data
            sample_data = self.create_sample_data()
            
            # Preprocess data
            processed_dataset = self.preprocess_data(sample_data)
            
            # Create train/val split
            dataset_dict = self.create_train_val_split(processed_dataset)
            
            # Save processed data
            stats = self.save_processed_data(dataset_dict)
            
            # Print summary
            print("\n" + "="*60)
            print("DEMO DATA PROCESSING COMPLETED")
            print("="*60)
            print(f"Total samples: {stats['total_samples']}")
            print(f"Training samples: {stats['train_samples']}")
            print(f"Validation samples: {stats['validation_samples']}")
            print(f"Average sequence length: {stats['train_statistics']['mean_length']:.1f}")
            print(f"Total training tokens: {stats['train_statistics']['total_tokens']:,}")
            print("="*60)
            
            logger.info("Demo processing completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Demo processing failed: {e}")
            return False

def main():
    processor = DemoDataProcessor()
    success = processor.run_demo_processing()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
