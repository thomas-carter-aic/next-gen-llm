#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Custom LLM Training
Document ID: data_preprocessing_20250705_070000
Created: July 5, 2025 07:00:00 UTC

This script handles downloading, cleaning, and preprocessing of training datasets
including The Pile, Red Pajama, and other open-source datasets for LLM training.
"""

import os
import sys
import json
import logging
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config_file='aws_config.json'):
        """Initialize the data preprocessor."""
        self.config = self.load_config(config_file)
        self.s3_client = boto3.client('s3', region_name=self.config['region'])
        self.tokenizer = None
        
        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logger.warning(f"Failed to download NLTK data: {e}")
    
    def load_config(self, config_file):
        """Load AWS configuration."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            raise
    
    def setup_tokenizer(self, model_name="meta-llama/Llama-2-7b-hf"):
        """Setup tokenizer for text processing."""
        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_pile_dataset(self, output_dir="./data/raw/pile"):
        """Download The Pile dataset."""
        logger.info("Downloading The Pile dataset...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load The Pile dataset in streaming mode to manage memory
            dataset = load_dataset("EleutherAI/pile", streaming=True, split="train")
            
            batch_size = 1000
            batch_count = 0
            current_batch = []
            
            for example in tqdm(dataset, desc="Processing The Pile"):
                current_batch.append(example)
                
                if len(current_batch) >= batch_size:
                    # Save batch to file
                    batch_file = os.path.join(output_dir, f"pile_batch_{batch_count:06d}.jsonl")
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        for item in current_batch:
                            f.write(json.dumps(item) + '\n')
                    
                    # Upload to S3
                    s3_key = f"raw/pile/pile_batch_{batch_count:06d}.jsonl"
                    self.s3_client.upload_file(
                        batch_file, 
                        self.config['buckets']['data'], 
                        s3_key
                    )
                    
                    current_batch = []
                    batch_count += 1
                    
                    # Limit for testing (remove in production)
                    if batch_count >= 100:  # Process first 100K examples
                        break
            
            # Handle remaining items
            if current_batch:
                batch_file = os.path.join(output_dir, f"pile_batch_{batch_count:06d}.jsonl")
                with open(batch_file, 'w', encoding='utf-8') as f:
                    for item in current_batch:
                        f.write(json.dumps(item) + '\n')
                
                s3_key = f"raw/pile/pile_batch_{batch_count:06d}.jsonl"
                self.s3_client.upload_file(
                    batch_file, 
                    self.config['buckets']['data'], 
                    s3_key
                )
            
            logger.info(f"Downloaded {batch_count + 1} batches of The Pile dataset")
            
        except Exception as e:
            logger.error(f"Error downloading The Pile dataset: {e}")
            raise
    
    def download_red_pajama_dataset(self, output_dir="./data/raw/red_pajama"):
        """Download Red Pajama dataset."""
        logger.info("Downloading Red Pajama dataset...")
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Load Red Pajama dataset
            dataset = load_dataset("togethercomputer/RedPajama-Data-1T", streaming=True)
            
            for split_name in ['train']:
                if split_name not in dataset:
                    continue
                
                split_data = dataset[split_name]
                batch_size = 1000
                batch_count = 0
                current_batch = []
                
                for example in tqdm(split_data, desc=f"Processing Red Pajama {split_name}"):
                    current_batch.append(example)
                    
                    if len(current_batch) >= batch_size:
                        # Save batch to file
                        batch_file = os.path.join(output_dir, f"red_pajama_{split_name}_batch_{batch_count:06d}.jsonl")
                        with open(batch_file, 'w', encoding='utf-8') as f:
                            for item in current_batch:
                                f.write(json.dumps(item) + '\n')
                        
                        # Upload to S3
                        s3_key = f"raw/red_pajama/red_pajama_{split_name}_batch_{batch_count:06d}.jsonl"
                        self.s3_client.upload_file(
                            batch_file, 
                            self.config['buckets']['data'], 
                            s3_key
                        )
                        
                        current_batch = []
                        batch_count += 1
                        
                        # Limit for testing
                        if batch_count >= 50:  # Process first 50K examples
                            break
                
                # Handle remaining items
                if current_batch:
                    batch_file = os.path.join(output_dir, f"red_pajama_{split_name}_batch_{batch_count:06d}.jsonl")
                    with open(batch_file, 'w', encoding='utf-8') as f:
                        for item in current_batch:
                            f.write(json.dumps(item) + '\n')
                    
                    s3_key = f"raw/red_pajama/red_pajama_{split_name}_batch_{batch_count:06d}.jsonl"
                    self.s3_client.upload_file(
                        batch_file, 
                        self.config['buckets']['data'], 
                        s3_key
                    )
                
                logger.info(f"Downloaded {batch_count + 1} batches of Red Pajama {split_name}")
                
        except Exception as e:
            logger.error(f"Error downloading Red Pajama dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> Optional[str]:
        """Clean and normalize text data."""
        if not text or not isinstance(text, str):
            return None
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Filter out very short or very long texts
        if len(text) < 100 or len(text) > 50000:
            return None
        
        # Basic quality filters
        if text.count('\n') / len(text) > 0.1:  # Too many line breaks
            return None
        
        # Check for minimum word count
        words = text.split()
        if len(words) < 20:
            return None
        
        # Filter out texts with too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        if special_char_ratio > 0.3:
            return None
        
        # Filter out texts that are mostly uppercase
        if sum(1 for c in text if c.isupper()) / len(text) > 0.5:
            return None
        
        return text
    
    def deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Remove duplicate texts using hash-based deduplication."""
        seen_hashes = set()
        deduplicated = []
        
        for text in texts:
            # Create hash of the text
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                deduplicated.append(text)
        
        logger.info(f"Deduplication: {len(texts)} -> {len(deduplicated)} texts")
        return deduplicated
    
    def process_batch(self, batch_data: List[Dict]) -> List[Dict]:
        """Process a batch of data."""
        processed_batch = []
        
        for item in batch_data:
            # Extract text content
            text = item.get('text', '')
            
            # Clean the text
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                processed_item = {
                    'text': cleaned_text,
                    'source': item.get('meta', {}).get('pile_set_name', 'unknown'),
                    'length': len(cleaned_text),
                    'word_count': len(cleaned_text.split())
                }
                processed_batch.append(processed_item)
        
        return processed_batch
    
    def tokenize_dataset(self, dataset: Dataset, max_length: int = 2048) -> Dataset:
        """Tokenize the dataset for training."""
        logger.info("Tokenizing dataset...")
        
        if self.tokenizer is None:
            self.setup_tokenizer()
        
        def tokenize_function(examples):
            # Tokenize the text
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=max_length,
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
            desc="Tokenizing dataset",
            num_proc=min(mp.cpu_count(), 8)
        )
        
        # Filter out sequences that are too short
        def filter_short_sequences(example):
            return len(example["input_ids"]) >= 50
        
        tokenized_dataset = tokenized_dataset.filter(filter_short_sequences)
        
        logger.info(f"Tokenized dataset size: {len(tokenized_dataset)}")
        return tokenized_dataset
    
    def process_raw_data(self, input_dir="./data/raw", output_dir="./data/processed"):
        """Process raw data files."""
        logger.info("Processing raw data...")
        os.makedirs(output_dir, exist_ok=True)
        
        all_processed_data = []
        
        # Process all JSONL files in the raw directory
        for jsonl_file in Path(input_dir).rglob("*.jsonl"):
            logger.info(f"Processing {jsonl_file}")
            
            batch_data = []
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        batch_data.append(item)
                    except json.JSONDecodeError:
                        continue
            
            # Process the batch
            processed_batch = self.process_batch(batch_data)
            all_processed_data.extend(processed_batch)
        
        # Deduplicate
        texts = [item['text'] for item in all_processed_data]
        deduplicated_texts = self.deduplicate_texts(texts)
        
        # Create final dataset
        final_data = []
        for i, text in enumerate(deduplicated_texts):
            final_data.append({
                'text': text,
                'id': i,
                'length': len(text),
                'word_count': len(text.split())
            })
        
        # Convert to Hugging Face dataset
        dataset = Dataset.from_list(final_data)
        
        # Split into train/validation
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        # Tokenize datasets
        train_tokenized = self.tokenize_dataset(split_dataset['train'])
        val_tokenized = self.tokenize_dataset(split_dataset['test'])
        
        # Create final dataset dict
        final_dataset = DatasetDict({
            'train': train_tokenized,
            'validation': val_tokenized
        })
        
        # Save processed dataset
        final_dataset.save_to_disk(output_dir)
        
        # Upload to S3
        self.upload_processed_data(output_dir)
        
        logger.info(f"Processed dataset saved to {output_dir}")
        logger.info(f"Train samples: {len(final_dataset['train'])}")
        logger.info(f"Validation samples: {len(final_dataset['validation'])}")
        
        return final_dataset
    
    def upload_processed_data(self, local_dir):
        """Upload processed data to S3."""
        logger.info("Uploading processed data to S3...")
        
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, local_dir)
                s3_key = f"processed/{relative_path}"
                
                try:
                    self.s3_client.upload_file(
                        local_path,
                        self.config['buckets']['data'],
                        s3_key
                    )
                    logger.info(f"Uploaded {s3_key}")
                except Exception as e:
                    logger.error(f"Failed to upload {s3_key}: {e}")
    
    def generate_dataset_statistics(self, dataset_path="./data/processed"):
        """Generate statistics about the processed dataset."""
        logger.info("Generating dataset statistics...")
        
        try:
            dataset = DatasetDict.load_from_disk(dataset_path)
            
            stats = {
                'total_samples': len(dataset['train']) + len(dataset['validation']),
                'train_samples': len(dataset['train']),
                'validation_samples': len(dataset['validation']),
                'splits': list(dataset.keys())
            }
            
            # Calculate token statistics
            train_lengths = [len(example['input_ids']) for example in dataset['train']]
            val_lengths = [len(example['input_ids']) for example in dataset['validation']]
            
            stats['token_statistics'] = {
                'train': {
                    'mean_length': np.mean(train_lengths),
                    'median_length': np.median(train_lengths),
                    'min_length': np.min(train_lengths),
                    'max_length': np.max(train_lengths),
                    'total_tokens': np.sum(train_lengths)
                },
                'validation': {
                    'mean_length': np.mean(val_lengths),
                    'median_length': np.median(val_lengths),
                    'min_length': np.min(val_lengths),
                    'max_length': np.max(val_lengths),
                    'total_tokens': np.sum(val_lengths)
                }
            }
            
            # Save statistics
            stats_file = os.path.join(dataset_path, "dataset_statistics.json")
            with open(stats_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
            
            # Upload to S3
            self.s3_client.upload_file(
                stats_file,
                self.config['buckets']['data'],
                "processed/dataset_statistics.json"
            )
            
            logger.info("Dataset statistics:")
            logger.info(f"Total samples: {stats['total_samples']:,}")
            logger.info(f"Train samples: {stats['train_samples']:,}")
            logger.info(f"Validation samples: {stats['validation_samples']:,}")
            logger.info(f"Average train sequence length: {stats['token_statistics']['train']['mean_length']:.1f}")
            logger.info(f"Total training tokens: {stats['token_statistics']['train']['total_tokens']:,}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error generating statistics: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for LLM training")
    parser.add_argument("--config", default="aws_config.json", help="AWS configuration file")
    parser.add_argument("--download-pile", action="store_true", help="Download The Pile dataset")
    parser.add_argument("--download-red-pajama", action="store_true", help="Download Red Pajama dataset")
    parser.add_argument("--process-data", action="store_true", help="Process raw data")
    parser.add_argument("--generate-stats", action="store_true", help="Generate dataset statistics")
    parser.add_argument("--model-name", default="meta-llama/Llama-2-7b-hf", help="Model name for tokenizer")
    parser.add_argument("--max-length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--output-dir", default="./data/processed", help="Output directory for processed data")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(args.config)
    preprocessor.setup_tokenizer(args.model_name)
    
    try:
        if args.download_pile:
            preprocessor.download_pile_dataset()
        
        if args.download_red_pajama:
            preprocessor.download_red_pajama_dataset()
        
        if args.process_data:
            dataset = preprocessor.process_raw_data(output_dir=args.output_dir)
            logger.info("Data processing completed successfully")
        
        if args.generate_stats:
            stats = preprocessor.generate_dataset_statistics(args.output_dir)
            logger.info("Statistics generation completed")
        
        logger.info("Data preprocessing pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
