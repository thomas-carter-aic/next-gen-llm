#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Document ID: model_evaluation_20250705_080000
Created: July 5, 2025 08:00:00 UTC

This script provides comprehensive evaluation of the trained LLM model
including standard benchmarks, custom metrics, and performance analysis.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import evaluate
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """Initialize the model evaluator."""
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Evaluation results
        self.results = {
            'model_info': {},
            'benchmark_results': {},
            'custom_metrics': {},
            'performance_analysis': {},
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Initialized evaluator for model: {model_path}")
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        logger.info("Loading model and tokenizer...")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_path,
                trust_remote_code=True,
                use_fast=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Model info
            param_count = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            self.results['model_info'] = {
                'model_path': self.model_path,
                'total_parameters': param_count,
                'trainable_parameters': trainable_params,
                'model_size_gb': param_count * 2 / (1024**3),  # Assuming float16
                'device': str(self.device),
                'torch_dtype': str(self.model.dtype)
            }
            
            logger.info(f"Model loaded successfully")
            logger.info(f"Total parameters: {param_count:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate_perplexity(self, dataset_name: str = "wikitext", subset: str = "wikitext-2-raw-v1", max_samples: int = 1000):
        """Evaluate model perplexity on a standard dataset."""
        logger.info(f"Evaluating perplexity on {dataset_name}...")
        
        try:
            # Load dataset
            dataset = load_dataset(dataset_name, subset, split="test")
            
            # Limit samples for faster evaluation
            if len(dataset) > max_samples:
                dataset = dataset.select(range(max_samples))
            
            # Tokenize dataset
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding=False,
                    max_length=1024,
                    return_overflowing_tokens=False
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Calculate perplexity
            total_loss = 0
            total_tokens = 0
            
            self.model.eval()
            with torch.no_grad():
                for example in tqdm(tokenized_dataset, desc="Calculating perplexity"):
                    input_ids = torch.tensor([example["input_ids"]]).to(self.device)
                    
                    if len(input_ids[0]) < 10:  # Skip very short sequences
                        continue
                    
                    # Calculate loss
                    outputs = self.model(input_ids, labels=input_ids)
                    loss = outputs.loss
                    
                    total_loss += loss.item() * len(input_ids[0])
                    total_tokens += len(input_ids[0])
            
            # Calculate perplexity
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            self.results['benchmark_results']['perplexity'] = {
                'dataset': f"{dataset_name}-{subset}",
                'samples_evaluated': len(tokenized_dataset),
                'total_tokens': total_tokens,
                'average_loss': avg_loss,
                'perplexity': perplexity
            }
            
            logger.info(f"Perplexity: {perplexity:.2f}")
            return perplexity
            
        except Exception as e:
            logger.error(f"Perplexity evaluation failed: {e}")
            return None
    
    def evaluate_text_generation_quality(self, prompts: List[str], max_new_tokens: int = 100):
        """Evaluate text generation quality with custom prompts."""
        logger.info("Evaluating text generation quality...")
        
        generation_results = []
        
        self.model.eval()
        with torch.no_grad():
            for i, prompt in enumerate(tqdm(prompts, desc="Generating text")):
                try:
                    # Tokenize prompt
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=1024
                    ).to(self.device)
                    
                    # Generate response
                    start_time = time.time()
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=0.8,
                        top_p=0.95,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                    generation_time = time.time() - start_time
                    
                    # Decode response
                    generated_text = self.tokenizer.decode(
                        outputs[0][inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    
                    # Calculate metrics
                    tokens_generated = len(outputs[0]) - inputs.input_ids.shape[1]
                    tokens_per_second = tokens_generated / generation_time
                    
                    result = {
                        'prompt_id': i,
                        'prompt': prompt,
                        'generated_text': generated_text,
                        'tokens_generated': tokens_generated,
                        'generation_time': generation_time,
                        'tokens_per_second': tokens_per_second,
                        'prompt_length': len(prompt),
                        'response_length': len(generated_text)
                    }
                    
                    generation_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to generate text for prompt {i}: {e}")
                    continue
        
        # Calculate aggregate metrics
        if generation_results:
            avg_generation_time = np.mean([r['generation_time'] for r in generation_results])
            avg_tokens_per_second = np.mean([r['tokens_per_second'] for r in generation_results])
            avg_response_length = np.mean([r['response_length'] for r in generation_results])
            
            self.results['custom_metrics']['text_generation'] = {
                'total_prompts': len(prompts),
                'successful_generations': len(generation_results),
                'success_rate': len(generation_results) / len(prompts),
                'avg_generation_time': avg_generation_time,
                'avg_tokens_per_second': avg_tokens_per_second,
                'avg_response_length': avg_response_length,
                'sample_generations': generation_results[:3]  # First 3 for review
            }
            
            logger.info(f"Generated {len(generation_results)}/{len(prompts)} responses successfully")
            logger.info(f"Average generation time: {avg_generation_time:.2f}s")
            logger.info(f"Average tokens/second: {avg_tokens_per_second:.1f}")
        
        return generation_results
    
    def evaluate_reasoning_capabilities(self):
        """Evaluate model's reasoning capabilities with custom tasks."""
        logger.info("Evaluating reasoning capabilities...")
        
        reasoning_tasks = [
            {
                'type': 'arithmetic',
                'prompt': 'What is 15 + 27?',
                'expected_answer': '42',
                'category': 'basic_math'
            },
            {
                'type': 'logical',
                'prompt': 'If all cats are animals, and Fluffy is a cat, what can we conclude about Fluffy?',
                'expected_keywords': ['animal', 'Fluffy is an animal'],
                'category': 'logical_reasoning'
            },
            {
                'type': 'common_sense',
                'prompt': 'What happens when you drop a glass on a hard floor?',
                'expected_keywords': ['break', 'shatter', 'pieces'],
                'category': 'common_sense'
            },
            {
                'type': 'reading_comprehension',
                'prompt': 'The sun is a star located at the center of our solar system. It provides light and heat to Earth. What type of celestial body is the sun?',
                'expected_answer': 'star',
                'category': 'reading_comprehension'
            }
        ]
        
        reasoning_results = []
        
        for task in tqdm(reasoning_tasks, desc="Evaluating reasoning"):
            try:
                # Generate response
                inputs = self.tokenizer(
                    task['prompt'],
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.3,  # Lower temperature for more focused answers
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                ).strip().lower()
                
                # Evaluate response
                score = 0
                if 'expected_answer' in task:
                    if task['expected_answer'].lower() in response:
                        score = 1
                elif 'expected_keywords' in task:
                    keyword_matches = sum(1 for keyword in task['expected_keywords'] 
                                        if keyword.lower() in response)
                    score = keyword_matches / len(task['expected_keywords'])
                
                result = {
                    'task_type': task['type'],
                    'category': task['category'],
                    'prompt': task['prompt'],
                    'response': response,
                    'score': score,
                    'max_score': 1.0
                }
                
                reasoning_results.append(result)
                
            except Exception as e:
                logger.warning(f"Failed to evaluate reasoning task {task['type']}: {e}")
                continue
        
        # Calculate aggregate scores
        if reasoning_results:
            category_scores = {}
            for result in reasoning_results:
                category = result['category']
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(result['score'])
            
            # Average scores by category
            avg_category_scores = {
                category: np.mean(scores) 
                for category, scores in category_scores.items()
            }
            
            overall_score = np.mean([r['score'] for r in reasoning_results])
            
            self.results['custom_metrics']['reasoning'] = {
                'overall_score': overall_score,
                'category_scores': avg_category_scores,
                'total_tasks': len(reasoning_tasks),
                'completed_tasks': len(reasoning_results),
                'detailed_results': reasoning_results
            }
            
            logger.info(f"Reasoning evaluation completed")
            logger.info(f"Overall reasoning score: {overall_score:.2f}")
            for category, score in avg_category_scores.items():
                logger.info(f"  {category}: {score:.2f}")
        
        return reasoning_results
    
    def benchmark_performance(self, num_samples: int = 100):
        """Benchmark model performance metrics."""
        logger.info("Benchmarking performance...")
        
        # Test prompts of varying lengths
        test_prompts = [
            "Hello",  # Very short
            "What is artificial intelligence?",  # Short
            "Explain the concept of machine learning and its applications in modern technology.",  # Medium
            "Write a detailed explanation of quantum computing, including its principles, current limitations, and potential future applications in various fields such as cryptography, drug discovery, and financial modeling."  # Long
        ]
        
        performance_results = []
        
        self.model.eval()
        with torch.no_grad():
            for prompt_length_category, prompt in enumerate(test_prompts):
                category_name = ['very_short', 'short', 'medium', 'long'][prompt_length_category]
                
                # Run multiple iterations for statistical significance
                times = []
                token_counts = []
                
                for _ in range(num_samples // len(test_prompts)):
                    try:
                        inputs = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=1024
                        ).to(self.device)
                        
                        start_time = time.time()
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=100,
                            temperature=0.8,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                        end_time = time.time()
                        
                        generation_time = end_time - start_time
                        tokens_generated = len(outputs[0]) - inputs.input_ids.shape[1]
                        
                        times.append(generation_time)
                        token_counts.append(tokens_generated)
                        
                    except Exception as e:
                        logger.warning(f"Performance benchmark failed for {category_name}: {e}")
                        continue
                
                if times:
                    result = {
                        'prompt_category': category_name,
                        'prompt_length': len(prompt),
                        'samples': len(times),
                        'avg_time': np.mean(times),
                        'median_time': np.median(times),
                        'std_time': np.std(times),
                        'min_time': np.min(times),
                        'max_time': np.max(times),
                        'avg_tokens': np.mean(token_counts),
                        'avg_tokens_per_second': np.mean(token_counts) / np.mean(times)
                    }
                    performance_results.append(result)
        
        self.results['performance_analysis']['benchmark'] = performance_results
        
        logger.info("Performance benchmarking completed")
        for result in performance_results:
            logger.info(f"{result['prompt_category']}: {result['avg_time']:.3f}s avg, {result['avg_tokens_per_second']:.1f} tokens/s")
        
        return performance_results
    
    def create_evaluation_report(self, output_dir: str = "./evaluation_results"):
        """Create comprehensive evaluation report."""
        logger.info("Creating evaluation report...")
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create summary report
        summary_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.md")
        self.create_markdown_report(summary_file)
        
        # Create visualizations
        self.create_visualizations(output_dir, timestamp)
        
        logger.info(f"Evaluation report saved to {output_dir}")
        logger.info(f"Results file: {results_file}")
        logger.info(f"Summary report: {summary_file}")
        
        return results_file, summary_file
    
    def create_markdown_report(self, output_file: str):
        """Create a markdown summary report."""
        with open(output_file, 'w') as f:
            f.write("# Model Evaluation Report\n\n")
            f.write(f"**Generated**: {self.results['timestamp']}\n")
            f.write(f"**Model Path**: {self.results['model_info']['model_path']}\n\n")
            
            # Model Information
            f.write("## Model Information\n\n")
            model_info = self.results['model_info']
            f.write(f"- **Total Parameters**: {model_info['total_parameters']:,}\n")
            f.write(f"- **Trainable Parameters**: {model_info['trainable_parameters']:,}\n")
            f.write(f"- **Model Size**: {model_info['model_size_gb']:.2f} GB\n")
            f.write(f"- **Device**: {model_info['device']}\n")
            f.write(f"- **Data Type**: {model_info['torch_dtype']}\n\n")
            
            # Benchmark Results
            if 'benchmark_results' in self.results and self.results['benchmark_results']:
                f.write("## Benchmark Results\n\n")
                
                if 'perplexity' in self.results['benchmark_results']:
                    perp = self.results['benchmark_results']['perplexity']
                    f.write(f"### Perplexity\n")
                    f.write(f"- **Dataset**: {perp['dataset']}\n")
                    f.write(f"- **Perplexity**: {perp['perplexity']:.2f}\n")
                    f.write(f"- **Samples Evaluated**: {perp['samples_evaluated']:,}\n")
                    f.write(f"- **Total Tokens**: {perp['total_tokens']:,}\n\n")
            
            # Custom Metrics
            if 'custom_metrics' in self.results and self.results['custom_metrics']:
                f.write("## Custom Metrics\n\n")
                
                if 'text_generation' in self.results['custom_metrics']:
                    tg = self.results['custom_metrics']['text_generation']
                    f.write(f"### Text Generation Quality\n")
                    f.write(f"- **Success Rate**: {tg['success_rate']:.2%}\n")
                    f.write(f"- **Average Generation Time**: {tg['avg_generation_time']:.2f}s\n")
                    f.write(f"- **Average Tokens/Second**: {tg['avg_tokens_per_second']:.1f}\n")
                    f.write(f"- **Average Response Length**: {tg['avg_response_length']:.0f} characters\n\n")
                
                if 'reasoning' in self.results['custom_metrics']:
                    reasoning = self.results['custom_metrics']['reasoning']
                    f.write(f"### Reasoning Capabilities\n")
                    f.write(f"- **Overall Score**: {reasoning['overall_score']:.2f}/1.0\n")
                    f.write(f"- **Category Scores**:\n")
                    for category, score in reasoning['category_scores'].items():
                        f.write(f"  - {category.replace('_', ' ').title()}: {score:.2f}/1.0\n")
                    f.write("\n")
            
            # Performance Analysis
            if 'performance_analysis' in self.results and self.results['performance_analysis']:
                f.write("## Performance Analysis\n\n")
                
                if 'benchmark' in self.results['performance_analysis']:
                    f.write("### Performance Benchmark\n\n")
                    f.write("| Prompt Category | Avg Time (s) | Tokens/Second | Samples |\n")
                    f.write("|----------------|--------------|---------------|----------|\n")
                    
                    for result in self.results['performance_analysis']['benchmark']:
                        f.write(f"| {result['prompt_category'].title()} | "
                               f"{result['avg_time']:.3f} | "
                               f"{result['avg_tokens_per_second']:.1f} | "
                               f"{result['samples']} |\n")
                    f.write("\n")
    
    def create_visualizations(self, output_dir: str, timestamp: str):
        """Create evaluation visualizations."""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Evaluation Results', fontsize=16)
            
            # Performance benchmark chart
            if 'performance_analysis' in self.results and 'benchmark' in self.results['performance_analysis']:
                perf_data = self.results['performance_analysis']['benchmark']
                categories = [r['prompt_category'] for r in perf_data]
                times = [r['avg_time'] for r in perf_data]
                tokens_per_sec = [r['avg_tokens_per_second'] for r in perf_data]
                
                # Response time by category
                axes[0, 0].bar(categories, times, color='skyblue', alpha=0.7)
                axes[0, 0].set_title('Average Response Time by Prompt Category')
                axes[0, 0].set_ylabel('Time (seconds)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                
                # Tokens per second by category
                axes[0, 1].bar(categories, tokens_per_sec, color='lightgreen', alpha=0.7)
                axes[0, 1].set_title('Tokens per Second by Prompt Category')
                axes[0, 1].set_ylabel('Tokens/Second')
                axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Reasoning scores
            if 'custom_metrics' in self.results and 'reasoning' in self.results['custom_metrics']:
                reasoning_data = self.results['custom_metrics']['reasoning']['category_scores']
                categories = list(reasoning_data.keys())
                scores = list(reasoning_data.values())
                
                axes[1, 0].bar(categories, scores, color='coral', alpha=0.7)
                axes[1, 0].set_title('Reasoning Capability Scores')
                axes[1, 0].set_ylabel('Score (0-1)')
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Model size comparison (placeholder)
            model_sizes = ['GPT-3.5', 'Our Model', 'GPT-4']
            param_counts = [175, self.results['model_info']['total_parameters'] / 1e9, 1760]  # In billions
            
            axes[1, 1].bar(model_sizes, param_counts, color='gold', alpha=0.7)
            axes[1, 1].set_title('Model Size Comparison')
            axes[1, 1].set_ylabel('Parameters (Billions)')
            axes[1, 1].set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'evaluation_charts_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualizations saved to evaluation_charts_{timestamp}.png")
            
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
    
    def run_comprehensive_evaluation(self):
        """Run all evaluation tests."""
        logger.info("Starting comprehensive model evaluation...")
        
        try:
            # Load model
            self.load_model()
            
            # Run evaluations
            logger.info("Running perplexity evaluation...")
            self.evaluate_perplexity()
            
            logger.info("Running text generation quality evaluation...")
            test_prompts = [
                "Explain artificial intelligence",
                "What is the meaning of life?",
                "Describe the process of photosynthesis",
                "Write a short story about a robot",
                "What are the benefits of renewable energy?"
            ]
            self.evaluate_text_generation_quality(test_prompts)
            
            logger.info("Running reasoning capabilities evaluation...")
            self.evaluate_reasoning_capabilities()
            
            logger.info("Running performance benchmark...")
            self.benchmark_performance()
            
            # Generate report
            results_file, summary_file = self.create_evaluation_report()
            
            logger.info("Comprehensive evaluation completed successfully!")
            return results_file, summary_file
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LLM model")
    parser.add_argument("--model-path", required=True, help="Path to the trained model")
    parser.add_argument("--tokenizer-path", help="Path to tokenizer (defaults to model path)")
    parser.add_argument("--output-dir", default="./evaluation_results", help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples for perplexity evaluation")
    parser.add_argument("--benchmark-samples", type=int, default=100, help="Samples for performance benchmark")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model_path, args.tokenizer_path)
    
    try:
        # Run comprehensive evaluation
        results_file, summary_file = evaluator.run_comprehensive_evaluation()
        
        print("\n" + "="*80)
        print("MODEL EVALUATION COMPLETED")
        print("="*80)
        print(f"Results saved to: {results_file}")
        print(f"Summary report: {summary_file}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
