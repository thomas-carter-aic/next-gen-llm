#!/usr/bin/env python3
"""
Load Testing Script for Custom LLM API
Document ID: load_test_script_20250705_080000
Created: July 5, 2025 08:00:00 UTC

This script performs comprehensive load testing of the deployed LLM API
to validate performance, scalability, and reliability under various load conditions.
"""

import asyncio
import aiohttp
import time
import json
import argparse
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadTester:
    def __init__(self, base_url: str, concurrent_users: int = 10):
        """Initialize the load tester."""
        self.base_url = base_url.rstrip('/')
        self.concurrent_users = concurrent_users
        self.results = []
        self.errors = []
        
        # Test prompts of varying complexity
        self.test_prompts = [
            "What is artificial intelligence?",
            "Explain the concept of machine learning in simple terms.",
            "Write a short story about a robot learning to paint.",
            "Describe the process of photosynthesis in plants.",
            "What are the main differences between Python and JavaScript?",
            "Explain quantum computing and its potential applications.",
            "Write a poem about the beauty of mathematics.",
            "Describe the history and evolution of the internet.",
            "What are the ethical implications of artificial intelligence?",
            "Explain the theory of relativity in layman's terms."
        ]
    
    async def make_request(self, session: aiohttp.ClientSession, prompt: str, user_id: int) -> Dict[str, Any]:
        """Make a single API request."""
        start_time = time.time()
        
        payload = {
            "prompt": prompt,
            "max_tokens": 150,
            "temperature": 0.8,
            "top_p": 0.95
        }
        
        try:
            async with session.post(
                f"{self.base_url}/generate",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'user_id': user_id,
                        'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        'status': 'success',
                        'response_time': response_time,
                        'tokens_generated': data.get('tokens_generated', 0),
                        'response_length': len(data.get('response', '')),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_text = await response.text()
                    return {
                        'user_id': user_id,
                        'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                        'status': 'error',
                        'response_time': response_time,
                        'error_code': response.status,
                        'error_message': error_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    
        except asyncio.TimeoutError:
            return {
                'user_id': user_id,
                'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                'status': 'timeout',
                'response_time': 30.0,
                'error_message': 'Request timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'user_id': user_id,
                'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                'status': 'error',
                'response_time': time.time() - start_time,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def user_simulation(self, session: aiohttp.ClientSession, user_id: int, num_requests: int):
        """Simulate a single user making multiple requests."""
        logger.info(f"Starting user {user_id} simulation with {num_requests} requests")
        
        for i in range(num_requests):
            # Select a random prompt
            prompt = self.test_prompts[i % len(self.test_prompts)]
            
            # Make request
            result = await self.make_request(session, prompt, user_id)
            self.results.append(result)
            
            if result['status'] != 'success':
                self.errors.append(result)
                logger.warning(f"User {user_id} request {i+1} failed: {result.get('error_message', 'Unknown error')}")
            else:
                logger.debug(f"User {user_id} request {i+1} completed in {result['response_time']:.2f}s")
            
            # Small delay between requests to simulate realistic usage
            await asyncio.sleep(0.5)
    
    async def run_load_test(self, total_requests: int, duration_seconds: int = None):
        """Run the load test with specified parameters."""
        logger.info(f"Starting load test with {self.concurrent_users} concurrent users")
        logger.info(f"Target: {total_requests} total requests")
        
        if duration_seconds:
            logger.info(f"Duration: {duration_seconds} seconds")
        
        start_time = time.time()
        
        # Calculate requests per user
        requests_per_user = total_requests // self.concurrent_users
        
        # Create HTTP session with connection pooling
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=50)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Test API health first
            health_check = await self.check_api_health(session)
            if not health_check:
                logger.error("API health check failed. Aborting load test.")
                return
            
            # Create tasks for concurrent users
            tasks = []
            for user_id in range(self.concurrent_users):
                task = asyncio.create_task(
                    self.user_simulation(session, user_id, requests_per_user)
                )
                tasks.append(task)
            
            # Wait for all tasks to complete or timeout
            if duration_seconds:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=duration_seconds
                    )
                except asyncio.TimeoutError:
                    logger.info(f"Load test stopped after {duration_seconds} seconds")
                    # Cancel remaining tasks
                    for task in tasks:
                        if not task.done():
                            task.cancel()
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        logger.info(f"Load test completed in {total_duration:.2f} seconds")
        
        # Generate and save results
        self.analyze_results(total_duration)
    
    async def check_api_health(self, session: aiohttp.ClientSession) -> bool:
        """Check if the API is healthy before starting load test."""
        try:
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('model_loaded', False):
                        logger.info("API health check passed - model is loaded and ready")
                        return True
                    else:
                        logger.error("API health check failed - model not loaded")
                        return False
                else:
                    logger.error(f"API health check failed - status code: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"API health check failed - error: {e}")
            return False
    
    def analyze_results(self, total_duration: float):
        """Analyze and report load test results."""
        if not self.results:
            logger.error("No results to analyze")
            return
        
        # Separate successful and failed requests
        successful_requests = [r for r in self.results if r['status'] == 'success']
        failed_requests = [r for r in self.results if r['status'] != 'success']
        
        # Calculate metrics
        total_requests = len(self.results)
        success_rate = len(successful_requests) / total_requests * 100
        error_rate = len(failed_requests) / total_requests * 100
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            p99_response_time = sorted(response_times)[int(len(response_times) * 0.99)]
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            
            total_tokens = sum(r.get('tokens_generated', 0) for r in successful_requests)
            avg_tokens_per_request = total_tokens / len(successful_requests)
            tokens_per_second = total_tokens / total_duration
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = 0
            total_tokens = avg_tokens_per_request = tokens_per_second = 0
        
        requests_per_second = total_requests / total_duration
        
        # Generate report
        report = {
            'test_summary': {
                'total_requests': total_requests,
                'successful_requests': len(successful_requests),
                'failed_requests': len(failed_requests),
                'success_rate_percent': round(success_rate, 2),
                'error_rate_percent': round(error_rate, 2),
                'total_duration_seconds': round(total_duration, 2),
                'requests_per_second': round(requests_per_second, 2),
                'concurrent_users': self.concurrent_users
            },
            'performance_metrics': {
                'avg_response_time_seconds': round(avg_response_time, 3),
                'median_response_time_seconds': round(median_response_time, 3),
                'p95_response_time_seconds': round(p95_response_time, 3),
                'p99_response_time_seconds': round(p99_response_time, 3),
                'min_response_time_seconds': round(min_response_time, 3),
                'max_response_time_seconds': round(max_response_time, 3)
            },
            'token_metrics': {
                'total_tokens_generated': total_tokens,
                'avg_tokens_per_request': round(avg_tokens_per_request, 1),
                'tokens_per_second': round(tokens_per_second, 1)
            },
            'error_analysis': self.analyze_errors()
        }
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        with open(f'load_test_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save detailed results
        with open(f'load_test_results_{timestamp}.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate visualizations
        self.create_visualizations(timestamp)
        
        # Print summary
        self.print_summary(report)
        
        logger.info(f"Detailed results saved to load_test_results_{timestamp}.json")
        logger.info(f"Report saved to load_test_report_{timestamp}.json")
    
    def analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        if not self.errors:
            return {'total_errors': 0, 'error_types': {}}
        
        error_types = {}
        for error in self.errors:
            error_type = error.get('status', 'unknown')
            if error_type not in error_types:
                error_types[error_type] = 0
            error_types[error_type] += 1
        
        return {
            'total_errors': len(self.errors),
            'error_types': error_types,
            'sample_errors': self.errors[:5]  # First 5 errors for debugging
        }
    
    def create_visualizations(self, timestamp: str):
        """Create performance visualization charts."""
        try:
            successful_requests = [r for r in self.results if r['status'] == 'success']
            
            if not successful_requests:
                logger.warning("No successful requests to visualize")
                return
            
            # Response time distribution
            response_times = [r['response_time'] for r in successful_requests]
            
            plt.figure(figsize=(12, 8))
            
            # Response time histogram
            plt.subplot(2, 2, 1)
            plt.hist(response_times, bins=30, alpha=0.7, color='blue')
            plt.title('Response Time Distribution')
            plt.xlabel('Response Time (seconds)')
            plt.ylabel('Frequency')
            
            # Response time over time
            plt.subplot(2, 2, 2)
            plt.plot(range(len(response_times)), response_times, alpha=0.7)
            plt.title('Response Time Over Time')
            plt.xlabel('Request Number')
            plt.ylabel('Response Time (seconds)')
            
            # Tokens generated distribution
            tokens_generated = [r.get('tokens_generated', 0) for r in successful_requests]
            plt.subplot(2, 2, 3)
            plt.hist(tokens_generated, bins=20, alpha=0.7, color='green')
            plt.title('Tokens Generated Distribution')
            plt.xlabel('Tokens Generated')
            plt.ylabel('Frequency')
            
            # Success rate over time (sliding window)
            window_size = max(10, len(self.results) // 20)
            success_rates = []
            for i in range(window_size, len(self.results)):
                window = self.results[i-window_size:i]
                success_count = sum(1 for r in window if r['status'] == 'success')
                success_rates.append(success_count / window_size * 100)
            
            plt.subplot(2, 2, 4)
            plt.plot(range(len(success_rates)), success_rates, alpha=0.7, color='red')
            plt.title(f'Success Rate Over Time (Window: {window_size})')
            plt.xlabel('Request Window')
            plt.ylabel('Success Rate (%)')
            plt.ylim(0, 105)
            
            plt.tight_layout()
            plt.savefig(f'load_test_charts_{timestamp}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to load_test_charts_{timestamp}.png")
            
        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a formatted summary of the load test results."""
        print("\n" + "="*80)
        print("LOAD TEST RESULTS SUMMARY")
        print("="*80)
        
        summary = report['test_summary']
        performance = report['performance_metrics']
        tokens = report['token_metrics']
        
        print(f"Total Requests:        {summary['total_requests']:,}")
        print(f"Successful Requests:   {summary['successful_requests']:,}")
        print(f"Failed Requests:       {summary['failed_requests']:,}")
        print(f"Success Rate:          {summary['success_rate_percent']:.2f}%")
        print(f"Error Rate:            {summary['error_rate_percent']:.2f}%")
        print(f"Test Duration:         {summary['total_duration_seconds']:.2f} seconds")
        print(f"Requests/Second:       {summary['requests_per_second']:.2f}")
        print(f"Concurrent Users:      {summary['concurrent_users']}")
        
        print("\nPERFORMANCE METRICS:")
        print(f"Average Response Time: {performance['avg_response_time_seconds']:.3f}s")
        print(f"Median Response Time:  {performance['median_response_time_seconds']:.3f}s")
        print(f"95th Percentile:       {performance['p95_response_time_seconds']:.3f}s")
        print(f"99th Percentile:       {performance['p99_response_time_seconds']:.3f}s")
        print(f"Min Response Time:     {performance['min_response_time_seconds']:.3f}s")
        print(f"Max Response Time:     {performance['max_response_time_seconds']:.3f}s")
        
        print("\nTOKEN METRICS:")
        print(f"Total Tokens Generated: {tokens['total_tokens_generated']:,}")
        print(f"Avg Tokens/Request:     {tokens['avg_tokens_per_request']:.1f}")
        print(f"Tokens/Second:          {tokens['tokens_per_second']:.1f}")
        
        if report['error_analysis']['total_errors'] > 0:
            print("\nERROR ANALYSIS:")
            print(f"Total Errors: {report['error_analysis']['total_errors']}")
            for error_type, count in report['error_analysis']['error_types'].items():
                print(f"  {error_type}: {count}")
        
        print("="*80)

async def main():
    parser = argparse.ArgumentParser(description="Load test the LLM API")
    parser.add_argument("--url", required=True, help="Base URL of the API")
    parser.add_argument("--concurrent", type=int, default=10, help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=100, help="Total number of requests")
    parser.add_argument("--duration", type=int, help="Test duration in seconds (optional)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize load tester
    tester = LoadTester(args.url, args.concurrent)
    
    # Run load test
    await tester.run_load_test(args.requests, args.duration)

if __name__ == "__main__":
    asyncio.run(main())
