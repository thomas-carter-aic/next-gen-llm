#!/usr/bin/env python3
"""
Comprehensive API Testing Suite
Document ID: api_test_suite_20250705_080000
Created: July 5, 2025 08:00:00 UTC

This module provides comprehensive testing for the LLM API including
unit tests, integration tests, and performance tests.
"""

import pytest
import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestLLMAPI:
    """Test suite for LLM API endpoints."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for API testing."""
        return "http://localhost:8080"  # Adjust for your deployment
    
    @pytest.fixture
    async def http_session(self):
        """HTTP session for async requests."""
        async with aiohttp.ClientSession() as session:
            yield session
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, http_session, api_base_url):
        """Test the health check endpoint."""
        async with http_session.get(f"{api_base_url}/health") as response:
            assert response.status == 200
            data = await response.json()
            
            # Check required fields
            assert "status" in data
            assert "model_loaded" in data
            assert "uptime" in data
            assert "requests_served" in data
            assert "system_info" in data
            
            # Check data types
            assert isinstance(data["status"], str)
            assert isinstance(data["model_loaded"], bool)
            assert isinstance(data["uptime"], (int, float))
            assert isinstance(data["requests_served"], int)
            assert isinstance(data["system_info"], dict)
    
    @pytest.mark.asyncio
    async def test_model_info_endpoint(self, http_session, api_base_url):
        """Test the model info endpoint."""
        async with http_session.get(f"{api_base_url}/model/info") as response:
            if response.status == 503:
                # Model not loaded yet
                pytest.skip("Model not loaded")
            
            assert response.status == 200
            data = await response.json()
            
            # Check required fields
            required_fields = ["model_name", "model_size", "parameters", "context_length", "loaded_at"]
            for field in required_fields:
                assert field in data
            
            # Check data types
            assert isinstance(data["model_name"], str)
            assert isinstance(data["model_size"], str)
            assert isinstance(data["parameters"], int)
            assert isinstance(data["context_length"], int)
            assert isinstance(data["loaded_at"], str)
    
    @pytest.mark.asyncio
    async def test_generate_endpoint_basic(self, http_session, api_base_url):
        """Test basic text generation."""
        payload = {
            "prompt": "What is artificial intelligence?",
            "max_tokens": 50,
            "temperature": 0.8
        }
        
        async with http_session.post(
            f"{api_base_url}/generate",
            json=payload
        ) as response:
            if response.status == 503:
                pytest.skip("Model not loaded")
            
            assert response.status == 200
            data = await response.json()
            
            # Check required fields
            required_fields = ["response", "tokens_generated", "generation_time", "model_info", "request_id"]
            for field in required_fields:
                assert field in data
            
            # Check data types and values
            assert isinstance(data["response"], str)
            assert isinstance(data["tokens_generated"], int)
            assert isinstance(data["generation_time"], (int, float))
            assert isinstance(data["model_info"], dict)
            assert isinstance(data["request_id"], str)
            
            # Check response quality
            assert len(data["response"]) > 0
            assert data["tokens_generated"] > 0
            assert data["generation_time"] > 0
    
    @pytest.mark.asyncio
    async def test_generate_endpoint_parameters(self, http_session, api_base_url):
        """Test text generation with various parameters."""
        test_cases = [
            {
                "prompt": "Explain machine learning",
                "max_tokens": 100,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 40
            },
            {
                "prompt": "Write a haiku about technology",
                "max_tokens": 30,
                "temperature": 1.0,
                "top_p": 0.95,
                "repetition_penalty": 1.2
            }
        ]
        
        for payload in test_cases:
            async with http_session.post(
                f"{api_base_url}/generate",
                json=payload
            ) as response:
                if response.status == 503:
                    pytest.skip("Model not loaded")
                
                assert response.status == 200
                data = await response.json()
                
                # Verify response structure
                assert "response" in data
                assert "tokens_generated" in data
                assert len(data["response"]) > 0
                assert data["tokens_generated"] <= payload["max_tokens"]
    
    @pytest.mark.asyncio
    async def test_generate_endpoint_validation(self, http_session, api_base_url):
        """Test input validation for generate endpoint."""
        invalid_payloads = [
            {},  # Missing prompt
            {"prompt": ""},  # Empty prompt
            {"prompt": "test", "max_tokens": 0},  # Invalid max_tokens
            {"prompt": "test", "temperature": -1},  # Invalid temperature
            {"prompt": "test", "top_p": 1.5},  # Invalid top_p
            {"prompt": "A" * 20000}  # Prompt too long
        ]
        
        for payload in invalid_payloads:
            async with http_session.post(
                f"{api_base_url}/generate",
                json=payload
            ) as response:
                assert response.status == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, http_session, api_base_url):
        """Test rate limiting functionality."""
        # Make rapid requests to trigger rate limiting
        tasks = []
        for i in range(150):  # Exceed rate limit
            task = asyncio.create_task(
                self._make_generate_request(http_session, api_base_url, f"Test prompt {i}")
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that some requests were rate limited
        rate_limited_count = sum(1 for r in responses if isinstance(r, dict) and r.get('status') == 429)
        assert rate_limited_count > 0, "Rate limiting should have been triggered"
    
    async def _make_generate_request(self, session, base_url, prompt):
        """Helper method to make a generate request."""
        payload = {"prompt": prompt, "max_tokens": 10}
        try:
            async with session.post(f"{base_url}/generate", json=payload) as response:
                return {"status": response.status, "data": await response.json()}
        except Exception as e:
            return {"error": str(e)}
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, http_session, api_base_url):
        """Test handling of concurrent requests."""
        num_concurrent = 10
        prompts = [f"Tell me about topic {i}" for i in range(num_concurrent)]
        
        # Create concurrent tasks
        tasks = []
        for prompt in prompts:
            payload = {"prompt": prompt, "max_tokens": 50}
            task = asyncio.create_task(
                http_session.post(f"{api_base_url}/generate", json=payload)
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)
        
        # Check that all requests succeeded
        successful_responses = 0
        for response in responses:
            if response.status == 200:
                successful_responses += 1
            elif response.status == 503:
                pytest.skip("Model not loaded")
        
        # At least 80% should succeed under normal conditions
        success_rate = successful_responses / len(responses)
        assert success_rate >= 0.8, f"Success rate too low: {success_rate}"
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self, http_session, api_base_url):
        """Test response time performance."""
        test_prompts = [
            "What is AI?",  # Short
            "Explain the concept of machine learning in detail.",  # Medium
            "Write a comprehensive essay about the future of artificial intelligence and its impact on society."  # Long
        ]
        
        response_times = []
        
        for prompt in test_prompts:
            payload = {"prompt": prompt, "max_tokens": 100}
            
            start_time = time.time()
            async with http_session.post(
                f"{api_base_url}/generate",
                json=payload
            ) as response:
                if response.status == 503:
                    pytest.skip("Model not loaded")
                
                assert response.status == 200
                data = await response.json()
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # Check that reported generation time is reasonable
                reported_time = data["generation_time"]
                assert abs(response_time - reported_time) < 1.0, "Reported time should be close to actual time"
        
        # Check performance targets
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        assert avg_response_time < 5.0, f"Average response time too high: {avg_response_time}s"
        assert max_response_time < 10.0, f"Maximum response time too high: {max_response_time}s"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, http_session, api_base_url):
        """Test error handling and recovery."""
        # Test malformed JSON
        async with http_session.post(
            f"{api_base_url}/generate",
            data="invalid json"
        ) as response:
            assert response.status == 422
        
        # Test unsupported content type
        async with http_session.post(
            f"{api_base_url}/generate",
            data="prompt=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        ) as response:
            assert response.status in [422, 415]  # Unprocessable Entity or Unsupported Media Type
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, http_session, api_base_url):
        """Test metrics endpoint."""
        async with http_session.get(f"{api_base_url}/metrics") as response:
            assert response.status == 200
            data = await response.json()
            
            # Check required metrics
            required_fields = ["uptime_seconds", "requests_served", "requests_per_second", "model_loaded", "system_info"]
            for field in required_fields:
                assert field in data
            
            # Check data types
            assert isinstance(data["uptime_seconds"], (int, float))
            assert isinstance(data["requests_served"], int)
            assert isinstance(data["requests_per_second"], (int, float))
            assert isinstance(data["model_loaded"], bool)
            assert isinstance(data["system_info"], dict)

class TestAPIIntegration:
    """Integration tests for the complete API system."""
    
    @pytest.fixture
    def api_base_url(self):
        """Base URL for integration testing."""
        return "http://localhost:8080"
    
    @pytest.mark.asyncio
    async def test_full_conversation_flow(self, api_base_url):
        """Test a complete conversation flow."""
        conversation_prompts = [
            "Hello, can you introduce yourself?",
            "What can you help me with?",
            "Explain quantum computing in simple terms.",
            "Thank you for the explanation."
        ]
        
        async with aiohttp.ClientSession() as session:
            conversation_history = []
            
            for prompt in conversation_prompts:
                payload = {
                    "prompt": prompt,
                    "max_tokens": 150,
                    "temperature": 0.8
                }
                
                async with session.post(f"{api_base_url}/generate", json=payload) as response:
                    if response.status == 503:
                        pytest.skip("Model not loaded")
                    
                    assert response.status == 200
                    data = await response.json()
                    
                    conversation_turn = {
                        "prompt": prompt,
                        "response": data["response"],
                        "tokens": data["tokens_generated"],
                        "time": data["generation_time"]
                    }
                    conversation_history.append(conversation_turn)
            
            # Verify conversation quality
            assert len(conversation_history) == len(conversation_prompts)
            
            # Check that responses are contextually appropriate
            for turn in conversation_history:
                assert len(turn["response"]) > 0
                assert turn["tokens"] > 0
                assert turn["time"] > 0
    
    @pytest.mark.asyncio
    async def test_system_recovery(self, api_base_url):
        """Test system recovery after errors."""
        async with aiohttp.ClientSession() as session:
            # First, make a successful request
            payload = {"prompt": "Test prompt", "max_tokens": 50}
            async with session.post(f"{api_base_url}/generate", json=payload) as response:
                if response.status == 503:
                    pytest.skip("Model not loaded")
                assert response.status == 200
            
            # Then make an invalid request
            invalid_payload = {"prompt": "", "max_tokens": -1}
            async with session.post(f"{api_base_url}/generate", json=invalid_payload) as response:
                assert response.status == 422
            
            # Verify system can still handle valid requests
            async with session.post(f"{api_base_url}/generate", json=payload) as response:
                assert response.status == 200

class TestAPIPerformance:
    """Performance tests for the API."""
    
    @pytest.fixture
    def api_base_url(self):
        return "http://localhost:8080"
    
    @pytest.mark.asyncio
    async def test_sustained_load(self, api_base_url):
        """Test sustained load handling."""
        duration_seconds = 60  # 1 minute test
        requests_per_second = 5
        total_requests = duration_seconds * requests_per_second
        
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            successful_requests = 0
            failed_requests = 0
            response_times = []
            
            for i in range(total_requests):
                payload = {
                    "prompt": f"Test prompt {i}",
                    "max_tokens": 50,
                    "temperature": 0.8
                }
                
                request_start = time.time()
                try:
                    async with session.post(f"{api_base_url}/generate", json=payload) as response:
                        request_end = time.time()
                        response_times.append(request_end - request_start)
                        
                        if response.status == 200:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                            
                except Exception:
                    failed_requests += 1
                
                # Control request rate
                elapsed = time.time() - start_time
                expected_requests = elapsed * requests_per_second
                if i > expected_requests:
                    await asyncio.sleep(0.1)
            
            # Analyze results
            total_time = time.time() - start_time
            actual_rps = total_requests / total_time
            success_rate = successful_requests / total_requests
            
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]
            else:
                avg_response_time = 0
                p95_response_time = 0
            
            # Performance assertions
            assert success_rate >= 0.95, f"Success rate too low: {success_rate}"
            assert avg_response_time < 3.0, f"Average response time too high: {avg_response_time}s"
            assert p95_response_time < 5.0, f"95th percentile response time too high: {p95_response_time}s"
            
            logger.info(f"Sustained load test results:")
            logger.info(f"  Total requests: {total_requests}")
            logger.info(f"  Successful: {successful_requests}")
            logger.info(f"  Failed: {failed_requests}")
            logger.info(f"  Success rate: {success_rate:.2%}")
            logger.info(f"  Actual RPS: {actual_rps:.2f}")
            logger.info(f"  Avg response time: {avg_response_time:.3f}s")
            logger.info(f"  95th percentile: {p95_response_time:.3f}s")

# Test configuration
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
