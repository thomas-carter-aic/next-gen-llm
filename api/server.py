#!/usr/bin/env python3
"""
Production API Server for Custom LLM
Document ID: api_server_20250705_070000
Created: July 5, 2025 07:00:00 UTC

FastAPI-based production server for serving the custom LLM with
enterprise features including rate limiting, monitoring, and auto-scaling.
"""

import os
import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import boto3
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import psutil
import GPUtil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/var/log/llm-api.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for model and tokenizer
model = None
tokenizer = None
generation_config = None
model_loaded = False
request_count = 0
start_time = time.time()

# AWS clients
cloudwatch = boto3.client('cloudwatch', region_name=os.environ.get('AWS_REGION', 'us-west-2'))
s3_client = boto3.client('s3', region_name=os.environ.get('AWS_REGION', 'us-west-2'))

# Pydantic models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="Input prompt for text generation")
    max_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.1, le=1.0, description="Top-p (nucleus) sampling parameter")
    top_k: int = Field(default=50, ge=1, le=100, description="Top-k sampling parameter")
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0, description="Repetition penalty")
    do_sample: bool = Field(default=True, description="Whether to use sampling")
    stream: bool = Field(default=False, description="Whether to stream the response")

class GenerationResponse(BaseModel):
    response: str = Field(..., description="Generated text response")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    request_id: str = Field(..., description="Unique request identifier")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime: float = Field(..., description="Server uptime in seconds")
    requests_served: int = Field(..., description="Total requests served")
    system_info: Dict[str, Any] = Field(..., description="System information")

class ModelInfo(BaseModel):
    model_name: str = Field(..., description="Model name")
    model_size: str = Field(..., description="Model size")
    parameters: int = Field(..., description="Number of parameters")
    context_length: int = Field(..., description="Maximum context length")
    loaded_at: str = Field(..., description="Model load timestamp")

# Initialize FastAPI app
app = FastAPI(
    title="Nexus LLM API",
    description="Production API for custom LLM serving",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Rate limiting (simple in-memory implementation)
request_times = {}
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit."""
    current_time = time.time()
    
    if client_ip not in request_times:
        request_times[client_ip] = []
    
    # Remove old requests outside the window
    request_times[client_ip] = [
        req_time for req_time in request_times[client_ip]
        if current_time - req_time < RATE_LIMIT_WINDOW
    ]
    
    # Check if limit exceeded
    if len(request_times[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False
    
    # Add current request
    request_times[client_ip].append(current_time)
    return True

async def get_client_ip(request: Request) -> str:
    """Extract client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host

async def load_model():
    """Load the model and tokenizer."""
    global model, tokenizer, generation_config, model_loaded
    
    if model_loaded:
        return
    
    try:
        logger.info("Loading model and tokenizer...")
        
        model_path = os.environ.get("MODEL_PATH", "/opt/ml/model")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Set up generation config
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=512
        )
        
        model_loaded = True
        logger.info("Model loaded successfully")
        
        # Log model info
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {param_count:,}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def publish_metrics(response_time: float, tokens_generated: int, error_count: int = 0):
    """Publish custom metrics to CloudWatch."""
    try:
        metrics = [
            {
                'MetricName': 'ResponseTime',
                'Value': response_time,
                'Unit': 'Seconds',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'TokensGenerated',
                'Value': tokens_generated,
                'Unit': 'Count',
                'Timestamp': datetime.utcnow()
            },
            {
                'MetricName': 'RequestCount',
                'Value': 1,
                'Unit': 'Count',
                'Timestamp': datetime.utcnow()
            }
        ]
        
        if error_count > 0:
            metrics.append({
                'MetricName': 'ErrorCount',
                'Value': error_count,
                'Unit': 'Count',
                'Timestamp': datetime.utcnow()
            })
        
        cloudwatch.put_metric_data(
            Namespace='NexusLLM/API',
            MetricData=metrics
        )
        
    except Exception as e:
        logger.warning(f"Failed to publish metrics: {e}")

def get_system_info() -> Dict[str, Any]:
    """Get system information."""
    try:
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available = memory.available / (1024**3)  # GB
        
        # GPU info
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'id': gpu.id,
                    'name': gpu.name,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'memory_percent': gpu.memoryUtil * 100,
                    'temperature': gpu.temperature,
                    'load': gpu.load * 100
                })
        except Exception:
            gpu_info = []
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'memory_percent': memory_percent,
            'memory_available_gb': memory_available,
            'gpu_info': gpu_info
        }
        
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {}

@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    logger.info("Starting Nexus LLM API server...")
    await load_model()
    logger.info("API server startup completed")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global request_count, start_time
    
    uptime = time.time() - start_time
    system_info = get_system_info()
    
    return HealthResponse(
        status="healthy" if model_loaded else "loading",
        model_loaded=model_loaded,
        uptime=uptime,
        requests_served=request_count,
        system_info=system_info
    )

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    param_count = sum(p.numel() for p in model.parameters())
    
    return ModelInfo(
        model_name=os.environ.get("MODEL_NAME", "custom-llm"),
        model_size="7B",  # Adjust based on actual model
        parameters=param_count,
        context_length=2048,  # Adjust based on actual model
        loaded_at=datetime.utcnow().isoformat()
    )

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(
    request: GenerationRequest,
    background_tasks: BackgroundTasks,
    client_request: Request
):
    """Generate text using the loaded model."""
    global request_count
    
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Rate limiting
    client_ip = await get_client_ip(client_request)
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        logger.info(f"Processing generation request {request_id}")
        
        # Tokenize input
        inputs = tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - request.max_tokens
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Update generation config
        gen_config = GenerationConfig(
            do_sample=request.do_sample,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            max_new_tokens=request.max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=gen_config,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode response
        generated_tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        generation_time = time.time() - start_time
        tokens_generated = len(generated_tokens)
        
        # Update counters
        request_count += 1
        
        # Publish metrics in background
        background_tasks.add_task(
            publish_metrics,
            generation_time,
            tokens_generated,
            0
        )
        
        logger.info(f"Request {request_id} completed in {generation_time:.2f}s")
        
        return GenerationResponse(
            response=response_text,
            tokens_generated=tokens_generated,
            generation_time=generation_time,
            model_info={
                "model_name": os.environ.get("MODEL_NAME", "custom-llm"),
                "parameters": sum(p.numel() for p in model.parameters())
            },
            request_id=request_id
        )
        
    except Exception as e:
        generation_time = time.time() - start_time
        logger.error(f"Generation failed for request {request_id}: {e}")
        
        # Publish error metrics
        background_tasks.add_task(
            publish_metrics,
            generation_time,
            0,
            1
        )
        
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get server metrics."""
    global request_count, start_time
    
    uptime = time.time() - start_time
    system_info = get_system_info()
    
    return {
        "uptime_seconds": uptime,
        "requests_served": request_count,
        "requests_per_second": request_count / uptime if uptime > 0 else 0,
        "model_loaded": model_loaded,
        "system_info": system_info
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    # Configuration
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8080))
    workers = int(os.environ.get("WORKERS", 1))
    
    # Run server
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True,
        reload=False
    )
