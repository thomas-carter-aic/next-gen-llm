# AWS Implementation Guide: Custom LLM Development

**Document ID**: 04_aws_implementation_guide_20250705_070000  
**Created**: July 5, 2025 07:00:00 UTC  
**Implementation Type**: Step-by-Step AWS Deployment  
**Target Audience**: DevOps Engineers, Technical Leads

## Prerequisites

### AWS Account Setup
- AWS Account with appropriate service limits
- IAM user with administrative privileges
- AWS CLI configured with credentials
- Terraform or CloudFormation for Infrastructure as Code

### Required Service Limits
```bash
# Check current limits
aws service-quotas get-service-quota --service-code ec2 --quota-code L-34B43A08  # p4d instances
aws service-quotas get-service-quota --service-code sagemaker --quota-code L-1194F27D  # Training instances
```

## Phase 1: Infrastructure Foundation (Week 1)

### Step 1: Core Infrastructure Setup

**1.1 Create S3 Buckets**
```bash
# Create primary data bucket
aws s3 mb s3://nexus-llm-data-$(date +%s) --region us-west-2

# Create model artifacts bucket
aws s3 mb s3://nexus-llm-models-$(date +%s) --region us-west-2

# Create logs bucket
aws s3 mb s3://nexus-llm-logs-$(date +%s) --region us-west-2

# Configure bucket policies and lifecycle
aws s3api put-bucket-lifecycle-configuration \
  --bucket nexus-llm-data-$(date +%s) \
  --lifecycle-configuration file://s3-lifecycle.json
```

**1.2 VPC and Networking**
```bash
# Create VPC using CloudFormation
aws cloudformation create-stack \
  --stack-name nexus-llm-vpc \
  --template-body file://cloudformation/vpc-template.yaml \
  --parameters ParameterKey=Environment,ParameterValue=production
```

**1.3 IAM Roles and Policies**
```bash
# Create SageMaker execution role
aws iam create-role \
  --role-name NexusLLMSageMakerRole \
  --assume-role-policy-document file://iam/sagemaker-trust-policy.json

# Attach policies
aws iam attach-role-policy \
  --role-name NexusLLMSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
  --role-name NexusLLMSageMakerRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### Step 2: Development Environment

**2.1 Launch Development Instance**
```bash
# Launch GPU-enabled development instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type p3.2xlarge \
  --key-name nexus-keypair \
  --security-group-ids sg-12345678 \
  --subnet-id subnet-12345678 \
  --user-data file://scripts/dev-setup.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=nexus-llm-dev}]'
```

**2.2 Development Environment Setup Script**
```bash
#!/bin/bash
# dev-setup.sh

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install NVIDIA drivers and CUDA
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.0-525.60.13-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Install Python and dependencies
sudo apt-get install -y python3-pip python3-venv
python3 -m venv /opt/nexus-llm
source /opt/nexus-llm/bin/activate

# Install ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate deepspeed
pip install sagemaker boto3 awscli
pip install wandb tensorboard

# Clone repository
git clone https://github.com/your-org/nexus-llm.git /opt/nexus-llm/code
```

## Phase 2: Data Pipeline Implementation (Weeks 2-3)

### Step 3: Data Acquisition and Processing

**3.1 Download Datasets to S3**
```python
# scripts/download_datasets.py
import boto3
import requests
from datasets import load_dataset

s3 = boto3.client('s3')
bucket_name = 'nexus-llm-data-{timestamp}'

def download_pile_dataset():
    """Download The Pile dataset to S3"""
    dataset = load_dataset("EleutherAI/pile", streaming=True)
    
    # Process in chunks to manage memory
    for i, batch in enumerate(dataset['train'].iter(batch_size=1000)):
        # Upload batch to S3
        key = f"raw/pile/batch_{i:06d}.jsonl"
        s3.put_object(
            Bucket=bucket_name,
            Key=key,
            Body='\n'.join([json.dumps(item) for item in batch])
        )
        
        if i % 100 == 0:
            print(f"Processed {i} batches")

def download_red_pajama():
    """Download Red Pajama dataset"""
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", streaming=True)
    
    for split in ['train']:
        for i, batch in enumerate(dataset[split].iter(batch_size=1000)):
            key = f"raw/red_pajama/{split}/batch_{i:06d}.jsonl"
            s3.put_object(
                Bucket=bucket_name,
                Key=key,
                Body='\n'.join([json.dumps(item) for item in batch])
            )

if __name__ == "__main__":
    download_pile_dataset()
    download_red_pajama()
```

**3.2 AWS Glue ETL Job**
```python
# glue_jobs/data_preprocessing.py
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from pyspark.sql.types import *

args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

def clean_text(text):
    """Clean and normalize text data"""
    if not text:
        return None
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Filter out very short or very long texts
    if len(text) < 50 or len(text) > 10000:
        return None
    
    # Basic quality filters
    if text.count('\n') / len(text) > 0.1:  # Too many line breaks
        return None
    
    return text

# Read raw data
df = spark.read.json(args['INPUT_PATH'])

# Apply cleaning transformations
cleaned_df = df.select(
    col("text"),
    col("meta")
).filter(
    col("text").isNotNull()
).withColumn(
    "cleaned_text", 
    udf(clean_text, StringType())(col("text"))
).filter(
    col("cleaned_text").isNotNull()
)

# Deduplicate based on text content
deduped_df = cleaned_df.dropDuplicates(["cleaned_text"])

# Write processed data
deduped_df.write.mode("overwrite").parquet(args['OUTPUT_PATH'])

job.commit()
```

**3.3 Launch Glue Job**
```bash
# Create and run Glue job
aws glue create-job \
  --name nexus-llm-preprocessing \
  --role arn:aws:iam::account:role/GlueServiceRole \
  --command '{
    "Name": "glueetl",
    "ScriptLocation": "s3://nexus-llm-code/glue_jobs/data_preprocessing.py"
  }' \
  --default-arguments '{
    "--INPUT_PATH": "s3://nexus-llm-data/raw/",
    "--OUTPUT_PATH": "s3://nexus-llm-data/processed/"
  }'

# Start job run
aws glue start-job-run --job-name nexus-llm-preprocessing
```

## Phase 3: Model Training with SageMaker (Weeks 4-6)

### Step 4: SageMaker Training Setup

**4.1 Create Training Container**
```dockerfile
# docker/training/Dockerfile
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

# Install additional dependencies
RUN pip install transformers==4.35.0 datasets==2.14.0 deepspeed==0.10.0 accelerate==0.24.0

# Copy training code
COPY train.py /opt/ml/code/train.py
COPY requirements.txt /opt/ml/code/requirements.txt

# Set environment variables
ENV SAGEMAKER_PROGRAM train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

WORKDIR /opt/ml/code
```

**4.2 Training Script**
```python
# train.py
import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import deepspeed

def main():
    # Parse SageMaker environment
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    training_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    
    # Load model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Load processed dataset
    dataset = load_from_disk(training_dir)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=2048
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{model_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=3,
        prediction_loss_only=True,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        deepspeed="ds_config.json",
        fp16=True,
        report_to=None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    main()
```

**4.3 DeepSpeed Configuration**
```json
{
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9
    },
    "fp16": {
        "enabled": true,
        "auto_cast": false,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "wall_clock_breakdown": false
}
```

**4.4 Launch SageMaker Training Job**
```python
# scripts/launch_training.py
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::account:role/NexusLLMSageMakerRole"

# Define estimator
estimator = PyTorch(
    entry_point="train.py",
    source_dir="training/",
    role=role,
    instance_type="ml.p4d.24xlarge",
    instance_count=1,
    framework_version="2.0.1",
    py_version="py310",
    volume_size=1000,
    max_run=259200,  # 72 hours
    use_spot_instances=True,
    max_wait=345600,  # 96 hours
    checkpoint_s3_uri="s3://nexus-llm-models/checkpoints/",
    hyperparameters={
        "epochs": 3,
        "batch_size": 4,
        "learning_rate": 2e-5
    }
)

# Start training
estimator.fit({
    "training": "s3://nexus-llm-data/processed/"
})
```

## Phase 4: Model Evaluation (Week 6-7)

### Step 5: Automated Evaluation Pipeline

**5.1 Evaluation Script**
```python
# evaluation/evaluate_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import evaluate
import json

def evaluate_model(model_path, test_datasets):
    """Comprehensive model evaluation"""
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    results = {}
    
    # GLUE benchmark evaluation
    glue_metric = evaluate.load("glue", "mrpc")
    glue_dataset = load_dataset("glue", "mrpc", split="test")
    
    # Generate predictions
    predictions = []
    for example in glue_dataset:
        inputs = tokenizer(example["sentence1"], example["sentence2"], 
                          return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1)
            predictions.append(prediction.item())
    
    # Calculate metrics
    glue_results = glue_metric.compute(predictions=predictions, 
                                      references=glue_dataset["label"])
    results["glue_mrpc"] = glue_results
    
    # Perplexity evaluation
    perplexity_metric = evaluate.load("perplexity", module_type="metric")
    test_texts = ["Sample text for perplexity evaluation..."]
    
    perplexity_results = perplexity_metric.compute(
        predictions=test_texts,
        model_id=model_path
    )
    results["perplexity"] = perplexity_results
    
    return results

def main():
    model_path = "s3://nexus-llm-models/llama-3-1-finetuned/"
    results = evaluate_model(model_path, ["glue", "superglue"])
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation completed. Results saved to evaluation_results.json")

if __name__ == "__main__":
    main()
```

## Phase 5: Production Deployment (Week 7-8)

### Step 6: ECS Deployment

**6.1 Create ECS Cluster**
```bash
# Create ECS cluster
aws ecs create-cluster \
  --cluster-name nexus-llm-production \
  --capacity-providers FARGATE \
  --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
```

**6.2 Task Definition**
```json
{
    "family": "nexus-llm-api",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "4096",
    "memory": "16384",
    "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "llm-api",
            "image": "account.dkr.ecr.us-west-2.amazonaws.com/nexus-llm:latest",
            "portMappings": [
                {
                    "containerPort": 8080,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {
                    "name": "MODEL_PATH",
                    "value": "s3://nexus-llm-models/llama-3-1-finetuned/"
                },
                {
                    "name": "MAX_CONCURRENT_REQUESTS",
                    "value": "100"
                }
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/nexus-llm-api",
                    "awslogs-region": "us-west-2",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

**6.3 API Server Implementation**
```python
# api/server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import os

app = FastAPI(title="Nexus LLM API", version="1.0.0")

# Global model and tokenizer
model = None
tokenizer = None

class GenerationRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95

class GenerationResponse(BaseModel):
    response: str
    tokens_generated: int

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    
    model_path = os.environ.get("MODEL_PATH", "/opt/ml/model")
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("Model loaded successfully")

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    try:
        # Tokenize input
        inputs = tokenizer(request.prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
        
        return GenerationResponse(
            response=response_text,
            tokens_generated=len(outputs[0]) - inputs.input_ids.shape[1]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Step 7: Load Balancer and Auto Scaling

**7.1 Application Load Balancer**
```bash
# Create ALB
aws elbv2 create-load-balancer \
  --name nexus-llm-alb \
  --subnets subnet-12345678 subnet-87654321 \
  --security-groups sg-12345678 \
  --scheme internet-facing \
  --type application

# Create target group
aws elbv2 create-target-group \
  --name nexus-llm-targets \
  --protocol HTTP \
  --port 8080 \
  --vpc-id vpc-12345678 \
  --target-type ip \
  --health-check-path /health
```

**7.2 Auto Scaling Configuration**
```bash
# Create ECS service with auto scaling
aws ecs create-service \
  --cluster nexus-llm-production \
  --service-name nexus-llm-api \
  --task-definition nexus-llm-api:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345678,subnet-87654321],securityGroups=[sg-12345678],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-west-2:account:targetgroup/nexus-llm-targets/1234567890123456,containerName=llm-api,containerPort=8080"

# Configure auto scaling
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/nexus-llm-production/nexus-llm-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10
```

## Phase 6: Monitoring and Optimization (Week 8)

### Step 8: CloudWatch Monitoring

**8.1 Custom Metrics**
```python
# monitoring/metrics.py
import boto3
import time
from datetime import datetime

cloudwatch = boto3.client('cloudwatch')

def publish_custom_metrics(response_time, tokens_generated, error_count=0):
    """Publish custom metrics to CloudWatch"""
    
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
            'MetricName': 'ErrorCount',
            'Value': error_count,
            'Unit': 'Count',
            'Timestamp': datetime.utcnow()
        }
    ]
    
    cloudwatch.put_metric_data(
        Namespace='NexusLLM/API',
        MetricData=metrics
    )
```

**8.2 CloudWatch Alarms**
```bash
# Create high latency alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "NexusLLM-HighLatency" \
  --alarm-description "Alert when response time exceeds 5 seconds" \
  --metric-name ResponseTime \
  --namespace NexusLLM/API \
  --statistic Average \
  --period 300 \
  --threshold 5.0 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 2

# Create error rate alarm
aws cloudwatch put-metric-alarm \
  --alarm-name "NexusLLM-HighErrorRate" \
  --alarm-description "Alert when error rate exceeds 5%" \
  --metric-name ErrorCount \
  --namespace NexusLLM/API \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1
```

## Cost Optimization and Management

### Cost Monitoring
```bash
# Set up cost budget
aws budgets create-budget \
  --account-id 123456789012 \
  --budget '{
    "BudgetName": "NexusLLM-Monthly",
    "BudgetLimit": {
      "Amount": "500",
      "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST"
  }'
```

### Resource Optimization
- Use Spot instances for training (70% cost savings)
- Implement S3 Intelligent Tiering
- Configure ECS auto-scaling based on demand
- Use Reserved Instances for predictable workloads

## Troubleshooting Guide

### Common Issues
1. **GPU Memory Errors**: Reduce batch size, enable gradient checkpointing
2. **Training Timeouts**: Increase max runtime, implement checkpointing
3. **API Latency**: Scale up ECS tasks, optimize model loading
4. **Cost Overruns**: Monitor usage, implement cost alerts

### Debugging Commands
```bash
# Check SageMaker training logs
aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs

# Monitor ECS service health
aws ecs describe-services --cluster nexus-llm-production --services nexus-llm-api

# Check CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace NexusLLM/API \
  --metric-name ResponseTime \
  --start-time 2025-07-05T00:00:00Z \
  --end-time 2025-07-05T23:59:59Z \
  --period 3600 \
  --statistics Average
```

## Next Steps

1. **Week 1**: Complete infrastructure setup and development environment
2. **Week 2-3**: Implement data pipeline and preprocessing
3. **Week 4-6**: Execute model training and optimization
4. **Week 7**: Deploy to production and conduct testing
5. **Week 8**: Monitoring setup and documentation completion

This comprehensive AWS implementation guide provides a production-ready path to deploying a custom LLM using AWS managed services while maintaining cost efficiency and scalability.

---

**Implementation Status**: Ready to Execute  
**Estimated Total Cost**: $700 development + $300/month production  
**Timeline**: 8 weeks to production deployment
