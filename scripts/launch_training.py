#!/usr/bin/env python3
"""
AWS SageMaker Training Launch Script
Document ID: launch_training_script_20250705_070000
Created: July 5, 2025 07:00:00 UTC
"""

import boto3
import json
import os
import time
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMTrainingLauncher:
    def __init__(self, config_file='aws_config.json'):
        """Initialize the training launcher with AWS configuration."""
        self.config = self.load_config(config_file)
        self.sagemaker = boto3.client('sagemaker', region_name=self.config['region'])
        self.s3 = boto3.client('s3', region_name=self.config['region'])
        self.sts = boto3.client('sts')
        
        # Get account ID
        self.account_id = self.sts.get_caller_identity()['Account']
        
    def load_config(self, config_file):
        """Load AWS configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            raise
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file {config_file}")
            raise
    
    def prepare_training_data(self):
        """Prepare and validate training data in S3."""
        logger.info("Preparing training data...")
        
        data_bucket = self.config['buckets']['data']
        
        # Check if processed data exists
        try:
            response = self.s3.list_objects_v2(
                Bucket=data_bucket,
                Prefix='processed/',
                MaxKeys=1
            )
            
            if 'Contents' not in response:
                logger.warning("No processed data found. Run data preprocessing first.")
                return False
                
            logger.info("Training data validated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error checking training data: {e}")
            return False
    
    def create_training_job_config(self, model_name, hyperparameters):
        """Create SageMaker training job configuration."""
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"{self.config['project']}-{model_name}-{timestamp}"
        
        # ECR image URI for training
        ecr_uri = f"{self.account_id}.dkr.ecr.{self.config['region']}.amazonaws.com/{self.config['project']}/training:latest"
        
        # IAM role ARN
        role_arn = f"arn:aws:iam::{self.account_id}:role/{self.config['project']}-sagemaker-role"
        
        config = {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": ecr_uri,
                "TrainingInputMode": "File"
            },
            "RoleArn": role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "training",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": f"s3://{self.config['buckets']['data']}/processed/",
                            "S3DataDistributionType": "FullyReplicated"
                        }
                    },
                    "ContentType": "application/json",
                    "CompressionType": "None"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": f"s3://{self.config['buckets']['models']}/training-output/"
            },
            "ResourceConfig": {
                "InstanceType": hyperparameters.get("instance_type", "ml.p4d.24xlarge"),
                "InstanceCount": 1,
                "VolumeSizeInGB": hyperparameters.get("volume_size", 1000)
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": hyperparameters.get("max_runtime", 259200)  # 72 hours
            },
            "HyperParameters": {
                "epochs": str(hyperparameters.get("epochs", 3)),
                "batch_size": str(hyperparameters.get("batch_size", 4)),
                "learning_rate": str(hyperparameters.get("learning_rate", 2e-5)),
                "model_name": model_name,
                "max_seq_length": str(hyperparameters.get("max_seq_length", 2048))
            },
            "Tags": [
                {"Key": "Project", "Value": self.config['project']},
                {"Key": "Environment", "Value": "training"},
                {"Key": "Model", "Value": model_name}
            ]
        }
        
        # Add spot instance configuration if requested
        if hyperparameters.get("use_spot_instances", True):
            config["EnableManagedSpotTraining"] = True
            config["StoppingCondition"]["MaxWaitTimeInSeconds"] = hyperparameters.get("max_wait_time", 345600)  # 96 hours
        
        # Add checkpointing configuration
        if hyperparameters.get("enable_checkpointing", True):
            config["CheckpointConfig"] = {
                "S3Uri": f"s3://{self.config['buckets']['models']}/checkpoints/{job_name}/",
                "LocalPath": "/opt/ml/checkpoints"
            }
        
        return config
    
    def launch_training_job(self, config):
        """Launch the SageMaker training job."""
        try:
            logger.info(f"Launching training job: {config['TrainingJobName']}")
            
            response = self.sagemaker.create_training_job(**config)
            
            logger.info(f"Training job launched successfully")
            logger.info(f"Job ARN: {response['TrainingJobArn']}")
            
            return config['TrainingJobName']
            
        except Exception as e:
            logger.error(f"Failed to launch training job: {e}")
            raise
    
    def monitor_training_job(self, job_name, poll_interval=60):
        """Monitor the training job progress."""
        logger.info(f"Monitoring training job: {job_name}")
        
        while True:
            try:
                response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                
                logger.info(f"Training job status: {status}")
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    break
                
                # Log additional details if available
                if 'SecondaryStatus' in response:
                    logger.info(f"Secondary status: {response['SecondaryStatus']}")
                
                if 'TrainingStartTime' in response and 'TrainingEndTime' not in response:
                    start_time = response['TrainingStartTime']
                    elapsed = datetime.now(start_time.tzinfo) - start_time
                    logger.info(f"Training time elapsed: {elapsed}")
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Error monitoring training job: {e}")
                break
        
        # Final status
        try:
            response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            final_status = response['TrainingJobStatus']
            
            logger.info(f"Training job final status: {final_status}")
            
            if final_status == 'Completed':
                logger.info(f"Model artifacts location: {response['ModelArtifacts']['S3ModelArtifacts']}")
                return True
            else:
                if 'FailureReason' in response:
                    logger.error(f"Training failed: {response['FailureReason']}")
                return False
                
        except Exception as e:
            logger.error(f"Error getting final training status: {e}")
            return False
    
    def create_model_endpoint_config(self, job_name, model_name):
        """Create model and endpoint configuration for deployment."""
        try:
            # Get training job details
            training_job = self.sagemaker.describe_training_job(TrainingJobName=job_name)
            model_artifacts = training_job['ModelArtifacts']['S3ModelArtifacts']
            
            # Create model
            model_config = {
                "ModelName": f"{model_name}-model",
                "PrimaryContainer": {
                    "Image": f"{self.account_id}.dkr.ecr.{self.config['region']}.amazonaws.com/{self.config['project']}/api:latest",
                    "ModelDataUrl": model_artifacts,
                    "Environment": {
                        "MODEL_NAME": model_name,
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code"
                    }
                },
                "ExecutionRoleArn": f"arn:aws:iam::{self.account_id}:role/{self.config['project']}-sagemaker-role",
                "Tags": [
                    {"Key": "Project", "Value": self.config['project']},
                    {"Key": "Environment", "Value": "production"}
                ]
            }
            
            self.sagemaker.create_model(**model_config)
            logger.info(f"Model created: {model_name}-model")
            
            # Create endpoint configuration
            endpoint_config = {
                "EndpointConfigName": f"{model_name}-endpoint-config",
                "ProductionVariants": [
                    {
                        "VariantName": "primary",
                        "ModelName": f"{model_name}-model",
                        "InitialInstanceCount": 1,
                        "InstanceType": "ml.g4dn.xlarge",
                        "InitialVariantWeight": 1.0
                    }
                ],
                "Tags": [
                    {"Key": "Project", "Value": self.config['project']},
                    {"Key": "Environment", "Value": "production"}
                ]
            }
            
            self.sagemaker.create_endpoint_config(**endpoint_config)
            logger.info(f"Endpoint configuration created: {model_name}-endpoint-config")
            
            return f"{model_name}-endpoint-config"
            
        except Exception as e:
            logger.error(f"Error creating model/endpoint config: {e}")
            raise
    
    def save_training_metadata(self, job_name, config, model_name):
        """Save training metadata for future reference."""
        metadata = {
            "job_name": job_name,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "status": "completed"
        }
        
        metadata_file = f"training_metadata_{job_name}.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Upload to S3
        try:
            self.s3.upload_file(
                metadata_file,
                self.config['buckets']['models'],
                f"metadata/{metadata_file}"
            )
            logger.info(f"Training metadata saved to S3: metadata/{metadata_file}")
        except Exception as e:
            logger.warning(f"Failed to upload metadata to S3: {e}")
        
        return metadata_file

def main():
    parser = argparse.ArgumentParser(description="Launch LLM training on AWS SageMaker")
    parser.add_argument("--model-name", default="llama-3-1-finetuned", help="Model name")
    parser.add_argument("--config", default="aws_config.json", help="AWS configuration file")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--instance-type", default="ml.p4d.24xlarge", help="SageMaker instance type")
    parser.add_argument("--use-spot", action="store_true", default=True, help="Use spot instances")
    parser.add_argument("--monitor", action="store_true", default=True, help="Monitor training progress")
    parser.add_argument("--create-endpoint", action="store_true", help="Create endpoint after training")
    
    args = parser.parse_args()
    
    # Initialize launcher
    launcher = LLMTrainingLauncher(args.config)
    
    # Prepare hyperparameters
    hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "instance_type": args.instance_type,
        "use_spot_instances": args.use_spot,
        "max_runtime": 259200,  # 72 hours
        "max_wait_time": 345600,  # 96 hours for spot
        "volume_size": 1000,
        "max_seq_length": 2048,
        "enable_checkpointing": True
    }
    
    try:
        # Validate training data
        if not launcher.prepare_training_data():
            logger.error("Training data preparation failed")
            return 1
        
        # Create training configuration
        config = launcher.create_training_job_config(args.model_name, hyperparameters)
        
        # Launch training job
        job_name = launcher.launch_training_job(config)
        
        # Monitor training if requested
        if args.monitor:
            success = launcher.monitor_training_job(job_name)
            
            if success:
                logger.info("Training completed successfully")
                
                # Save metadata
                launcher.save_training_metadata(job_name, config, args.model_name)
                
                # Create endpoint configuration if requested
                if args.create_endpoint:
                    endpoint_config = launcher.create_model_endpoint_config(job_name, args.model_name)
                    logger.info(f"Endpoint configuration created: {endpoint_config}")
                
                return 0
            else:
                logger.error("Training failed")
                return 1
        else:
            logger.info(f"Training job launched: {job_name}")
            logger.info("Use --monitor flag to track progress")
            return 0
            
    except Exception as e:
        logger.error(f"Training launch failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
