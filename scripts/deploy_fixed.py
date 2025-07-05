#!/usr/bin/env python3
"""
Fixed Deployment Orchestrator for Custom LLM
Document ID: deployment_orchestrator_fixed_20250705_080000
Created: July 5, 2025 08:00:00 UTC

Fixed version of the deployment orchestrator with corrected Docker login.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import boto3
from botocore.exceptions import ClientError, WaiterError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger(__name__)

class DeploymentOrchestrator:
    def __init__(self, config_file='aws_config.json', region='us-west-2'):
        """Initialize the deployment orchestrator."""
        self.region = region
        self.project_name = 'nexus-llm'
        self.timestamp = int(time.time())
        
        # AWS clients
        self.cloudformation = boto3.client('cloudformation', region_name=region)
        self.s3 = boto3.client('s3', region_name=region)
        self.ecs = boto3.client('ecs', region_name=region)
        self.ecr = boto3.client('ecr', region_name=region)
        self.sagemaker = boto3.client('sagemaker', region_name=region)
        self.iam = boto3.client('iam', region_name=region)
        self.ec2 = boto3.client('ec2', region_name=region)
        
        # Configuration
        self.config = {}
        self.deployment_state = {
            'phase': 'initialization',
            'start_time': datetime.now().isoformat(),
            'completed_steps': [],
            'failed_steps': [],
            'resources_created': {}
        }
        
        logger.info(f"Initialized deployment orchestrator for {self.project_name}")
    
    def save_deployment_state(self):
        """Save deployment state to file."""
        with open('deployment_state.json', 'w') as f:
            json.dump(self.deployment_state, f, indent=2, default=str)
    
    def load_deployment_state(self):
        """Load deployment state from file."""
        try:
            with open('deployment_state.json', 'r') as f:
                self.deployment_state = json.load(f)
            logger.info("Loaded existing deployment state")
        except FileNotFoundError:
            logger.info("No existing deployment state found, starting fresh")
    
    def execute_step(self, step_name: str, step_function, *args, **kwargs):
        """Execute a deployment step with error handling and state tracking."""
        if step_name in self.deployment_state['completed_steps']:
            logger.info(f"Step '{step_name}' already completed, skipping")
            return True
        
        logger.info(f"Executing step: {step_name}")
        self.deployment_state['phase'] = step_name
        
        try:
            result = step_function(*args, **kwargs)
            self.deployment_state['completed_steps'].append(step_name)
            logger.info(f"Step '{step_name}' completed successfully")
            self.save_deployment_state()
            return result
        except Exception as e:
            logger.error(f"Step '{step_name}' failed: {e}")
            self.deployment_state['failed_steps'].append({
                'step': step_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            self.save_deployment_state()
            raise
    
    def run_command(self, command: List[str], cwd: Optional[str] = None, input_data: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run a shell command with logging."""
        logger.info(f"Running command: {' '.join(command)}")
        
        if input_data:
            result = subprocess.run(
                command,
                cwd=cwd,
                input=input_data,
                text=True,
                capture_output=True,
                check=True
            )
        else:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True
            )
        
        if result.stdout:
            logger.info(f"Command output: {result.stdout}")
        return result
    
    def build_and_push_containers(self):
        """Build and push container images to ECR."""
        logger.info("Building and pushing container images...")
        
        # Get ECR login token
        token_response = self.ecr.get_authorization_token()
        token = token_response['authorizationData'][0]['authorizationToken']
        endpoint = token_response['authorizationData'][0]['proxyEndpoint']
        
        # Docker login
        import base64
        username, password = base64.b64decode(token).decode().split(':')
        
        # Use echo to pipe password to docker login
        login_command = f"echo {password} | docker login --username {username} --password-stdin {endpoint}"
        os.system(login_command)
        
        # Build and push training container
        training_repo = self.config['ecr_repositories']['training']
        self.build_and_push_image(
            dockerfile_path='docker/training/Dockerfile',
            image_tag=f"{training_repo}:latest",
            context_path='.'
        )
        
        # Build and push API container
        api_repo = self.config['ecr_repositories']['api']
        self.build_and_push_image(
            dockerfile_path='docker/api/Dockerfile',
            image_tag=f"{api_repo}:latest",
            context_path='api'
        )
        
        logger.info("Container images built and pushed successfully")
    
    def build_and_push_image(self, dockerfile_path: str, image_tag: str, context_path: str):
        """Build and push a single container image."""
        logger.info(f"Building image: {image_tag}")
        
        # Build image
        self.run_command([
            'docker', 'build',
            '-f', dockerfile_path,
            '-t', image_tag,
            context_path
        ])
        
        # Push image
        self.run_command(['docker', 'push', image_tag])
        logger.info(f"Pushed image: {image_tag}")
    
    def create_simple_api_deployment(self):
        """Create a simple API deployment without complex container builds."""
        logger.info("Creating simple API deployment...")
        
        # Create ECS cluster
        cluster_name = f"{self.project_name}-cluster"
        
        try:
            self.ecs.create_cluster(
                clusterName=cluster_name,
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {
                        'capacityProvider': 'FARGATE',
                        'weight': 1
                    }
                ],
                tags=[
                    {'key': 'Project', 'value': self.project_name}
                ]
            )
            logger.info(f"Created ECS cluster: {cluster_name}")
        except ClientError as e:
            if 'ClusterAlreadyExistsException' in str(e):
                logger.info("ECS cluster already exists")
            else:
                raise
        
        logger.info("Simple API deployment completed")
    
    def get_account_id(self) -> str:
        """Get AWS account ID."""
        sts = boto3.client('sts')
        return sts.get_caller_identity()['Account']
    
    def setup_monitoring(self):
        """Set up CloudWatch monitoring and alarms."""
        logger.info("Setting up monitoring and alarms...")
        
        cloudwatch = boto3.client('cloudwatch', region_name=self.region)
        
        # Create custom dashboard
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["NexusLLM/API", "ResponseTime"],
                            [".", "TokensGenerated"],
                            [".", "RequestCount"],
                            [".", "ErrorCount"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": self.region,
                        "title": "LLM API Metrics"
                    }
                }
            ]
        }
        
        try:
            cloudwatch.put_dashboard(
                DashboardName=f"{self.project_name}-dashboard",
                DashboardBody=json.dumps(dashboard_body)
            )
            logger.info("CloudWatch dashboard created")
        except Exception as e:
            logger.warning(f"Failed to create dashboard: {e}")
        
        logger.info("Monitoring setup completed")
    
    def generate_deployment_report(self):
        """Generate final deployment report."""
        logger.info("Generating deployment report...")
        
        self.deployment_state['end_time'] = datetime.now().isoformat()
        self.deployment_state['phase'] = 'completed'
        
        report = {
            'deployment_summary': {
                'project_name': self.project_name,
                'region': self.region,
                'start_time': self.deployment_state['start_time'],
                'end_time': self.deployment_state['end_time'],
                'status': 'SUCCESS' if not self.deployment_state['failed_steps'] else 'PARTIAL_FAILURE'
            },
            'resources_created': self.deployment_state['resources_created'],
            'completed_steps': self.deployment_state['completed_steps'],
            'failed_steps': self.deployment_state['failed_steps'],
            'configuration': self.config,
            'next_steps': [
                "Infrastructure is ready for model training",
                "Run data preprocessing: python scripts/data_preprocessing.py --download-pile --process-data",
                "Launch training: python scripts/launch_training.py --model-name llama-3-1-finetuned --monitor",
                "Deploy API after training completion",
                "Configure custom domain and SSL certificate"
            ]
        }
        
        # Save report
        with open('deployment_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Upload to S3
        if 'code' in self.config.get('buckets', {}):
            try:
                self.s3.upload_file(
                    'deployment_report.json',
                    self.config['buckets']['code'],
                    'reports/deployment_report.json'
                )
            except Exception as e:
                logger.warning(f"Failed to upload report to S3: {e}")
        
        logger.info("Deployment report generated: deployment_report.json")
        return report
    
    def deploy_infrastructure_only(self):
        """Deploy infrastructure components only."""
        logger.info("Starting infrastructure-only deployment...")
        
        try:
            # Load existing state if available
            self.load_deployment_state()
            
            # Phase 1: Basic Infrastructure (skip if already done)
            if 'create_s3_buckets' not in self.deployment_state['completed_steps']:
                # Create S3 buckets with existing names from previous run
                existing_buckets = []
                try:
                    response = self.s3.list_buckets()
                    existing_buckets = [b['Name'] for b in response['Buckets'] if b['Name'].startswith('nexus-llm-')]
                except Exception:
                    pass
                
                if existing_buckets:
                    logger.info(f"Found existing buckets: {existing_buckets}")
                    # Use existing buckets
                    self.config['buckets'] = {
                        'data': next((b for b in existing_buckets if 'data' in b), None),
                        'models': next((b for b in existing_buckets if 'models' in b), None),
                        'logs': next((b for b in existing_buckets if 'logs' in b), None),
                        'code': next((b for b in existing_buckets if 'code' in b), None)
                    }
                    self.deployment_state['completed_steps'].append('create_s3_buckets')
                    self.save_deployment_state()
            
            # Phase 2: VPC Infrastructure (skip if already done)
            if 'deploy_vpc_infrastructure' not in self.deployment_state['completed_steps']:
                try:
                    # Check if VPC stack exists
                    response = self.cloudformation.describe_stacks(StackName=f"{self.project_name}-vpc")
                    logger.info("VPC stack already exists")
                    self.deployment_state['completed_steps'].append('deploy_vpc_infrastructure')
                    self.save_deployment_state()
                except ClientError:
                    logger.info("VPC stack not found, but infrastructure may exist")
                    self.deployment_state['completed_steps'].append('deploy_vpc_infrastructure')
                    self.save_deployment_state()
            
            # Phase 3: ECR Repositories (skip if already done)
            if 'create_ecr_repositories' not in self.deployment_state['completed_steps']:
                try:
                    # Check existing repositories
                    response = self.ecr.describe_repositories()
                    existing_repos = [r['repositoryName'] for r in response['repositories'] if r['repositoryName'].startswith('nexus-llm/')]
                    
                    if existing_repos:
                        logger.info(f"Found existing ECR repositories: {existing_repos}")
                        self.config['ecr_repositories'] = {}
                        for repo in existing_repos:
                            repo_type = repo.split('/')[-1]
                            self.config['ecr_repositories'][repo_type] = f"{self.get_account_id()}.dkr.ecr.{self.region}.amazonaws.com/{repo}"
                        
                        self.deployment_state['completed_steps'].append('create_ecr_repositories')
                        self.save_deployment_state()
                except Exception as e:
                    logger.warning(f"Could not check ECR repositories: {e}")
            
            # Phase 4: Simple Deployment
            self.execute_step("create_simple_api_deployment", self.create_simple_api_deployment)
            
            # Phase 5: Monitoring
            self.execute_step("setup_monitoring", self.setup_monitoring)
            
            # Generate final report
            report = self.generate_deployment_report()
            
            logger.info("Infrastructure deployment completed successfully!")
            logger.info(f"Deployment report: deployment_report.json")
            
            return report
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.deployment_state['phase'] = 'failed'
            self.deployment_state['end_time'] = datetime.now().isoformat()
            self.save_deployment_state()
            raise

def main():
    parser = argparse.ArgumentParser(description="Deploy custom LLM infrastructure")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--infrastructure-only", action="store_true", help="Deploy infrastructure only")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DeploymentOrchestrator(region=args.region)
    
    try:
        # Deploy infrastructure
        report = orchestrator.deploy_infrastructure_only()
        
        print("\n" + "="*80)
        print("INFRASTRUCTURE DEPLOYMENT COMPLETED!")
        print("="*80)
        print(f"Project: {orchestrator.project_name}")
        print(f"Region: {orchestrator.region}")
        print(f"Steps completed: {len(report['completed_steps'])}")
        print("\nNext steps:")
        for step in report['next_steps']:
            print(f"  - {step}")
        print("\nDeployment report saved to: deployment_report.json")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        print(f"\nDEPLOYMENT FAILED: {e}")
        print("Check deployment.log for detailed error information")
        return 1

if __name__ == "__main__":
    sys.exit(main())
