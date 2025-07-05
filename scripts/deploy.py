#!/usr/bin/env python3
"""
Complete Deployment Orchestrator for Custom LLM
Document ID: deployment_orchestrator_20250705_070000
Created: July 5, 2025 07:00:00 UTC

This script orchestrates the complete deployment of the custom LLM system
including infrastructure provisioning, data processing, training, and production deployment.
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
    
    def run_command(self, command: List[str], cwd: Optional[str] = None) -> subprocess.CompletedProcess:
        """Run a shell command with logging."""
        logger.info(f"Running command: {' '.join(command)}")
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
    
    def wait_for_stack(self, stack_name: str, operation: str = 'CREATE'):
        """Wait for CloudFormation stack operation to complete."""
        logger.info(f"Waiting for stack {stack_name} {operation.lower()} to complete...")
        
        waiter_name = f'stack_{operation.lower()}_complete'
        waiter = self.cloudformation.get_waiter(waiter_name)
        
        try:
            waiter.wait(
                StackName=stack_name,
                WaiterConfig={
                    'Delay': 30,
                    'MaxAttempts': 120  # 60 minutes max
                }
            )
            logger.info(f"Stack {stack_name} {operation.lower()} completed successfully")
        except WaiterError as e:
            logger.error(f"Stack {stack_name} {operation.lower()} failed: {e}")
            raise
    
    def create_s3_buckets(self):
        """Create S3 buckets for the project."""
        logger.info("Creating S3 buckets...")
        
        bucket_names = {
            'data': f"{self.project_name}-data-{self.timestamp}",
            'models': f"{self.project_name}-models-{self.timestamp}",
            'logs': f"{self.project_name}-logs-{self.timestamp}",
            'code': f"{self.project_name}-code-{self.timestamp}"
        }
        
        for bucket_type, bucket_name in bucket_names.items():
            try:
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
                logger.info(f"Created {bucket_type} bucket: {bucket_name}")
                
                # Configure bucket policies
                if bucket_type == 'data':
                    self.configure_data_bucket_lifecycle(bucket_name)
                
            except ClientError as e:
                if e.response['Error']['Code'] != 'BucketAlreadyOwnedByYou':
                    raise
        
        self.config['buckets'] = bucket_names
        self.deployment_state['resources_created']['s3_buckets'] = bucket_names
        return bucket_names
    
    def configure_data_bucket_lifecycle(self, bucket_name: str):
        """Configure lifecycle policy for data bucket."""
        lifecycle_config = {
            'Rules': [
                {
                    'ID': 'DatasetLifecycle',
                    'Status': 'Enabled',
                    'Filter': {'Prefix': 'raw/'},
                    'Transitions': [
                        {
                            'Days': 30,
                            'StorageClass': 'STANDARD_IA'
                        },
                        {
                            'Days': 90,
                            'StorageClass': 'GLACIER'
                        }
                    ]
                }
            ]
        }
        
        self.s3.put_bucket_lifecycle_configuration(
            Bucket=bucket_name,
            LifecycleConfiguration=lifecycle_config
        )
        logger.info(f"Configured lifecycle policy for bucket {bucket_name}")
    
    def deploy_vpc_infrastructure(self):
        """Deploy VPC infrastructure using CloudFormation."""
        logger.info("Deploying VPC infrastructure...")
        
        stack_name = f"{self.project_name}-vpc"
        template_path = Path(__file__).parent.parent / "cloudformation" / "vpc-template.yaml"
        
        with open(template_path, 'r') as f:
            template_body = f.read()
        
        try:
            self.cloudformation.create_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=[
                    {'ParameterKey': 'ProjectName', 'ParameterValue': self.project_name},
                    {'ParameterKey': 'Environment', 'ParameterValue': 'production'}
                ],
                Capabilities=['CAPABILITY_IAM'],
                Tags=[
                    {'Key': 'Project', 'Value': self.project_name},
                    {'Key': 'Environment', 'Value': 'production'}
                ]
            )
            
            self.wait_for_stack(stack_name, 'CREATE')
            
            # Get stack outputs
            response = self.cloudformation.describe_stacks(StackName=stack_name)
            outputs = response['Stacks'][0].get('Outputs', [])
            
            vpc_info = {}
            for output in outputs:
                vpc_info[output['OutputKey']] = output['OutputValue']
            
            self.config['vpc'] = vpc_info
            self.deployment_state['resources_created']['vpc_stack'] = stack_name
            
            logger.info("VPC infrastructure deployed successfully")
            return vpc_info
            
        except ClientError as e:
            if 'AlreadyExistsException' in str(e):
                logger.info("VPC stack already exists, skipping creation")
                return self.get_existing_vpc_info(stack_name)
            raise
    
    def get_existing_vpc_info(self, stack_name: str) -> Dict[str, str]:
        """Get information from existing VPC stack."""
        response = self.cloudformation.describe_stacks(StackName=stack_name)
        outputs = response['Stacks'][0].get('Outputs', [])
        
        vpc_info = {}
        for output in outputs:
            vpc_info[output['OutputKey']] = output['OutputValue']
        
        return vpc_info
    
    def create_ecr_repositories(self):
        """Create ECR repositories for container images."""
        logger.info("Creating ECR repositories...")
        
        repositories = ['training', 'api', 'data-processing']
        created_repos = {}
        
        for repo_name in repositories:
            full_repo_name = f"{self.project_name}/{repo_name}"
            
            try:
                response = self.ecr.create_repository(
                    repositoryName=full_repo_name,
                    imageScanningConfiguration={'scanOnPush': True},
                    encryptionConfiguration={'encryptionType': 'AES256'}
                )
                
                repo_uri = response['repository']['repositoryUri']
                created_repos[repo_name] = repo_uri
                logger.info(f"Created ECR repository: {repo_uri}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == 'RepositoryAlreadyExistsException':
                    # Get existing repository URI
                    response = self.ecr.describe_repositories(repositoryNames=[full_repo_name])
                    repo_uri = response['repositories'][0]['repositoryUri']
                    created_repos[repo_name] = repo_uri
                    logger.info(f"ECR repository already exists: {repo_uri}")
                else:
                    raise
        
        self.config['ecr_repositories'] = created_repos
        self.deployment_state['resources_created']['ecr_repositories'] = created_repos
        return created_repos
    
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
        
        self.run_command([
            'docker', 'login', '--username', username, '--password-stdin', endpoint
        ], input=password)
        
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
    
    def setup_data_pipeline(self):
        """Set up and execute data preprocessing pipeline."""
        logger.info("Setting up data preprocessing pipeline...")
        
        # Upload preprocessing script to S3
        code_bucket = self.config['buckets']['code']
        
        self.s3.upload_file(
            'scripts/data_preprocessing.py',
            code_bucket,
            'scripts/data_preprocessing.py'
        )
        
        # Create configuration file
        config_data = {
            'buckets': self.config['buckets'],
            'region': self.region,
            'project': self.project_name,
            'timestamp': self.timestamp
        }
        
        with open('aws_config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Upload config
        self.s3.upload_file(
            'aws_config.json',
            code_bucket,
            'config/aws_config.json'
        )
        
        logger.info("Data pipeline setup completed")
    
    def launch_training_job(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        """Launch SageMaker training job."""
        logger.info("Launching SageMaker training job...")
        
        # Use the training launcher script
        self.run_command([
            'python', 'scripts/launch_training.py',
            '--model-name', 'llama-3-1-finetuned',
            '--config', 'aws_config.json',
            '--epochs', '3',
            '--batch-size', '4',
            '--use-spot',
            '--monitor'
        ])
        
        logger.info("Training job launched successfully")
    
    def deploy_production_api(self):
        """Deploy production API using ECS."""
        logger.info("Deploying production API...")
        
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
        
        # Create task definition and service
        self.create_ecs_task_definition()
        self.create_ecs_service()
        
        logger.info("Production API deployed successfully")
    
    def create_ecs_task_definition(self):
        """Create ECS task definition for API service."""
        task_def = {
            'family': f"{self.project_name}-api",
            'networkMode': 'awsvpc',
            'requiresCompatibilities': ['FARGATE'],
            'cpu': '4096',
            'memory': '16384',
            'executionRoleArn': f"arn:aws:iam::{self.get_account_id()}:role/{self.project_name}-ecs-task-execution-role",
            'taskRoleArn': f"arn:aws:iam::{self.get_account_id()}:role/{self.project_name}-ecs-task-role",
            'containerDefinitions': [
                {
                    'name': 'llm-api',
                    'image': self.config['ecr_repositories']['api'],
                    'portMappings': [
                        {
                            'containerPort': 8080,
                            'protocol': 'tcp'
                        }
                    ],
                    'environment': [
                        {
                            'name': 'MODEL_PATH',
                            'value': f"s3://{self.config['buckets']['models']}/llama-3-1-finetuned/"
                        },
                        {
                            'name': 'AWS_REGION',
                            'value': self.region
                        }
                    ],
                    'logConfiguration': {
                        'logDriver': 'awslogs',
                        'options': {
                            'awslogs-group': f"/ecs/{self.project_name}-api",
                            'awslogs-region': self.region,
                            'awslogs-stream-prefix': 'ecs'
                        }
                    },
                    'healthCheck': {
                        'command': ['CMD-SHELL', 'curl -f http://localhost:8080/health || exit 1'],
                        'interval': 30,
                        'timeout': 5,
                        'retries': 3,
                        'startPeriod': 60
                    }
                }
            ]
        }
        
        response = self.ecs.register_task_definition(**task_def)
        task_def_arn = response['taskDefinition']['taskDefinitionArn']
        
        self.deployment_state['resources_created']['ecs_task_definition'] = task_def_arn
        logger.info(f"Created ECS task definition: {task_def_arn}")
        
        return task_def_arn
    
    def create_ecs_service(self):
        """Create ECS service for API deployment."""
        service_config = {
            'cluster': f"{self.project_name}-cluster",
            'serviceName': f"{self.project_name}-api",
            'taskDefinition': f"{self.project_name}-api",
            'desiredCount': 2,
            'launchType': 'FARGATE',
            'networkConfiguration': {
                'awsvpcConfiguration': {
                    'subnets': self.config['vpc']['PrivateSubnets'].split(','),
                    'securityGroups': [self.config['vpc']['APISecurityGroup']],
                    'assignPublicIp': 'DISABLED'
                }
            },
            'enableExecuteCommand': True,
            'tags': [
                {'key': 'Project', 'value': self.project_name}
            ]
        }
        
        response = self.ecs.create_service(**service_config)
        service_arn = response['service']['serviceArn']
        
        self.deployment_state['resources_created']['ecs_service'] = service_arn
        logger.info(f"Created ECS service: {service_arn}")
        
        return service_arn
    
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
        
        cloudwatch.put_dashboard(
            DashboardName=f"{self.project_name}-dashboard",
            DashboardBody=json.dumps(dashboard_body)
        )
        
        # Create alarms
        self.create_cloudwatch_alarms(cloudwatch)
        
        logger.info("Monitoring setup completed")
    
    def create_cloudwatch_alarms(self, cloudwatch):
        """Create CloudWatch alarms."""
        alarms = [
            {
                'AlarmName': f"{self.project_name}-high-response-time",
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 2,
                'MetricName': 'ResponseTime',
                'Namespace': 'NexusLLM/API',
                'Period': 300,
                'Statistic': 'Average',
                'Threshold': 5.0,
                'ActionsEnabled': True,
                'AlarmDescription': 'Alert when response time exceeds 5 seconds',
                'Unit': 'Seconds'
            },
            {
                'AlarmName': f"{self.project_name}-high-error-rate",
                'ComparisonOperator': 'GreaterThanThreshold',
                'EvaluationPeriods': 1,
                'MetricName': 'ErrorCount',
                'Namespace': 'NexusLLM/API',
                'Period': 300,
                'Statistic': 'Sum',
                'Threshold': 10,
                'ActionsEnabled': True,
                'AlarmDescription': 'Alert when error count exceeds 10 per 5 minutes',
                'Unit': 'Count'
            }
        ]
        
        for alarm in alarms:
            cloudwatch.put_metric_alarm(**alarm)
            logger.info(f"Created alarm: {alarm['AlarmName']}")
    
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
                "Monitor training job progress in SageMaker console",
                "Test API endpoints after training completion",
                "Configure custom domain and SSL certificate",
                "Set up automated backups and disaster recovery",
                "Review and optimize costs using AWS Cost Explorer"
            ]
        }
        
        # Save report
        with open('deployment_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Upload to S3
        if 'code' in self.config.get('buckets', {}):
            self.s3.upload_file(
                'deployment_report.json',
                self.config['buckets']['code'],
                'reports/deployment_report.json'
            )
        
        logger.info("Deployment report generated: deployment_report.json")
        return report
    
    def deploy_complete_system(self):
        """Deploy the complete LLM system."""
        logger.info("Starting complete system deployment...")
        
        try:
            # Phase 1: Infrastructure
            self.execute_step("create_s3_buckets", self.create_s3_buckets)
            self.execute_step("deploy_vpc_infrastructure", self.deploy_vpc_infrastructure)
            self.execute_step("create_ecr_repositories", self.create_ecr_repositories)
            
            # Phase 2: Container Images
            self.execute_step("build_and_push_containers", self.build_and_push_containers)
            
            # Phase 3: Data Pipeline
            self.execute_step("setup_data_pipeline", self.setup_data_pipeline)
            
            # Phase 4: Training (optional - can be run separately)
            # self.execute_step("launch_training_job", self.launch_training_job)
            
            # Phase 5: Production Deployment
            self.execute_step("deploy_production_api", self.deploy_production_api)
            
            # Phase 6: Monitoring
            self.execute_step("setup_monitoring", self.setup_monitoring)
            
            # Generate final report
            report = self.generate_deployment_report()
            
            logger.info("Complete system deployment finished successfully!")
            logger.info(f"Deployment report: deployment_report.json")
            
            return report
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            self.deployment_state['phase'] = 'failed'
            self.deployment_state['end_time'] = datetime.now().isoformat()
            self.save_deployment_state()
            raise

def main():
    parser = argparse.ArgumentParser(description="Deploy complete custom LLM system")
    parser.add_argument("--region", default="us-west-2", help="AWS region")
    parser.add_argument("--config", default="aws_config.json", help="Configuration file")
    parser.add_argument("--resume", action="store_true", help="Resume from previous deployment")
    parser.add_argument("--skip-training", action="store_true", help="Skip training job launch")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DeploymentOrchestrator(args.config, args.region)
    
    # Load previous state if resuming
    if args.resume:
        orchestrator.load_deployment_state()
    
    try:
        # Deploy complete system
        report = orchestrator.deploy_complete_system()
        
        print("\n" + "="*80)
        print("DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Project: {orchestrator.project_name}")
        print(f"Region: {orchestrator.region}")
        print(f"Resources created: {len(report['resources_created'])}")
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
        print("Use --resume flag to continue from last successful step")
        return 1

if __name__ == "__main__":
    sys.exit(main())
