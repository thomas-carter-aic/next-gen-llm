#!/bin/bash

# AWS Setup Script for Custom LLM Development
# Document ID: aws_setup_script_20250705_070000
# Created: July 5, 2025 07:00:00 UTC

set -e  # Exit on any error

# Configuration
PROJECT_NAME="nexus-llm"
AWS_REGION="us-west-2"
TIMESTAMP=$(date +%s)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI not found. Please install AWS CLI first."
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured. Run 'aws configure' first."
    fi
    
    # Check jq
    if ! command -v jq &> /dev/null; then
        warn "jq not found. Installing jq..."
        sudo apt-get update && sudo apt-get install -y jq
    fi
    
    log "Prerequisites check completed."
}

# Create S3 buckets
create_s3_buckets() {
    log "Creating S3 buckets..."
    
    # Data bucket
    DATA_BUCKET="${PROJECT_NAME}-data-${TIMESTAMP}"
    aws s3 mb "s3://${DATA_BUCKET}" --region ${AWS_REGION}
    log "Created data bucket: ${DATA_BUCKET}"
    
    # Models bucket
    MODELS_BUCKET="${PROJECT_NAME}-models-${TIMESTAMP}"
    aws s3 mb "s3://${MODELS_BUCKET}" --region ${AWS_REGION}
    log "Created models bucket: ${MODELS_BUCKET}"
    
    # Logs bucket
    LOGS_BUCKET="${PROJECT_NAME}-logs-${TIMESTAMP}"
    aws s3 mb "s3://${LOGS_BUCKET}" --region ${AWS_REGION}
    log "Created logs bucket: ${LOGS_BUCKET}"
    
    # Code bucket
    CODE_BUCKET="${PROJECT_NAME}-code-${TIMESTAMP}"
    aws s3 mb "s3://${CODE_BUCKET}" --region ${AWS_REGION}
    log "Created code bucket: ${CODE_BUCKET}"
    
    # Configure bucket policies and lifecycle
    configure_s3_lifecycle
    
    # Save bucket names to config file
    cat > aws_config.json << EOF
{
    "buckets": {
        "data": "${DATA_BUCKET}",
        "models": "${MODELS_BUCKET}",
        "logs": "${LOGS_BUCKET}",
        "code": "${CODE_BUCKET}"
    },
    "region": "${AWS_REGION}",
    "project": "${PROJECT_NAME}",
    "timestamp": "${TIMESTAMP}"
}
EOF
    
    log "S3 buckets created and configured."
}

# Configure S3 lifecycle policies
configure_s3_lifecycle() {
    log "Configuring S3 lifecycle policies..."
    
    # Create lifecycle configuration
    cat > s3-lifecycle.json << EOF
{
    "Rules": [
        {
            "ID": "DatasetLifecycle",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "raw/"
            },
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ]
        },
        {
            "ID": "LogsLifecycle",
            "Status": "Enabled",
            "Filter": {
                "Prefix": "logs/"
            },
            "Expiration": {
                "Days": 90
            }
        }
    ]
}
EOF
    
    # Apply lifecycle to data bucket
    aws s3api put-bucket-lifecycle-configuration \
        --bucket "${DATA_BUCKET}" \
        --lifecycle-configuration file://s3-lifecycle.json
    
    log "S3 lifecycle policies configured."
}

# Create IAM roles
create_iam_roles() {
    log "Creating IAM roles..."
    
    # SageMaker execution role
    create_sagemaker_role
    
    # ECS task execution role
    create_ecs_roles
    
    # EC2 instance role
    create_ec2_role
    
    log "IAM roles created."
}

# Create SageMaker execution role
create_sagemaker_role() {
    log "Creating SageMaker execution role..."
    
    # Trust policy for SageMaker
    cat > sagemaker-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    
    # Create role
    aws iam create-role \
        --role-name "${PROJECT_NAME}-sagemaker-role" \
        --assume-role-policy-document file://sagemaker-trust-policy.json \
        --description "SageMaker execution role for ${PROJECT_NAME}"
    
    # Attach policies
    aws iam attach-role-policy \
        --role-name "${PROJECT_NAME}-sagemaker-role" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    
    aws iam attach-role-policy \
        --role-name "${PROJECT_NAME}-sagemaker-role" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    
    # Create custom policy for additional permissions
    cat > sagemaker-custom-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "arn:aws:logs:*:*:*"
        }
    ]
}
EOF
    
    aws iam create-policy \
        --policy-name "${PROJECT_NAME}-sagemaker-custom-policy" \
        --policy-document file://sagemaker-custom-policy.json
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    aws iam attach-role-policy \
        --role-name "${PROJECT_NAME}-sagemaker-role" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-sagemaker-custom-policy"
    
    log "SageMaker role created."
}

# Create ECS roles
create_ecs_roles() {
    log "Creating ECS roles..."
    
    # ECS task execution role trust policy
    cat > ecs-task-execution-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ecs-tasks.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    
    # Create ECS task execution role
    aws iam create-role \
        --role-name "${PROJECT_NAME}-ecs-task-execution-role" \
        --assume-role-policy-document file://ecs-task-execution-trust-policy.json
    
    aws iam attach-role-policy \
        --role-name "${PROJECT_NAME}-ecs-task-execution-role" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
    
    # Create ECS task role
    aws iam create-role \
        --role-name "${PROJECT_NAME}-ecs-task-role" \
        --assume-role-policy-document file://ecs-task-execution-trust-policy.json
    
    # Custom policy for ECS task role
    cat > ecs-task-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${MODELS_BUCKET}",
                "arn:aws:s3:::${MODELS_BUCKET}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "cloudwatch:PutMetricData"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
    aws iam create-policy \
        --policy-name "${PROJECT_NAME}-ecs-task-policy" \
        --policy-document file://ecs-task-policy.json
    
    aws iam attach-role-policy \
        --role-name "${PROJECT_NAME}-ecs-task-role" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-ecs-task-policy"
    
    log "ECS roles created."
}

# Create EC2 instance role
create_ec2_role() {
    log "Creating EC2 instance role..."
    
    # EC2 trust policy
    cat > ec2-trust-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "ec2.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOF
    
    # Create role
    aws iam create-role \
        --role-name "${PROJECT_NAME}-ec2-role" \
        --assume-role-policy-document file://ec2-trust-policy.json
    
    # Custom policy for EC2
    cat > ec2-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::${DATA_BUCKET}",
                "arn:aws:s3:::${DATA_BUCKET}/*",
                "arn:aws:s3:::${MODELS_BUCKET}",
                "arn:aws:s3:::${MODELS_BUCKET}/*",
                "arn:aws:s3:::${CODE_BUCKET}",
                "arn:aws:s3:::${CODE_BUCKET}/*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob"
            ],
            "Resource": "*"
        }
    ]
}
EOF
    
    aws iam create-policy \
        --policy-name "${PROJECT_NAME}-ec2-policy" \
        --policy-document file://ec2-policy.json
    
    aws iam attach-role-policy \
        --role-name "${PROJECT_NAME}-ec2-role" \
        --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${PROJECT_NAME}-ec2-policy"
    
    # Create instance profile
    aws iam create-instance-profile \
        --instance-profile-name "${PROJECT_NAME}-ec2-instance-profile"
    
    aws iam add-role-to-instance-profile \
        --instance-profile-name "${PROJECT_NAME}-ec2-instance-profile" \
        --role-name "${PROJECT_NAME}-ec2-role"
    
    log "EC2 role and instance profile created."
}

# Create VPC and networking
create_vpc() {
    log "Creating VPC and networking components..."
    
    # Create VPC
    VPC_ID=$(aws ec2 create-vpc \
        --cidr-block 10.0.0.0/16 \
        --query 'Vpc.VpcId' \
        --output text)
    
    aws ec2 create-tags \
        --resources ${VPC_ID} \
        --tags Key=Name,Value="${PROJECT_NAME}-vpc"
    
    log "Created VPC: ${VPC_ID}"
    
    # Create Internet Gateway
    IGW_ID=$(aws ec2 create-internet-gateway \
        --query 'InternetGateway.InternetGatewayId' \
        --output text)
    
    aws ec2 attach-internet-gateway \
        --vpc-id ${VPC_ID} \
        --internet-gateway-id ${IGW_ID}
    
    aws ec2 create-tags \
        --resources ${IGW_ID} \
        --tags Key=Name,Value="${PROJECT_NAME}-igw"
    
    log "Created Internet Gateway: ${IGW_ID}"
    
    # Create subnets
    create_subnets ${VPC_ID}
    
    # Create security groups
    create_security_groups ${VPC_ID}
    
    # Update config file
    jq --arg vpc_id "${VPC_ID}" --arg igw_id "${IGW_ID}" \
        '.vpc_id = $vpc_id | .igw_id = $igw_id' aws_config.json > tmp.json && mv tmp.json aws_config.json
    
    log "VPC and networking created."
}

# Create subnets
create_subnets() {
    local vpc_id=$1
    log "Creating subnets..."
    
    # Public subnet 1
    PUBLIC_SUBNET_1=$(aws ec2 create-subnet \
        --vpc-id ${vpc_id} \
        --cidr-block 10.0.1.0/24 \
        --availability-zone "${AWS_REGION}a" \
        --query 'Subnet.SubnetId' \
        --output text)
    
    aws ec2 create-tags \
        --resources ${PUBLIC_SUBNET_1} \
        --tags Key=Name,Value="${PROJECT_NAME}-public-subnet-1"
    
    # Public subnet 2
    PUBLIC_SUBNET_2=$(aws ec2 create-subnet \
        --vpc-id ${vpc_id} \
        --cidr-block 10.0.2.0/24 \
        --availability-zone "${AWS_REGION}b" \
        --query 'Subnet.SubnetId' \
        --output text)
    
    aws ec2 create-tags \
        --resources ${PUBLIC_SUBNET_2} \
        --tags Key=Name,Value="${PROJECT_NAME}-public-subnet-2"
    
    # Private subnet 1
    PRIVATE_SUBNET_1=$(aws ec2 create-subnet \
        --vpc-id ${vpc_id} \
        --cidr-block 10.0.3.0/24 \
        --availability-zone "${AWS_REGION}a" \
        --query 'Subnet.SubnetId' \
        --output text)
    
    aws ec2 create-tags \
        --resources ${PRIVATE_SUBNET_1} \
        --tags Key=Name,Value="${PROJECT_NAME}-private-subnet-1"
    
    # Private subnet 2
    PRIVATE_SUBNET_2=$(aws ec2 create-subnet \
        --vpc-id ${vpc_id} \
        --cidr-block 10.0.4.0/24 \
        --availability-zone "${AWS_REGION}b" \
        --query 'Subnet.SubnetId' \
        --output text)
    
    aws ec2 create-tags \
        --resources ${PRIVATE_SUBNET_2} \
        --tags Key=Name,Value="${PROJECT_NAME}-private-subnet-2"
    
    # Create route table for public subnets
    PUBLIC_RT=$(aws ec2 create-route-table \
        --vpc-id ${vpc_id} \
        --query 'RouteTable.RouteTableId' \
        --output text)
    
    aws ec2 create-route \
        --route-table-id ${PUBLIC_RT} \
        --destination-cidr-block 0.0.0.0/0 \
        --gateway-id ${IGW_ID}
    
    # Associate public subnets with route table
    aws ec2 associate-route-table \
        --subnet-id ${PUBLIC_SUBNET_1} \
        --route-table-id ${PUBLIC_RT}
    
    aws ec2 associate-route-table \
        --subnet-id ${PUBLIC_SUBNET_2} \
        --route-table-id ${PUBLIC_RT}
    
    # Update config
    jq --arg pub1 "${PUBLIC_SUBNET_1}" --arg pub2 "${PUBLIC_SUBNET_2}" \
       --arg priv1 "${PRIVATE_SUBNET_1}" --arg priv2 "${PRIVATE_SUBNET_2}" \
       '.subnets = {"public": [$pub1, $pub2], "private": [$priv1, $priv2]}' \
       aws_config.json > tmp.json && mv tmp.json aws_config.json
    
    log "Subnets created and configured."
}

# Create security groups
create_security_groups() {
    local vpc_id=$1
    log "Creating security groups..."
    
    # Development security group
    DEV_SG=$(aws ec2 create-security-group \
        --group-name "${PROJECT_NAME}-dev-sg" \
        --description "Security group for development instances" \
        --vpc-id ${vpc_id} \
        --query 'GroupId' \
        --output text)
    
    # Allow SSH access
    aws ec2 authorize-security-group-ingress \
        --group-id ${DEV_SG} \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
    
    # Allow Jupyter notebook access
    aws ec2 authorize-security-group-ingress \
        --group-id ${DEV_SG} \
        --protocol tcp \
        --port 8888 \
        --cidr 0.0.0.0/0
    
    # Production API security group
    API_SG=$(aws ec2 create-security-group \
        --group-name "${PROJECT_NAME}-api-sg" \
        --description "Security group for API servers" \
        --vpc-id ${vpc_id} \
        --query 'GroupId' \
        --output text)
    
    # Allow HTTP/HTTPS access
    aws ec2 authorize-security-group-ingress \
        --group-id ${API_SG} \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0
    
    aws ec2 authorize-security-group-ingress \
        --group-id ${API_SG} \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0
    
    aws ec2 authorize-security-group-ingress \
        --group-id ${API_SG} \
        --protocol tcp \
        --port 8080 \
        --cidr 0.0.0.0/0
    
    # Load balancer security group
    ALB_SG=$(aws ec2 create-security-group \
        --group-name "${PROJECT_NAME}-alb-sg" \
        --description "Security group for application load balancer" \
        --vpc-id ${vpc_id} \
        --query 'GroupId' \
        --output text)
    
    aws ec2 authorize-security-group-ingress \
        --group-id ${ALB_SG} \
        --protocol tcp \
        --port 80 \
        --cidr 0.0.0.0/0
    
    aws ec2 authorize-security-group-ingress \
        --group-id ${ALB_SG} \
        --protocol tcp \
        --port 443 \
        --cidr 0.0.0.0/0
    
    # Update config
    jq --arg dev_sg "${DEV_SG}" --arg api_sg "${API_SG}" --arg alb_sg "${ALB_SG}" \
       '.security_groups = {"dev": $dev_sg, "api": $api_sg, "alb": $alb_sg}' \
       aws_config.json > tmp.json && mv tmp.json aws_config.json
    
    log "Security groups created."
}

# Create ECR repositories
create_ecr_repositories() {
    log "Creating ECR repositories..."
    
    # Training container repository
    aws ecr create-repository \
        --repository-name "${PROJECT_NAME}/training" \
        --region ${AWS_REGION}
    
    # API container repository
    aws ecr create-repository \
        --repository-name "${PROJECT_NAME}/api" \
        --region ${AWS_REGION}
    
    # Data processing repository
    aws ecr create-repository \
        --repository-name "${PROJECT_NAME}/data-processing" \
        --region ${AWS_REGION}
    
    log "ECR repositories created."
}

# Create ECS cluster
create_ecs_cluster() {
    log "Creating ECS cluster..."
    
    aws ecs create-cluster \
        --cluster-name "${PROJECT_NAME}-cluster" \
        --capacity-providers FARGATE \
        --default-capacity-provider-strategy capacityProvider=FARGATE,weight=1
    
    log "ECS cluster created."
}

# Create CloudWatch log groups
create_log_groups() {
    log "Creating CloudWatch log groups..."
    
    # SageMaker training logs
    aws logs create-log-group \
        --log-group-name "/aws/sagemaker/TrainingJobs/${PROJECT_NAME}"
    
    # ECS API logs
    aws logs create-log-group \
        --log-group-name "/ecs/${PROJECT_NAME}-api"
    
    # Custom application logs
    aws logs create-log-group \
        --log-group-name "/aws/${PROJECT_NAME}/application"
    
    log "CloudWatch log groups created."
}

# Setup monitoring and alarms
setup_monitoring() {
    log "Setting up monitoring and alarms..."
    
    # Create SNS topic for alerts
    TOPIC_ARN=$(aws sns create-topic \
        --name "${PROJECT_NAME}-alerts" \
        --query 'TopicArn' \
        --output text)
    
    # Create cost budget
    cat > budget.json << EOF
{
    "BudgetName": "${PROJECT_NAME}-monthly-budget",
    "BudgetLimit": {
        "Amount": "500",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "BudgetType": "COST",
    "CostFilters": {
        "TagKey": ["Project"],
        "TagValue": ["${PROJECT_NAME}"]
    }
}
EOF
    
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    aws budgets create-budget \
        --account-id ${ACCOUNT_ID} \
        --budget file://budget.json
    
    log "Monitoring and budgets configured."
}

# Create key pair for EC2 instances
create_key_pair() {
    log "Creating EC2 key pair..."
    
    KEY_NAME="${PROJECT_NAME}-keypair"
    aws ec2 create-key-pair \
        --key-name ${KEY_NAME} \
        --query 'KeyMaterial' \
        --output text > ${KEY_NAME}.pem
    
    chmod 400 ${KEY_NAME}.pem
    
    log "Key pair created: ${KEY_NAME}.pem"
    warn "Keep the ${KEY_NAME}.pem file secure - it's needed for SSH access"
}

# Upload code and configurations to S3
upload_code() {
    log "Uploading code and configurations to S3..."
    
    # Create code archive
    tar -czf code.tar.gz ../scripts/ ../docker/ ../training/ ../api/ || true
    
    # Upload to S3
    aws s3 cp code.tar.gz "s3://${CODE_BUCKET}/code.tar.gz"
    aws s3 cp aws_config.json "s3://${CODE_BUCKET}/config/aws_config.json"
    
    # Upload configuration files
    aws s3 cp s3-lifecycle.json "s3://${CODE_BUCKET}/config/" || true
    aws s3 cp sagemaker-trust-policy.json "s3://${CODE_BUCKET}/config/" || true
    
    log "Code and configurations uploaded."
}

# Generate deployment summary
generate_summary() {
    log "Generating deployment summary..."
    
    cat > deployment_summary.md << EOF
# AWS Infrastructure Deployment Summary

**Project**: ${PROJECT_NAME}
**Region**: ${AWS_REGION}
**Timestamp**: $(date)

## Resources Created

### S3 Buckets
- Data: ${DATA_BUCKET}
- Models: ${MODELS_BUCKET}
- Logs: ${LOGS_BUCKET}
- Code: ${CODE_BUCKET}

### IAM Roles
- SageMaker: ${PROJECT_NAME}-sagemaker-role
- ECS Task Execution: ${PROJECT_NAME}-ecs-task-execution-role
- ECS Task: ${PROJECT_NAME}-ecs-task-role
- EC2: ${PROJECT_NAME}-ec2-role

### Networking
- VPC: ${VPC_ID}
- Internet Gateway: ${IGW_ID}
- Public Subnets: ${PUBLIC_SUBNET_1}, ${PUBLIC_SUBNET_2}
- Private Subnets: ${PRIVATE_SUBNET_1}, ${PRIVATE_SUBNET_2}

### Security Groups
- Development: ${DEV_SG}
- API: ${API_SG}
- Load Balancer: ${ALB_SG}

### Other Resources
- ECS Cluster: ${PROJECT_NAME}-cluster
- Key Pair: ${KEY_NAME}
- ECR Repositories: ${PROJECT_NAME}/training, ${PROJECT_NAME}/api, ${PROJECT_NAME}/data-processing

## Next Steps

1. Launch development instance:
   \`\`\`bash
   aws ec2 run-instances \\
     --image-id ami-0c02fb55956c7d316 \\
     --instance-type p3.2xlarge \\
     --key-name ${KEY_NAME} \\
     --security-group-ids ${DEV_SG} \\
     --subnet-id ${PUBLIC_SUBNET_1} \\
     --iam-instance-profile Name=${PROJECT_NAME}-ec2-instance-profile
   \`\`\`

2. Download datasets to S3 data bucket
3. Begin model training with SageMaker
4. Deploy API to ECS cluster

## Configuration File
All configuration details are saved in: aws_config.json

## Security Notes
- Keep ${KEY_NAME}.pem file secure
- Review IAM policies for least privilege
- Enable CloudTrail for audit logging
- Consider enabling GuardDuty for security monitoring

## Cost Monitoring
- Monthly budget set to \$500
- Monitor costs in AWS Cost Explorer
- Use Spot instances for training to reduce costs

EOF

    log "Deployment summary generated: deployment_summary.md"
}

# Main execution
main() {
    log "Starting AWS infrastructure setup for ${PROJECT_NAME}..."
    
    check_prerequisites
    create_s3_buckets
    create_iam_roles
    create_vpc
    create_ecr_repositories
    create_ecs_cluster
    create_log_groups
    setup_monitoring
    create_key_pair
    upload_code
    generate_summary
    
    log "AWS infrastructure setup completed successfully!"
    log "Review deployment_summary.md for next steps."
    log "Configuration saved in aws_config.json"
    
    # Clean up temporary files
    rm -f *.json *.pem.pub code.tar.gz 2>/dev/null || true
}

# Run main function
main "$@"
