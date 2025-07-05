# Training Container Dockerfile for Custom LLM
# Document ID: training_dockerfile_20250705_070000
# Created: July 5, 2025 07:00:00 UTC

FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

LABEL maintainer="Nexus LLM Team"
LABEL version="1.0"
LABEL description="Custom LLM Training Container with LLaMA 3.1 support"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV SAGEMAKER_PROGRAM=train.py
ENV SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/code

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install core ML libraries with specific versions for stability
RUN pip install \
    transformers==4.35.0 \
    datasets==2.14.0 \
    tokenizers==0.14.1 \
    accelerate==0.24.0 \
    deepspeed==0.10.0 \
    peft==0.6.0 \
    bitsandbytes==0.41.1 \
    scipy==1.11.3 \
    scikit-learn==1.3.0 \
    pandas==2.1.1 \
    numpy==1.24.3 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    tqdm==4.66.1 \
    wandb==0.15.12 \
    tensorboard==2.14.1 \
    evaluate==0.4.1 \
    rouge-score==0.1.2 \
    nltk==3.8.1 \
    sentencepiece==0.1.99 \
    protobuf==3.20.3

# Install additional utilities
RUN pip install \
    psutil==5.9.5 \
    GPUtil==1.4.0 \
    py3nvml==0.2.7 \
    boto3==1.29.7 \
    awscli==1.29.7 \
    jsonlines==3.1.0 \
    pyarrow==13.0.0 \
    fastparquet==2023.8.0

# Create necessary directories
RUN mkdir -p /opt/ml/code \
    && mkdir -p /opt/ml/input \
    && mkdir -p /opt/ml/output \
    && mkdir -p /opt/ml/model \
    && mkdir -p /opt/ml/checkpoints \
    && mkdir -p /tmp/cache

# Set up Hugging Face cache directory
ENV HF_HOME=/tmp/cache
ENV TRANSFORMERS_CACHE=/tmp/cache
ENV HF_DATASETS_CACHE=/tmp/cache

# Copy training scripts and configurations
COPY train.py /opt/ml/code/train.py
COPY utils.py /opt/ml/code/utils.py
COPY data_processing.py /opt/ml/code/data_processing.py
COPY model_utils.py /opt/ml/code/model_utils.py
COPY ds_config.json /opt/ml/code/ds_config.json
COPY requirements.txt /opt/ml/code/requirements.txt

# Copy evaluation scripts
COPY evaluate.py /opt/ml/code/evaluate.py
COPY metrics.py /opt/ml/code/metrics.py

# Make scripts executable
RUN chmod +x /opt/ml/code/*.py

# Set working directory
WORKDIR /opt/ml/code

# Create a non-root user for security
RUN useradd -m -u 1000 sagemaker && \
    chown -R sagemaker:sagemaker /opt/ml && \
    chown -R sagemaker:sagemaker /tmp/cache

# Switch to non-root user
USER sagemaker

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Verify installations
RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
RUN python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
RUN python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Set the entrypoint
ENTRYPOINT ["python", "train.py"]
