# Production API Container Dockerfile for Custom LLM
# Document ID: api_dockerfile_20250705_070000
# Created: July 5, 2025 07:00:00 UTC

FROM python:3.10-slim

LABEL maintainer="Nexus LLM Team"
LABEL version="1.0"
LABEL description="Production API server for custom LLM serving"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HOST=0.0.0.0
ENV PORT=8080
ENV WORKERS=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional ML libraries
RUN pip install --no-cache-dir \
    transformers==4.35.0 \
    accelerate==0.24.0 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.4.2 \
    boto3==1.29.7 \
    psutil==5.9.5 \
    GPUtil==1.4.0 \
    numpy==1.24.3 \
    torch-audio==0.13.1

# Create non-root user
RUN useradd -m -u 1000 apiuser && \
    mkdir -p /var/log && \
    chown -R apiuser:apiuser /app /var/log

# Copy application code
COPY --chown=apiuser:apiuser server.py .
COPY --chown=apiuser:apiuser requirements.txt .

# Create directories for model and logs
RUN mkdir -p /opt/ml/model /var/log && \
    chown -R apiuser:apiuser /opt/ml /var/log

# Switch to non-root user
USER apiuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD ["python", "server.py"]
