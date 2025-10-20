# Use Ray base image with Python 3.11 and CUDA support
FROM anyscale/ray:2.49.2-slim-py311-cpu
#FROM rayproject/ray:2.49.2-py311-cpu


# Install system dependencies
RUN sudo apt-get update && sudo apt-get install -y \
    build-essential \
    gcc \
    g++ \
    espeak-ng \
    espeak-ng-data \
    libespeak-ng1 \
    pkg-config \
    rustc \
    cargo \
    && sudo rm -rf /var/lib/apt/lists/* \
    && sudo rm -f /etc/apt/sources.list.d/*

# Upgrade pip before requirements (just before your pip install -r)
RUN pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-cpu.txt requirements.txt

# Install Python dependencies with space optimization
#TODO: RUN pip install --no-cache-dir -r requirements.txt && \
RUN pip install -r requirements.txt && \
    find /home/ray/anaconda3 -name "*.pyc" -delete && \
    find /home/ray/anaconda3 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true && \
    find /tmp -name "pip-*" -type d -exec rm -rf {} + 2>/dev/null || true && \
    python -c "import TTS; print('TTS imported successfully')"

# Copy application files
COPY app.py .
COPY tts.proto .

# Copy model directory
COPY model/ model/

# Generate gRPC code
RUN pip install grpcio-tools && \
    python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path=. tts.proto

# Expose port (will be overridden by PORT env var)
EXPOSE 8080

# Health check (Ray Serve specific)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import ray; ray.init(); from ray import serve; print('Ray Serve is running')" || exit 1

# Default command
CMD ["python", "app.py"]