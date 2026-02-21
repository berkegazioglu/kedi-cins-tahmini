# Python 3.11 base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY backend/ backend/
COPY frontend/ frontend/
COPY yolo11n.pt .
COPY cat_breed_info.json .
COPY *.pth .

# Copy Streamlit config
COPY .streamlit .streamlit

# Create necessary directories
RUN mkdir -p runs/resnet50_v2/weights
RUN mkdir -p runs/ensemble/weights
RUN mkdir -p runs/optimal_ensemble/weights
RUN mkdir -p uploads

# Copy the trained models (will be added by user)
# COPY runs/resnet50_v2/weights/best.pth runs/resnet50_v2/weights/
# COPY runs/ensemble/weights/*.pth runs/ensemble/weights/
# COPY runs/super_ensemble/weights/*.pth runs/super_ensemble/weights/

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default: Run FastAPI backend
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
