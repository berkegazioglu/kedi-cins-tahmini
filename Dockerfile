# Stage 1: Build Frontend
FROM node:18-alpine as frontend_builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# Stage 2: Python Runtime
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

# Copy requirements file (backend)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install huggingface_hub

# Copy application files
COPY app_resnet50.py .
COPY api.py .
COPY app.py .
COPY predict_resnet50.py .
COPY ensemble_model.py .
COPY transformer_models.py .
COPY augmentation_utils.py .
COPY cat_breed_info.py .
COPY show_classes.py .
COPY yolo11n.pt .
COPY windows.spec .

# Copy Streamlit config
COPY .streamlit .streamlit

# Create necessary directories
RUN mkdir -p runs/resnet50_v2/weights
RUN mkdir -p runs/optimal_ensemble

# Copy cat breed info database
COPY cat_breed_info.json .

# Copy compiled frontend from Stage 1
COPY --from=frontend_builder /app/frontend/dist ./frontend/dist

# Download models during build (since we utilize HF Spaces or similar, we might need them)
# But for now, we assume they might be downloaded via start script or handled by git-lfs if configured?
# The user's workflow downloads them via Python script in CI. 
# For Docker, we can try to copy them if they exist locally, or ignore if missing.
# We'll rely on the fact that for Spaces, we might need a different strategy, 
# but for "docker-compose up --build" locally, we want the local files if present.
# The previous Dockerfile tried to copy them.
COPY runs/resnet50_v2/weights/best.pth runs/resnet50_v2/weights/
# We ignore the error if source file is missing by using wildcard hack or just let it fail if user needs it?
# Actually, for the USER's local test, the files exist (we saw them logic).
# So we keep the COPY commands.

# Expose port 7860
EXPOSE 7860

# Run app.py
CMD ["python", "app.py"]
