# ─────────────────────────────────────────────
#  Kedi.ai — HF Spaces & Local Docker Build
#  Port 7860 (HF Spaces zorunlu)
# ─────────────────────────────────────────────
FROM python:3.11-slim

# ── Sistem bağımlılıkları + Node.js 20 ────────
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 git curl build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python bağımlılıkları (CPU PyTorch) ───────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── React frontend build ───────────────────────
COPY frontend-react/ frontend-react/
WORKDIR /app/frontend-react
RUN npm install --legacy-peer-deps
# VITE_API_URL="" → relative URL, same-origin API calls
RUN VITE_API_URL="" npm run build

WORKDIR /app

# ── Uygulama dosyaları ─────────────────────────
COPY backend/ backend/
COPY yolo11n.pt .
COPY cat_breed_info.json .
COPY *.pth .

# Runs klasöründeki model ağırlıkları (varsa)
COPY runs/ runs/

RUN mkdir -p uploads

# ── HF Spaces zorunlu port: 7860 ──────────────
EXPOSE 7860

# ── Başlatma ──────────────────────────────────
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
