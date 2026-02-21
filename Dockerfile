# ─────────────────────────────────────────────
#  Kedi.ai — HF Spaces & Local Docker Build
#  Port 7860 (HF Spaces zorunlu)
# ─────────────────────────────────────────────

# ── Stage 1: React frontend build ─────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend-react
COPY frontend-react/ .
RUN mkdir -p public/cat-sounds
RUN npm install --legacy-peer-deps
RUN VITE_API_URL="" npm run build

# ── Stage 2: Python backend ─────────────────────────────────
FROM python:3.10

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 \
    libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── PyTorch CPU (2.1.0 — well-established CPU wheels) ───────
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cpu

# ── Remaining Python deps ─────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Frontend dist (stage 1'den kopyala) ───────
COPY --from=frontend-builder /app/frontend-react/dist /app/frontend-react/dist

# ── Uygulama dosyaları ─────────────────────────
COPY backend/ backend/
COPY yolo11n.pt .
COPY cat_breed_info.json .
COPY *.pth .

RUN mkdir -p uploads runs/optimal_ensemble runs/resnet50_v2/weights
RUN ln -sf /app/optimal_ensemble_final.pth /app/runs/optimal_ensemble/optimal_ensemble_final.pth 2>/dev/null || true

# ── HF Spaces zorunlu port: 7860 ──────────────
EXPOSE 7860

CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
