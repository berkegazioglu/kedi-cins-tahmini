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

# ── React frontend bağımlılıkları ─────────────
COPY frontend-react/ frontend-react/
WORKDIR /app/frontend-react

# LFS binary assets: curl ile indir (build context dışında bırakıldı)
RUN mkdir -p public/cat-sounds \
    && curl -sL "https://media.githubusercontent.com/media/berkegazioglu/kedi-cins-tahmini/hf-final/frontend-react/public/kedi-ai-logo.png" -o public/kedi-ai-logo.png \
    && curl -sL "https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?w=80&h=80&fit=crop" -o public/avatar-siamese.jpg \
    && curl -sL "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=80&h=80&fit=crop" -o public/avatar-bombay.jpg \
    && curl -sL "https://images.unsplash.com/photo-1548247416-ec66f4900b2e?w=80&h=80&fit=crop" -o public/avatar-balinese.jpg

RUN npm install --legacy-peer-deps
# VITE_API_URL="" → relative URL, same-origin API calls
RUN VITE_API_URL="" npm run build

WORKDIR /app

# ── Uygulama dosyaları ─────────────────────────
COPY backend/ backend/
COPY yolo11n.pt .
COPY cat_breed_info.json .
COPY *.pth .

RUN mkdir -p uploads runs/optimal_ensemble runs/resnet50_v2/weights

# optimal_ensemble_final.pth zaten kök dizinde — symlink ile beklenen yola bağla
RUN ln -sf /app/optimal_ensemble_final.pth /app/runs/optimal_ensemble/optimal_ensemble_final.pth 2>/dev/null || true

# ── HF Spaces zorunlu port: 7860 ──────────────
EXPOSE 7860

# ── Başlatma ──────────────────────────────────
CMD ["python", "-m", "uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
