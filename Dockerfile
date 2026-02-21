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

# Kedi ırk görselleri: Unsplash'tan indir
RUN mkdir -p public/cats \
    && curl -sL "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=300&fit=crop" -o public/cats/persian.jpg \
    && curl -sL "https://images.unsplash.com/photo-1594900060009-89cda7e5a3cd?w=400&h=300&fit=crop" -o public/cats/maine-coon.jpg \
    && curl -sL "https://images.unsplash.com/photo-1533743983669-94fa5c4338ec?w=400&h=300&fit=crop" -o public/cats/british-shorthair.jpg \
    && curl -sL "https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?w=400&h=300&fit=crop" -o public/cats/siamese.jpg \
    && curl -sL "https://images.unsplash.com/photo-1550159930-40066082a4fc?w=400&h=300&fit=crop" -o public/cats/abyssinian.jpg \
    && curl -sL "https://images.unsplash.com/photo-1601979031925-424e53b6caaa?w=400&h=300&fit=crop" -o public/cats/bengal.jpg \
    && curl -sL "https://images.unsplash.com/photo-1519479651571-a2c9e714fee2?w=400&h=300&fit=crop" -o public/cats/russian-blue.jpg \
    && curl -sL "https://images.unsplash.com/photo-1555685812-4b943f1cb0eb?w=400&h=300&fit=crop" -o public/cats/birman.jpg \
    && curl -sL "https://images.unsplash.com/photo-1568043432885-75710c77fd61?w=400&h=300&fit=crop" -o public/cats/american-shorthair.jpg \
    && curl -sL "https://images.unsplash.com/photo-1573865526739-10659fec78a5?w=400&h=300&fit=crop" -o public/cats/bombay.jpg \
    && curl -sL "https://images.unsplash.com/photo-1548247416-ec66f4900b2e?w=400&h=300&fit=crop" -o public/cats/norwegian-forest-cat.jpg \
    && curl -sL "https://images.unsplash.com/photo-1596854407944-bf87f6fdd49e?w=400&h=300&fit=crop" -o public/cats/sphynx.jpg \
    && curl -sL "https://images.unsplash.com/photo-1571566882372-1598d88abd90?w=400&h=300&fit=crop" -o public/cats/scottish-fold.jpg \
    && curl -sL "https://images.unsplash.com/photo-1561948955-570b270e7c36?w=400&h=300&fit=crop" -o public/cats/chartreux.jpg \
    && curl -sL "https://images.unsplash.com/photo-1590418606746-018840f9ced2?w=400&h=300&fit=crop" -o public/cats/turkish-angora.jpg \
    && curl -sL "https://images.unsplash.com/photo-1529778873920-4da4926a72c2?w=400&h=300&fit=crop" -o public/cats/ocicat.jpg \
    && curl -sL "https://images.unsplash.com/photo-1535293478971-0a5e7a6c2230?w=400&h=300&fit=crop" -o public/cats/tonkinese.jpg \
    && curl -sL "https://images.unsplash.com/photo-1535295972055-1c762f4483e5?w=400&h=300&fit=crop" -o public/cats/burmese.jpg \
    && curl -sL "https://images.unsplash.com/photo-1518791841217-8f162f1912da?w=400&h=300&fit=crop" -o public/cats/savannah.jpg \
    && curl -sL "https://images.unsplash.com/photo-1543852786-1cf6624b9987?w=400&h=300&fit=crop" -o public/cats/munchkin.jpg \
    && curl -sL "https://images.unsplash.com/photo-1591825729269-caeb344f6df2?w=400&h=300&fit=crop" -o public/cats/cornish-rex.jpg \
    && curl -sL "https://images.unsplash.com/photo-1535268647677-300dbf3d78d1?w=400&h=300&fit=crop" -o public/cats/havana-brown.jpg \
    && curl -sL "https://images.unsplash.com/photo-1609220136736-443140cfeaa8?w=400&h=300&fit=crop" -o public/cats/manx.jpg \
    && curl -sL "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400&h=300&fit=crop" -o public/cats/exotic-shorthair.jpg \
    && curl -sL "https://images.unsplash.com/photo-1568043432885-75710c77fd61?w=400&h=300&fit=crop" -o public/cats/american-bobtail.jpg \
    && curl -sL "https://images.unsplash.com/photo-1591825729269-caeb344f6df2?w=400&h=300&fit=crop" -o public/cats/selkirk-rex.jpg \
    && curl -sL "https://images.unsplash.com/photo-1513360371669-4adf3dd7dff8?w=400&h=300&fit=crop" -o public/cats/snowshoe.jpg

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
