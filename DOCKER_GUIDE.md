# 🐱 Kedi Cinsi Tahmin Sistemi - Docker Kurulum Rehberi

## 📋 Gereksinimler

- **Docker Desktop** (Windows/Mac/Linux)
- **NVIDIA GPU** (opsiyonel, CPU'da da çalışır)
- **NVIDIA Container Toolkit** (GPU kullanımı için)
- En az **8GB RAM**
- En az **10GB disk alanı**

## 🚀 Hızlı Başlangıç

### 1️⃣ Model Dosyalarını Hazırlayın

Aşağıdaki model dosyalarının proje kök dizininde olduğundan emin olun:

```
kedi-cins-tahmini/
├── yolo11n.pt                    # YOLO detection model
├── optimal_ensemble_final.pth     # Ensemble model (3 models + meta-learner)
├── EfficientNetB0_best.pth       # Base model 1
├── MobileNetV3_best.pth          # Base model 2
└── cat_breed_info.json           # Breed information
```

### 2️⃣ Docker ile Başlatma

**Windows PowerShell:**
```powershell
.\docker-start.ps1
```

**Linux/Mac:**
```bash
chmod +x docker-start.sh
./docker-start.sh
```

**Manuel Başlatma:**
```bash
# Build
docker-compose build backend

# Start
docker-compose up -d backend

# Logları görüntüle
docker-compose logs -f backend
```

### 3️⃣ Erişim

✅ Sistem başladıktan sonra:

- **Backend API:** http://localhost:8000
- **API Docs (Swagger):** http://localhost:8000/docs
- **API Docs (ReDoc):** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## 🧪 API Kullanımı

### Health Check

```bash
curl http://localhost:8000/health
```

### Tahmin (cURL)

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@cat_image.jpg" \
  -F "top_k=5"
```

### Tahmin (Python)

```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("cat_image.jpg", "rb")}
data = {"top_k": 5}

response = requests.post(url, files=files, data=data)
result = response.json()

print(f"Breed: {result['stage2']['top_breed']}")
print(f"Confidence: {result['stage2']['confidence']:.2%}")
```

### Tahmin (JavaScript/Fetch)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log('Breed:', result.stage2.top_breed);
console.log('Confidence:', result.stage2.confidence);
```

## 🛠️ Yönetim Komutları

### Container Durumu

```bash
docker-compose ps
```

### Logları Görüntüle

```bash
# Tüm loglar
docker-compose logs -f backend

# Son 100 satır
docker-compose logs --tail=100 backend
```

### Container'a Bağlan

```bash
docker-compose exec backend bash
```

### Durdur

```bash
docker-compose down
```

### Yeniden Başlat

```bash
docker-compose restart backend
```

### Image'ı Temizle

```bash
docker-compose down --rmi local
```

## 📊 Sistem Mimarisi

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
├─────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────┐  │
│  │            FastAPI Backend (Port 8000)       │  │
│  ├──────────────────────────────────────────────┤  │
│  │  Stage 1: YOLO11n Cat Detection             │  │
│  │  - Detects cat in image                      │  │
│  │  - Crops to cat region                       │  │
│  │  - Confidence threshold: 0.25                │  │
│  └──────────────┬───────────────────────────────┘  │
│                 ↓                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  Stage 2: Optimal Ensemble Classification   │  │
│  │  - ResNet50 (64.67%)                         │  │
│  │  - EfficientNetB0 (60.66%)                   │  │
│  │  - MobileNetV3-Large (60.06%)               │  │
│  │  - Meta-Learner (FC: 177→256→128→59)        │  │
│  │  Final Accuracy: 63.85%                      │  │
│  └──────────────────────────────────────────────┘  │
│                                                      │
│  GPU: NVIDIA CUDA 12.1 + cuDNN 8                   │
│  CPU: Multi-core support                            │
└─────────────────────────────────────────────────────┘
```

## 🔧 Yapılandırma

### GPU/CPU Seçimi

`docker-compose.yml` dosyasında GPU ayarlarını değiştirin:

```yaml
# GPU kullan
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]

# Sadece CPU (GPU satırlarını kaldırın)
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
```

### Port Değiştirme

```yaml
services:
  backend:
    ports:
      - "8000:8000"  # Sol: Host port, Sağ: Container port
```

### Volume Ayarları

```yaml
volumes:
  - ./backend:/app/backend:ro           # Read-only backend code
  - ./uploads:/app/uploads               # Writable upload directory
  - ./models:/app/models:ro              # Model files
```

## 🐛 Sorun Giderme

### Build Hatası: CUDA bulunamadı

**Çözüm:** CPU-only image kullanın:

```dockerfile
FROM pytorch/pytorch:2.1.0-py3.10-cuda11.8-cudnn8-runtime
# Değiştir:
FROM python:3.10-slim
```

### Container başlamıyor

```bash
# Logları kontrol et
docker-compose logs backend

# Container'ı yeniden build et
docker-compose build --no-cache backend
```

### GPU tanınmıyor

```bash
# NVIDIA Container Toolkit kontrolü
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Docker Compose ile test
docker-compose exec backend nvidia-smi
```

### Model dosyası bulunamıyor

```bash
# Container içindeki dosyaları kontrol et
docker-compose exec backend ls -lh /app/

# Volume mount'ları kontrol et
docker-compose exec backend ls -lh /app/*.pth
```

### Port zaten kullanımda

```bash
# Port kullanan process'i bul
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac

# Farklı port kullan
docker-compose down
# docker-compose.yml'de port değiştir
docker-compose up -d
```

## 📈 Performans İpuçları

### GPU Kullanımı

- **CUDA Memory:** ~3-4GB (ensemble model)
- **Inference Time:** ~200-300ms (GPU), ~2-3s (CPU)

### Optimizasyon

1. **Batch Processing:** Birden fazla resmi aynı anda işle
2. **Model Caching:** İlk inference'dan sonra model cache'lenir
3. **Skip Detection:** Eğer resimlerin tümü kedi içeriyorsa, YOLO'yu atlayın:

```python
response = requests.post(
    "http://localhost:8000/predict",
    files={"file": image_file},
    data={"skip_detection": "true"}
)
```

## 📚 Ek Kaynaklar

- **API Documentation:** http://localhost:8000/docs
- **Backend Kodu:** `backend/api/main.py`
- **Pipeline Kodu:** `backend/models/pipeline.py`
- **Model Mimarisi:** `docs/PIPELINE_ARCHITECTURE.md`

## 🆘 Destek

Sorun yaşarsanız:

1. Logları kontrol edin: `docker-compose logs backend`
2. Health check yapın: `curl http://localhost:8000/health`
3. Container'a bağlanın: `docker-compose exec backend bash`
4. GitHub Issues'da bildirin

---

**Başarılar! 🐱🎉**
