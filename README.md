---
title: Kedi.ai
emoji: 🐱
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
short_description: Yapay Zeka ile kedi ırkı tanımlama — 59 cins, ücretsiz
tags:
  - cat
  - breed
  - classification
  - pytorch
  - computer-vision
  - fastapi
  - react
pinned: true
---

# 🐱 Kedi Cinsi Tahmin Uygulaması

Derin öğrenme tabanlı gelişmiş kedi cinsi sınıflandırma sistemi. 59 farklı kedi cinsini yüksek doğrulukla tahmin eder.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Model Mimarisi](#model-mimarisi)
- [Kurulum](#kurulum)
  - [Docker ile Kurulum (Önerilen)](#docker-ile-kurulum-önerilen)
  - [Manuel Kurulum](#manuel-kurulum)
- [Backend & Frontend Başlatma](#️-backend--frontend-başlatma)
- [Kullanım](#kullanım)
- [Model Eğitimi](#model-eğitimi)
- [Performans](#performans)
- [Veri Seti](#veri-seti)
- [Teknolojiler](#teknolojiler)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## ✨ Özellikler

- 🎯 **59 Kedi Cinsi Desteği**: Abyssinian'dan Tabby'ye kadar geniş cins yelpazesi
- 🐱 **YOLO11 Cat Detection**: Fotoğrafta kedi var mı yok mu otomatik tespit
- 🧠 **Ensemble Model**: ResNet50 + EfficientNet-B0 + MobileNetV3 kombinasyonu
- ⚡ **Mixed Precision Training**: FP16 desteği ile hızlı eğitim
- 🔄 **Gradient Accumulation**: Düşük VRAM için optimize edilmiş
- 🛡️ **Anti-Overfitting**: Strong augmentation, label smoothing, early stopping
- 🐳 **Docker Desteği**: Kolay deployment ve reproducibility
- 📊 **Detaylı Raporlama**: Training history, confusion matrix, performance metrics
- 🔍 **Akıllı Ön Kontrol**: Kedi olmayan fotoğrafları otomatik filtreler

## 🏗️ Model Mimarisi

### İki Aşamalı Pipeline

#### 1. Aşama: YOLO11 Cat Detection
```
┌─────────────────────────────────────┐
│        Input Image (any size)       │
└────────────────┬────────────────────┘
                 │
          ┌──────▼──────┐
          │   YOLO11n   │  ← Pre-trained
          │  (2.6M)     │     Object Detector
          └──────┬──────┘
                 │
        ┌────────▼─────────┐
        │  Cat Detection?  │
        │  (Confidence >   │
        │     0.25)        │
        └────┬────────┬────┘
             │        │
          NO │        │ YES
             │        │
      ┌──────▼──┐  ┌──▼─────────┐
      │  Reject │  │  Continue  │
      │  Image  │  │ to Stage 2 │
      └─────────┘  └────────────┘
```

#### 2. Aşama: Optimal Ensemble Classification
```
┌─────────────────────────────────────────────┐
│          Optimal 3-Model Ensemble           │
├─────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐         │
│  │  ResNet50    │  │ EfficientNet │         │
│  │  (24.6M)     │  │  -B0 (5.3M)  │         │
│  │  64.67%      │  │  60.66%      │         │
│  └──────┬───────┘  └──────┬───────┘         │
│         │                 │                  │
│         │   ┌─────────────┴────────┐         │
│         │   │   MobileNetV3-Large  │         │
│         │   │      (5.4M)          │         │
│         │   │      60.06%          │         │
│         │   └──────────┬───────────┘         │
│         │              │                     │
│         └──────┬───────┘                     │
│                │                             │
│         ┌──────▼──────────┐                  │
│         │  Meta-Learner   │                  │
│         │  (FC Layers)    │                  │
│         └──────┬──────────┘                  │
│                │                             │
│         ┌──────▼──────────┐                  │
│         │ Final Prediction│                  │
│         │    63.85%       │                  │
│         └─────────────────┘                  │
└─────────────────────────────────────────────┘
```

### Teknik Özellikler

#### YOLO11 Detection Stage
- **Model**: YOLO11n (Nano)
- **Parameters**: 2.6M
- **Purpose**: Cat presence detection
- **Confidence Threshold**: 0.25
- **Speed**: ~50ms per image (GPU)

#### Classification Stage
- **Total Parameters**: ~35.3M
- **Mixed Precision**: FP16 (VRAM %50 azaltma)
- **Batch Size**: 8 (Virtual: 32 with gradient accumulation)
- **Image Size**: 224x224
- **Augmentation**: RandomCrop, ColorJitter, Rotation, Erasing
- **Regularization**: Dropout, Label Smoothing, Weight Decay

### Neden İki Aşamalı Sistem?

1. **Doğruluk Artışı**: Kedi olmayan görseller erken filtrelenir
2. **Hata Azaltma**: Classification modeli sadece kedi fotoğraflarına odaklanır
3. **Kullanıcı Deneyimi**: Anlamlı hata mesajları ("Bu fotoğrafta kedi bulunamadı")
4. **Performans**: YOLO hızlı pre-check, gereksiz classification önlenir

## 🚀 Kurulum

### Docker ile Kurulum (Önerilen)

#### Gereksinimler
- Docker Desktop (Windows/Mac) veya Docker Engine (Linux)
- NVIDIA GPU (opsiyonel, CPU'da da çalışır)
- NVIDIA Container Toolkit (GPU kullanımı için)

#### 1. Repoyu Klonlayın
```bash
git clone https://github.com/berkegazioglu/kedi-cins-tahmini.git
cd kedi-cins-tahmini
```

#### 2. Model Dosyalarını İndirin (Git LFS)
Proje model dosyalarını **Git LFS** (Large File Storage) ile yönetir. Model dosyalarını indirmek için:

```bash
# Git LFS'i yükleyin (henüz yüklü değilse)
# Windows (Git for Windows ile gelir)
# Linux: sudo apt-get install git-lfs
# Mac: brew install git-lfs

# Git LFS'i başlatın
git lfs install

# Model dosyalarını çekin
git lfs pull
```

**Not**: `git clone` komutu otomatik olarak LFS dosyalarını çeker, ancak bazı durumlarda `git lfs pull` komutunu manuel olarak çalıştırmanız gerekebilir.

**İndirilen Modeller**:
- `yolo11n.pt` (5.3 MB) - YOLO11 cat detection model (pre-trained, already included)
- `runs/resnet50_v2/weights/best.pth` (270 MB) - ResNet50 model, %64.67 accuracy
- `runs/optimal_ensemble/optimal_ensemble_final.pth` (122 MB) - Ensemble model, %63.85 accuracy

#### 3. Docker Image Build Edin
```bash
# CPU versiyonu
docker-compose build

# GPU versiyonu (NVIDIA GPU gerekli)
docker-compose -f docker-compose.yml build
```

#### 4. Uygulamayı Başlatın
```bash
# Web uygulamasını başlat
docker-compose up

# Arka planda çalıştır
docker-compose up -d
```

#### 5. Tarayıcıda Açın
```
http://localhost:8501
```

#### 6. Durdurma
```bash
docker-compose down
```

### Manuel Kurulum

#### Gereksinimler
- Python 3.11+
- CUDA 12.1+ (GPU için)
- 4GB+ RAM (CPU) veya 4GB+ VRAM (GPU)

#### 1. Repoyu Klonlayın
```bash
git clone https://github.com/berkegazioglu/kedi-cins-tahmini.git
cd kedi-cins-tahmini
```

#### 2. Model Dosyalarını İndirin (Git LFS)
```bash
# Git LFS'i yükleyin (henüz yüklü değilse)
git lfs install

# Model dosyalarını çekin
git lfs pull
```

#### 3. Virtual Environment Oluşturun
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 4. Gereksinimleri Yükleyin
```bash
# GPU versiyonu (NVIDIA CUDA gerekli)
pip install -r requirements.txt

# CPU versiyonu
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

#### 5. Uygulamayı Başlatın
```bash
# Streamlit web uygulaması (ResNet50 - en iyi model)
streamlit run app_resnet50.py
```

---

## 🖥️ Backend & Frontend Başlatma

Bu proje iki bileşenden oluşur: **Flask API (Backend)** ve **React/Vite (Frontend)**. Her ikisinin de ayrı ayrı başlatılması gerekir.

### 1️⃣ Backend'i Başlatın (Flask API — Port 8002)

Backend, model tahminlerini REST API olarak sunar. Docker ile çalışır:

```bash
# Docker image'ı build edin (ilk seferde)
docker build -t kedi-backend .

# Container'ı başlatın (Port 8002)
docker run -d --name kedi-backend -p 8002:8002 kedi-backend

# Zaten oluşturulduysa, sadece yeniden başlatın
docker restart kedi-backend

# Logları takip edin
docker logs -f kedi-backend
```

Backend çalışıyor mu kontrol edin:
```bash
curl http://localhost:8002/health
# veya tarayıcıda: http://localhost:8002/health
```

> **Not:** Backend hazır olduğunda `"status": "ok"` yanıtı döner.

---

### 2️⃣ Frontend'i Başlatın (React/Vite — Port 5173)

Frontend, `frontend-react/` klasöründeki React uygulamasıdır:

```bash
# frontend-react klasörüne gidin
cd frontend-react

# Bağımlılıkları yükleyin (ilk seferde)
npm install

# Geliştirme sunucusunu başlatın
npm run dev
```

Tarayıcıda açın:
```
http://localhost:5173
```

**Windows PowerShell ile tek komutta:**
```powershell
Set-Location "frontend-react"; npm run dev
```

---

### ⚡ Her İkisini Birden Başlatma

```powershell
# 1. Backend'i arka planda başlat
docker restart kedi-backend

# 2. Frontend'i başlat
Set-Location "frontend-react"
npm run dev
```

### 🛑 Durdurma

```bash
# Backend'i durdur
docker stop kedi-backend

# Frontend: terminalde Ctrl+C
```

### 🔧 Bağlantı Yapılandırması

Frontend, backend API adresini `frontend-react/src/config.js` dosyasından okur:

```js
// frontend-react/src/config.js
export const API_BASE_URL = "http://localhost:8002";
```

Farklı bir port kullanıyorsanız bu dosyayı güncelleyin.

---

## 💻 Kullanım

### Web Arayüzü ile Kullanım

1. Uygulamayı başlatın (Docker veya manuel)
2. Tarayıcıda açın: `http://localhost:8501`
3. "Browse files" ile fotoğraf yükleyin
4. **Otomatik YOLO Kontrolü**: 
   - ✅ Kedi tespit edilirse → Classification'a geçer
   - ❌ Kedi tespit edilmezse → "Bu fotoğrafta kedi bulunamadı" uyarısı
5. "Tahmin Et" butonuna tıklayın
6. Sonuçları görün:
   - En olası 5 cins
   - Güven yüzdeleri
   - YOLO detection confidence
   - Her modelin tahmini

### Python API ile Kullanım (YOLO + Classification)

```python
from PIL import Image
import torch
from torchvision import transforms
from ultralytics import YOLO

# 1. YOLO ile kedi tespiti
yolo_model = YOLO('yolo11n.pt')
image = Image.open('test_image.jpg').convert('RGB')

results = yolo_model(image, verbose=False)
cat_detected = False

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        
        # Class 15 = cat in COCO dataset
        if class_id == 15 and confidence > 0.25:
            cat_detected = True
            print(f"🐱 Kedi tespit edildi! (Confidence: {confidence:.2%})")
            break

if not cat_detected:
    print("❌ Bu fotoğrafta kedi bulunamadı!")
    exit()

# 2. Kedi cinsi classification
from train_optimal_ensemble import OptimalEnsemble

model = OptimalEnsemble(num_classes=59)
model.load_state_dict(torch.load('runs/optimal_ensemble/optimal_ensemble_final.pth'))
model.eval()

# Görüntü hazırlama
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

input_tensor = transform(image).unsqueeze(0)

# Tahmin
with torch.no_grad():
    output = model(input_tensor, use_meta=True)
    probabilities = torch.softmax(output, dim=1)
    top5_prob, top5_idx = torch.topk(probabilities, 5)

print(f"Top 5 Tahminler:")
for prob, idx in zip(top5_prob[0], top5_idx[0]):
    print(f"  {class_names[idx]}: {prob*100:.2f}%")
```

### Komut Satırı ile Kullanım

```bash
# Tekil tahmin
python predict_optimal_ensemble.py --image path/to/cat.jpg

# Batch tahmin
python predict_optimal_ensemble.py --folder path/to/cat_images/

# Detaylı analiz
python predict_optimal_ensemble.py --image cat.jpg --detailed
```

## 🎓 Model Eğitimi

### Hızlı Başlangıç

```bash
# Optimal ensemble eğitimi (önerilen)
python train_optimal_ensemble.py

# Tek model eğitimi
python train_resnet50.py
```

### Eğitim Parametreleri

```python
# train_optimal_ensemble.py içinde ayarlar
BATCH_SIZE = 8              # Gerçek batch size
ACCUMULATION_STEPS = 4      # Sanal batch = 32
EPOCHS_BASE = 15            # Base model epoch'ları
EPOCHS_META = 10            # Meta-learner epoch'ları
EPOCHS_FINE = 5             # Fine-tuning epoch'ları
```

### Özel Eğitim

```bash
# Custom parameters
python train_optimal_ensemble.py \
    --batch-size 16 \
    --epochs-base 20 \
    --epochs-meta 12 \
    --epochs-fine 8 \
    --lr 0.001
```

### Eğitim İzleme

```bash
# Training history görselleştirme
python visualize_training.py

# TensorBoard (opsiyonel)
tensorboard --logdir runs/optimal_ensemble
```

## 📊 Performans

### Model Karşılaştırması

| Model | Parameters | Accuracy | Training Time | VRAM Usage | Inference Speed |
|-------|------------|----------|---------------|------------|-----------------|
| **YOLO11n** (Detection) | **2.6M** | **-** | **Pre-trained** | **0.5 GB** | **~50ms** |
| ResNet50 | 24.6M | 64.67% | ~4 hours | 3.2 GB | ~100ms |
| EfficientNet-B0 | 5.3M | 60.66% | ~4.5 hours | 2.8 GB | ~120ms |
| MobileNetV3 | 5.4M | 60.06% | ~2.5 hours | 2.5 GB | ~80ms |
| **Optimal Ensemble** | **35.3M** | **63.85%** | **~16 hours** | **3.8 GB** | **~150ms** |

**Total Pipeline**: YOLO (50ms) + Ensemble (150ms) = **~200ms per image**

### YOLO11 Detection Performance

| Metric | Value |
|--------|-------|
| Model | YOLO11n (Nano) |
| Parameters | 2.6M |
| Cat Detection Accuracy | ~95% (COCO pre-trained) |
| Confidence Threshold | 0.25 |
| False Positive Rate | <5% |
| Speed (GPU) | ~50ms/image |
| Speed (CPU) | ~200ms/image |

### Cins Bazlı Performance (Top 10)

| Kedi Cinsi | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| British Shorthair | 78.3% | 82.1% | 80.1% |
| Persian | 76.5% | 79.8% | 78.1% |
| Siamese | 74.2% | 77.3% | 75.7% |
| Maine Coon | 72.8% | 75.6% | 74.2% |
| Bengal | 71.3% | 73.9% | 72.6% |
| Russian Blue | 69.7% | 72.4% | 71.0% |
| Ragdoll | 68.4% | 71.2% | 69.8% |
| Sphynx | 67.2% | 69.8% | 68.5% |
| Abyssinian | 65.9% | 68.5% | 67.2% |
| Scottish Fold | 64.7% | 67.1% | 65.9% |

### Hardware Gereksinimleri

**Minimum (CPU):**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 8 GB
- Disk: 10 GB
- Inference: ~2-3 saniye/görüntü

**Önerilen (GPU):**
- GPU: NVIDIA RTX 3050 (4GB) veya üstü
- RAM: 16 GB
- VRAM: 4 GB
- Disk: 20 GB
- Inference: ~0.1 saniye/görüntü

## 📁 Veri Seti

### Yapı
```
images_split/
├── train/              # 88,741 görüntü
│   ├── Abyssinian/
│   ├── American Bobtail/
│   ├── ...
│   └── Tabby/
└── val/                # 21,816 görüntü
    ├── Abyssinian/
    ├── American Bobtail/
    ├── ...
    └── Tabby/
```

### Desteklenen Kedi Cinsleri (59)

<details>
<summary>Tüm cinsleri göster</summary>

1. Abyssinian
2. American Bobtail
3. American Curl
4. American Shorthair
5. American Wirehair
6. Applehead Siamese
7. Balinese
8. Bengal
9. Birman
10. Bombay
11. British Shorthair
12. Burmese
13. Burmilla
14. Calico
15. Canadian Hairless (Sphynx)
16. Chartreux
17. Chausie
18. Chinchilla
19. Cornish Rex
20. Cymric
21. Devon Rex
22. Dilute Calico
23. Dilute Tortoiseshell
24. Domestic Long Hair
25. Domestic Medium Hair
26. Domestic Short Hair
27. Egyptian Mau
28. Exotic Shorthair
29. Extra-Toes Cat (Polydactyl)
30. Havana
31. Himalayan
32. Japanese Bobtail
33. Javanese
34. Korat
35. LaPerm
36. Maine Coon
37. Manx
38. Munchkin
39. Nebelung
40. Norwegian Forest Cat
41. Ocicat
42. Oriental Long Hair
43. Oriental Short Hair
44. Oriental Tabby
45. Persian
46. Pixiebob
47. Ragamuffin
48. Ragdoll
49. Russian Blue
50. Scottish Fold
51. Selkirk Rex
52. Siamese
53. Siberian
54. Silver
55. Singapura
56. Snowshoe
57. Somali
58. Sphynx
59. Tabby

</details>

### Veri Artırma (Augmentation)

```python
# Training augmentations
- RandomResizedCrop(224, scale=(0.7, 1.0))
- RandomHorizontalFlip(p=0.5)
- RandomRotation(20°)
- ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
- RandomAffine(translate=(0.1, 0.1))
- RandomErasing(p=0.3, scale=(0.02, 0.15))
```

## 🛠️ Teknolojiler

### Core
- **Python 3.11**: Ana programlama dili
- **PyTorch 2.5.1**: Deep learning framework
- **CUDA 12.1**: GPU acceleration

### Deep Learning
- **torchvision**: Pretrained models ve transforms
- **timm**: Advanced model architectures (EfficientNet, MobileNetV3)
- **torch.cuda.amp**: Mixed precision training
- **Ultralytics YOLO**: Object detection (YOLO11n for cat detection)

### Web & API
- **Streamlit**: Web UI
- **Flask**: REST API
- **Pillow**: Image processing

### Data & Visualization
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization
- **tqdm**: Progress bars

### DevOps
- **Docker**: Containerization
- **docker-compose**: Multi-container orchestration

## 📂 Proje Yapısı

```
kedi-cins-tahmini/
├── README.md                          # Bu dosya
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker image tanımı
├── docker-compose.yml                 # Docker compose yapılandırması
├── deploy.sh                          # Linux deployment script
├── deploy.ps1                         # Windows deployment script
├── github-push.ps1                    # GitHub push script
│
├── data/
│   └── cats.csv                       # Metadata
│
├── images_split/
│   ├── train/                         # Training images
│   └── val/                           # Validation images
│
├── runs/
│   ├── resnet50_v2/                   # ResNet50 model
│   │   └── weights/
│   │       └── best.pth
│   ├── optimal_ensemble/              # Final ensemble
│   │   ├── optimal_ensemble_final.pth
│   │   ├── training_history.json
│   │   └── class_names.json
│   └── super_ensemble/                # Experimental models
│
├── uploads/                           # User uploaded images
│
├── __pycache__/                       # Python cache
│
├── models/
│   ├── ensemble_model.py              # ResNet50, ConvNeXt models
│   ├── transformer_models.py          # ViT, EfficientNetV2 models
│   └── augmentation_utils.py          # Training utilities
│
├── training/
│   ├── train_optimal_ensemble.py      # Final optimal ensemble trainer
│   ├── train_resnet50.py              # Single ResNet50 trainer
│   ├── train_resnet50_v2.py           # ResNet50 v2 trainer
│   ├── train_ensemble.py              # 3-model ensemble trainer
│   ├── train_super_ensemble.py        # 4-model super ensemble
│   ├── train_fast_ensemble.py         # Fast ensemble variant
│   └── train_ensemble_2models.py      # 2-model baseline
│
├── evaluation/
│   ├── evaluate_optimal_ensemble.py   # Ensemble evaluation
│   ├── evaluate_resnet50.py           # ResNet50 evaluation
│   ├── quick_evaluate.py              # Quick test
│   └── sample_evaluate.py             # Sample testing
│
├── prediction/
│   ├── predict_optimal_ensemble.py    # Ensemble prediction
│   ├── predict_resnet50.py            # ResNet50 prediction
│   └── predict_ensemble.py            # Basic ensemble prediction
│
├── apps/
│   ├── app_optimal_ensemble.py        # Optimal ensemble web app
│   ├── app_resnet50.py                # ResNet50 web app
│   └── app_ensemble.py                # Basic ensemble web app
│
├── visualization/
│   ├── visualize_training.py          # Training curves
│   └── visualize_ensemble_training.py # Ensemble analysis
│
├── utils/
│   ├── check_model.py                 # Model checker
│   ├── show_classes.py                # Class list viewer
│   └── test_ensemble.py               # Unit tests
│
└── experiments/
    ├── test_super_ensemble.py         # Super ensemble tests
    └── test_optimal_ensemble.py       # Optimal ensemble tests
```

## 🐳 Docker Detayları

### Dockerfile Açıklaması

```dockerfile
# Base image: Python 3.11 with CUDA support
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "app_optimal_ensemble.py", "--server.address=0.0.0.0"]
```

### Docker Compose Yapılandırması

```yaml
version: '3.8'

services:
  cat-classifier:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./runs:/app/runs
      - ./uploads:/app/uploads
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Docker Komutları

```bash
# Build
docker build -t cat-classifier .

# Run (CPU)
docker run -p 8501:8501 cat-classifier

# Run (GPU)
docker run --gpus all -p 8501:8501 cat-classifier

# Interactive shell
docker run -it cat-classifier /bin/bash

# View logs
docker logs -f <container-id>

# Stop
docker stop <container-id>

# Remove
docker rm <container-id>
docker rmi cat-classifier
```

## 🔧 Yapılandırma

### Eğitim Parametreleri

`train_optimal_ensemble.py` dosyasında:

```python
# Hardware Configuration
BATCH_SIZE = 8                    # Gerçek batch size
ACCUMULATION_STEPS = 4            # Gradient accumulation
NUM_WORKERS = 4                   # DataLoader workers

# Training Epochs
EPOCHS_BASE = 15                  # Base model epochs
EPOCHS_META = 10                  # Meta-learner epochs
EPOCHS_FINE = 5                   # Fine-tuning epochs

# Optimization
LEARNING_RATE = 0.001             # Initial learning rate
WEIGHT_DECAY = 0.01               # L2 regularization
DROPOUT = 0.5                     # Dropout rate

# Regularization
LABEL_SMOOTHING = 0.1             # Label smoothing
EARLY_STOPPING_PATIENCE = 7       # Early stopping patience
GRADIENT_CLIP = 1.0               # Gradient clipping
```

### Model Seçimi

```python
# Config: model_config.py (oluşturulabilir)
MODEL_TYPE = "optimal_ensemble"   # veya "resnet50", "ensemble"
USE_MIXED_PRECISION = True        # FP16 training
USE_GRADIENT_ACCUMULATION = True  # Memory optimization
```

## 📈 İlerleme Takibi

### Training History

```python
# Training history JSON formatı
{
    "resnet50": {
        "note": "Best pre-trained model",
        "val_acc": 64.67
    },
    "efficientnet_b0": {
        "train_loss": [2.51, 2.35, ...],
        "train_acc": [48.22, 49.67, ...],
        "val_loss": [2.82, 2.29, ...],
        "val_acc": [54.50, 55.94, ...]
    },
    "mobilenet": { ... },
    "meta_learner": { ... },
    "fine_tuning": { ... }
}
```

### Görselleştirme

```bash
# Training curves
python visualize_training.py --history runs/optimal_ensemble/training_history.json

# Confusion matrix
python evaluate_optimal_ensemble.py --confusion-matrix

# Per-class accuracy
python evaluate_optimal_ensemble.py --per-class
```

## 🧪 Test

### Unit Tests

```bash
# Tüm testleri çalıştır
python -m pytest tests/

# Specific test
python test_ensemble.py
```

### Model Validation

```bash
# Model integrity check
python check_model.py

# Quick validation
python quick_evaluate.py

# Full evaluation
python evaluate_optimal_ensemble.py
```

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

### Geliştirme Ortamı

```bash
# Development dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks
pre-commit install

# Code formatting
black .
isort .

# Linting
flake8 .
pylint *.py
```

## 🐛 Sorun Giderme

### CUDA Out of Memory

```python
# Batch size'ı düşürün
BATCH_SIZE = 4

# veya gradient accumulation artırın
ACCUMULATION_STEPS = 8
```

### Model Yükleme Hatası

```bash
# Model dosyasını kontrol edin
python check_model.py --model runs/optimal_ensemble/optimal_ensemble_final.pth

# Yeniden indirin
python download_models.py
```

### Docker Build Hatası

```bash
# Cache temizle
docker system prune -a

# Rebuild
docker-compose build --no-cache
```

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 👨‍💻 Yazar

**Berke Gazioğlu**
- GitHub: [@berkegazioglu](https://github.com/berkegazioglu)

## 🙏 Teşekkürler

- PyTorch ekibine deep learning framework için
- Kaggle'a veri seti için
- Açık kaynak topluluğuna pretrained modeller için
- Tüm katkıda bulunanlara

## 📚 Referanslar

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
2. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling"
3. Howard, A., et al. (2019). "Searching for MobileNetV3"
4. Zhang, H., et al. (2017). "mixup: Beyond Empirical Risk Minimization"

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!

📧 Sorularınız için: [Issue açın](https://github.com/berkegazioglu/kedi-cins-tahmini/issues)
