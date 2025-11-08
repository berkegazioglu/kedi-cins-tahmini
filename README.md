# 🐱 Kedi Cinsi Tahmin Sistemi - ResNet-50

Derin öğrenme kullanarak kedi fotoğraflarından cins tahmini yapan bir yapay zeka projesidir. ResNet-50 transfer learning modeli ile 59 farklı kedi cinsini yüksek doğrulukla tanır.

## � Hızlı Başlangıç (Docker - Önerilen)

```bash
# 1. Projeyi klonlayın
git clone https://github.com/KULLANICI_ADINIZ/kedi-cins-tahmini.git
cd kedi-cins-tahmini

# 2. Docker container'ı başlatın (Windows)
.\deploy.ps1

# Linux/Mac için
chmod +x deploy.sh
./deploy.sh

# 3. Tarayıcıda açın
http://localhost:8501
```

**Not:** Docker kullanımı için detaylı bilgi: [README_DOCKER.md](README_DOCKER.md)

---

## �📊 Proje Özeti

- **Model**: ResNet-50 (ImageNet pretrained, transfer learning)
- **Framework**: PyTorch + torchvision
- **Dataset**: ~110,000 kedi görüntüsü, 59 sınıf
- **GPU**: NVIDIA RTX 3050 (CUDA 12.1)
- **Web Arayüzü**: Streamlit
- **Performans** (2 epoch, 2000 sample):
  - Top-1 Accuracy: 56.95%
  - Top-3 Accuracy: 75.05%
  - Top-5 Accuracy: 83.35%

## 🚀 Özellikler

✅ **Transfer Learning**: ImageNet ağırlıkları ile pretrained ResNet-50  
✅ **Robust Training**: Corrupt image handling ile dayanıklı eğitim  
✅ **GPU Hızlandırma**: CUDA desteği ile hızlı eğitim ve inference  
✅ **Web Arayüzü**: Kullanıcı dostu Streamlit uygulaması  
✅ **Kapsamlı Değerlendirme**: Accuracy, confusion matrix, per-class metrics  
✅ **Görselleştirme**: Training curves ve evaluation plots  

## 📁 Proje Yapısı

```
kedi-cins-tahmini/
├── app_resnet50.py              # Streamlit web uygulaması
├── train_resnet50.py            # Model eğitim scripti
├── predict_resnet50.py          # Tek görüntü tahmin scripti
├── sample_evaluate.py           # Hızlı model değerlendirme
├── visualize_training.py        # Training curve görselleştirme
├── check_model.py               # Model checkpoint inceleme
│
├── data/
│   └── cats.csv                 # Dataset metadata
│
├── images_split/
│   ├── train/                   # Eğitim seti (~88,741 görüntü)
│   └── val/                     # Validation seti (~21,816 görüntü)
│
├── runs/
│   └── resnet50/
│       ├── weights/
│       │   ├── best.pth         # En iyi model (91.3 MB)
│       │   ├── last.pth         # Son checkpoint
│       │   └── epoch_*.pth      # Epoch checkpoints
│       ├── plots/
│       │   └── val_loss.png     # Training loss curve
│       └── evaluation/
│           └── sample_results.txt
│
└── README.md
```

## 🛠️ Kurulum

### 1. Python Sanal Ortamı Oluşturma

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Gerekli Kütüphaneleri Yükleme

```powershell
# PyTorch (CUDA 12.1 desteği ile)
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Diğer gereksinimler
pip install pillow numpy matplotlib seaborn scikit-learn tqdm streamlit pandas
```

### 3. GPU Kontrolü

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

## 🎯 Kullanım

### 1. Model Eğitimi

```powershell
# Yeni eğitim başlatma (20 epoch)
python train_resnet50.py --epochs 20 --batch 16 --device cuda --num-workers 4

# Mevcut modelden devam etme
python train_resnet50.py --epochs 20 --batch 16 --device cuda --resume runs/resnet50/weights/last.pth
```

**Eğitim Parametreleri:**
- `--epochs`: Eğitim epoch sayısı (varsayılan: 20)
- `--batch`: Batch size (varsayılan: 16)
- `--lr`: Learning rate (varsayılan: 0.001)
- `--device`: cuda veya cpu (varsayılan: cuda)
- `--num-workers`: DataLoader worker sayısı (varsayılan: 4)
- `--resume`: Checkpoint'ten devam etme

### 2. Model Değerlendirme

```powershell
# Hızlı sample evaluation (2000 görüntü)
python sample_evaluate.py --model runs/resnet50/weights/best.pth --sample-size 2000

# Training curve görselleştirme
python visualize_training.py --logdir runs/resnet50
```

### 3. Tahmin Yapma

```powershell
# Tek görüntü tahmini
python predict_resnet50.py --image path/to/cat.jpg --model runs/resnet50/weights/best.pth --top-k 5
```

### 4. Web Arayüzü

```powershell
streamlit run app_resnet50.py
```

Tarayıcınızda `http://localhost:8501` adresini açın.

## 📈 Model Performansı

### Sample Evaluation Sonuçları (2000 görüntü)

| Metric | Değer |
|--------|-------|
| **Top-1 Accuracy** | 56.95% |
| **Top-3 Accuracy** | 75.05% |
| **Top-5 Accuracy** | 83.35% |

### En Başarılı Sınıflar

1. **Domestic Short Hair**: 97.24% (979 samples)
2. **Persian**: 88.89% (72 samples)
3. **Siamese**: 44.26% (61 samples)

### Örnek Tahminler

- **Persian kedi**: 98.60% güven
- **Calico kedi**: 78.15% güven

## 🏗️ Model Mimarisi

### ResNet-50 Transfer Learning

```
ResNet-50 (ImageNet pretrained)
├── Frozen Layers
│   ├── conv1, bn1, relu, maxpool
│   ├── layer1 (3 bottleneck blocks)
│   ├── layer2 (4 bottleneck blocks)
│   ├── layer3 (6 bottleneck blocks)
│   └── layer4 (3 bottleneck blocks)
│
└── Trainable Layers
    └── fc (fully connected): 2048 → 59 classes

Total Parameters: 23,628,923
Trainable Parameters: 120,891 (only FC layer)
```

### Eğitim Stratejisi

- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: CrossEntropyLoss
- **Batch Size**: 16
- **Image Size**: 224x224
- **Data Augmentation**:
  - Random horizontal flip
  - Random rotation (±15°)
  - Color jitter
  - Normalization (ImageNet stats)

### Robust Error Handling

Corrupt image'ler için 3 katmanlı hata yakalama:
1. `RobustImageFolder`: Dataset seviyesinde None döndürme
2. `robust_collate_fn`: Batch oluşturma sırasında filtreleme
3. `SafeDataLoader`: DataLoader seviyesinde exception handling

## 📊 Veri Seti

- **Toplam Görüntü**: ~110,000
- **Eğitim Seti**: ~88,741 görüntü
- **Validation Seti**: ~21,816 görüntü
- **Sınıf Sayısı**: 59 kedi cinsi
- **Kaynak**: Kaggle Cat Breed Classification Dataset

### Sınıf Örnekleri

Abyssinian, American Bobtail, Bengal, British Shorthair, Calico, Domestic Short Hair, Exotic Shorthair, Himalayan, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx, Turkish Angora, vb.

## 🔧 Teknik Detaylar

### Gereksinimler

- Python 3.11+
- CUDA 12.1+ (GPU kullanımı için)
- Windows 10/11 (multiprocessing için freeze_support)
- En az 8GB RAM
- En az 4GB VRAM (GPU için)

### Önemli Notlar

1. **Windows Multiprocessing**: `if __name__ == '__main__':` guard ve `multiprocessing.freeze_support()` gereklidir.

2. **Corrupt Images**: Dataset'te bazı corrupt JPEG dosyaları vardır. `RobustImageFolder` bunları otomatik atlıyor.

3. **num_workers**: 
   - Stability için: `num_workers=0`
   - Speed için: `num_workers=4`

4. **OneDrive Sync**: OneDrive klasöründe çalışıyorsanız, bazen import işlemleri yavaş olabilir.

## 📝 Geliştirme Önerileri

### Kısa Vadeli İyileştirmeler

1. **Daha Fazla Eğitim**: 20 epoch'a tamamla (şu anda 2 epoch)
2. **Fine-tuning**: Backbone'u unfreeze edip düşük learning rate ile ince ayar
3. **Data Augmentation**: Daha agresif augmentation teknikleri
4. **Class Balancing**: Weighted sampling veya class weights kullanımı

### Uzun Vadeli İyileştirmeler

1. **Model Ensemble**: Birden fazla model kombinasyonu
2. **Two-Stage System**: 
   - Stage 1: Cat detection (YOLO)
   - Stage 2: Breed classification (ResNet-50)
3. **Attention Mechanisms**: Dikkat mekanizmaları ekleme
4. **Larger Models**: ResNet-101, EfficientNet, ViT denemeleri
5. **Active Learning**: Zor örnekleri manuel etiketleme

## 🐛 Bilinen Sorunlar

1. **Yavaş Evaluation**: 21,816 görüntülü tam evaluation yavaş → `sample_evaluate.py` kullanın
2. **Import Slowness**: OneDrive'da scipy/sklearn import'ları yavaş olabilir
3. **Corrupt Images**: Bazı validation görüntüleri corrupt → RobustImageFolder ile handle ediliyor
4. **Class Imbalance**: Domestic Short Hair çok baskın (979/2000 sample)

## 📚 Referanslar

- **ResNet Paper**: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- **Transfer Learning**: "A Survey on Transfer Learning" (Pan & Yang, 2010)
- **PyTorch**: https://pytorch.org/
- **Streamlit**: https://streamlit.io/

## 👨‍💻 Geliştirici

Proje, NVIDIA RTX 3050 GPU ile PyTorch kullanılarak geliştirilmiştir.

## 📄 Lisans

Bu proje eğitim amaçlıdır.

## 🙏 Teşekkürler

- PyTorch ekibine derin öğrenme framework'ü için
- Streamlit ekibine web arayüzü framework'ü için
- Kaggle topluluğuna dataset için
- NVIDIA'ya CUDA desteği için

---

**Proje Durumu**: ✅ Aktif Geliştirme  
**Son Güncelleme**: 8 Kasım 2025  
**Versiyon**: 1.0.0
