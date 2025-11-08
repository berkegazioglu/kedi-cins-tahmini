# Kedi Cinsi Tanıma Sistemi 🐱

ResNet-50 ve YOLO11 kullanarak kedi cinslerini tanıyan yapay zeka projesi.

## 🚀 Hızlı Başlangıç (Docker ile)

### Gereksinimler
- Docker Desktop (GPU desteği için: NVIDIA Docker)
- Git

### Kurulum

1. **Projeyi klonlayın:**
```bash
git clone https://github.com/KULLANICI_ADINIZ/kedi-cins-tahmini.git
cd kedi-cins-tahmini
```

2. **Model dosyasını indirin:**
   - `best.pth` dosyasını [buradan](MODEL_LINK) indirin
   - `runs/resnet50/weights/` klasörüne yerleştirin

3. **Docker ile çalıştırın:**

**GPU ile (NVIDIA):**
```bash
docker-compose up -d
```

**CPU ile:**
```bash
docker build -t kedi-cins-tahmini .
docker run -p 8501:8501 -v ./runs/resnet50/weights/best.pth:/app/runs/resnet50/weights/best.pth kedi-cins-tahmini
```

4. **Tarayıcınızda açın:**
```
http://localhost:8501
```

## 📊 Model Özellikleri

- **Model:** ResNet-50 (Transfer Learning)
- **Eğitim:** 20 epoch, 88,741 train + 21,816 validation görüntü
- **Doğruluk:** ~58% validation accuracy
- **Sınıf Sayısı:** 59 kedi cinsi
- **Kedi Tespiti:** YOLO11n (ön filtre)

## 🛠️ Manuel Kurulum (Docker olmadan)

### Gereksinimler
- Python 3.11
- CUDA 12.1 (GPU için)

### Adımlar

1. **Virtual environment oluşturun:**
```bash
python -m venv .venv
```

2. **Aktive edin:**
```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

3. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

4. **Web arayüzünü başlatın:**
```bash
streamlit run app_resnet50.py
```

## 📁 Proje Yapısı

```
kedi-cins-tahmini/
├── app_resnet50.py           # Streamlit web arayüzü
├── train_resnet50.py         # Model eğitim scripti
├── predict_resnet50.py       # Tahmin scripti
├── evaluate_resnet50.py      # Model değerlendirme
├── sample_evaluate.py        # Hızlı değerlendirme
├── visualize_training.py     # Eğitim grafikleri
├── show_classes.py           # Sınıf listesi
├── yolo11n.pt               # YOLO11 kedi tespit modeli
├── Dockerfile               # Docker image tanımı
├── docker-compose.yml       # Docker compose yapılandırması
├── requirements.txt         # Python bağımlılıkları
└── runs/
    └── resnet50/
        └── weights/
            └── best.pth     # Eğitilmiş model (91.3 MB)
```

## 🎯 Kullanım

### Web Arayüzü
1. Tarayıcıda `http://localhost:8501` adresini açın
2. Kedi fotoğrafı yükleyin (drag & drop)
3. Tahmin sonuçlarını görün

### Komut Satırı
```bash
python predict_resnet50.py --image resim.jpg
```

## 🔧 Konfigürasyon

### GPU/CPU Seçimi
`app_resnet50.py` dosyasında:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### YOLO Tespit Eşiği
```python
CONFIDENCE_THRESHOLD = 0.15  # %15 güven eşiği
```

## 📈 Model Eğitimi

Kendi modelinizi eğitmek için:

```bash
python train_resnet50.py --epochs 20 --batch 16 --lr 0.001 --device cuda
```

**Parametreler:**
- `--epochs`: Epoch sayısı (varsayılan: 20)
- `--batch`: Batch boyutu (varsayılan: 32)
- `--lr`: Learning rate (varsayılan: 0.001)
- `--device`: cuda veya cpu (varsayılan: cuda)
- `--num-workers`: Veri yükleme worker sayısı (varsayılan: 4)

## 🐱 Desteklenen Kedi Cinsleri

Toplam 59 kedi cinsi desteklenmektedir:
- Abyssinian, American Bobtail, American Curl
- British Shorthair, Bengal, Birman
- Persian, Ragdoll, Siamese
- Scottish Fold, Sphynx, Maine Coon
- Ve daha fazlası...

Tam liste için:
```bash
python show_classes.py
```

## 📊 Performans Metrikleri

- **Validation Loss:** 1.5556
- **Validation Accuracy:** ~58%
- **Model Boyutu:** 91.3 MB
- **Inference Süresi:** ~50-100ms (GPU)

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/YeniOzellik`)
3. Commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Push edin (`git push origin feature/YeniOzellik`)
5. Pull Request oluşturun

## 📝 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 🙏 Teşekkürler

- Dataset: [Kaggle Cat Breeds Dataset]
- PyTorch & torchvision
- Ultralytics YOLO
- Streamlit

## 📞 İletişim

Sorularınız için: [GITHUB_KULLANICI_ADI]

---

**Not:** Model dosyası (`best.pth`) boyutu nedeniyle GitHub'a yüklenmemiştir. Lütfen release sayfasından veya Google Drive'dan indirin.
