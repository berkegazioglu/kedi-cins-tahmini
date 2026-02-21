# 🐱 Kedi Cinsi Tahmin Sistemi - Multi-Model AI

Modern FastAPI backend ve React frontend ile çalışan gelişmiş kedi cinsi tanıma sistemi. 4 farklı derin öğrenme modeli ile 59 farklı kedi ırkını tahmin eder.

## 🌟 Temel Özellikler

### 🤖 Multi-Model AI Sistemi
- **ResNet-50 v2**: 64.67% doğruluk, derin öğrenme baseline
- **EfficientNetB0**: Verimli ve kompakt model
- **MobileNetV3**: Mobil cihazlar için optimize edilmiş
- **Optimal Ensemble**: Birleştirilmiş model sistemi

### 🎯 Akıllı Analiz
- ✅ Model karşılaştırma modu (4 modeli aynı anda çalıştırma)
- ✅ Konsensüs tahmini (en yaygın sonuç)
- ✅ YOLO11n ile otomatik kedi tespiti
- ✅ Shannon entropi ile vahşi kedi analizi (threshold: 2.5)
- ✅ 59 farklı kedi ırkı desteği
- ✅ Detaylı ırk bilgileri (sağlık, beslenme, bakım, karakter)

### 💻 Modern Mimari
- **Backend**: FastAPI (async REST API)
- **Frontend**: React 18 + Vite
- **Model**: PyTorch 2.5.1
- **Detection**: YOLO11n (Ultralytics)
- **Database**: JSON-based breed info

## 🚀 Hızlı Başlangıç

### Gereksinimler

#### Backend
- Python 3.9+
- PyTorch 2.5.1+
- CUDA (opsiyonel, GPU desteği için)

#### Frontend
- Node.js 16+
- npm veya yarn

### Kurulum

#### 1. Backend Kurulumu

```bash
# Proje dizinine gidin
cd backend

# Virtual environment oluşturun
python -m venv venv

# Virtual environment'ı aktif edin
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

# Backend'i başlatın
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Backend şu adreste çalışacak: `http://localhost:8000`

#### 2. Frontend Kurulumu

```bash
# Frontend dizinine gidin
cd frontend

# Bağımlılıkları yükleyin
npm install

# Development server'ı başlatın
npm run dev
```

Frontend şu adreste çalışacak: `http://localhost:5173`

### Model Dosyaları

Aşağıdaki model dosyalarının proje kök dizininde olması gerekir:

```
kedi-cins-tahmini/
├── best_cat_breed_model_resnet50_v2.pth  # ResNet-50 v2
├── MobileNetV3_best.pth                   # MobileNetV3
├── EfficientNetB0_best.pth                # EfficientNetB0
├── optimal_ensemble_final.pth             # Optimal Ensemble
└── yolo11n.pt                             # YOLO11n
```

## 📁 Proje Yapısı

```
kedi-cins-tahmini/
├── backend/                        # FastAPI Backend
│   ├── api/
│   │   └── main.py                # Ana API endpoint'leri
│   ├── config/
│   │   └── settings.py            # Konfigürasyon
│   ├── models/
│   │   └── loader.py              # Model yükleme sistemi
│   ├── utils/
│   │   └── inference.py           # Tahmin ve analiz
│   └── requirements.txt           # Python bağımlılıkları
│
├── frontend/                       # React Frontend
│   ├── src/
│   │   ├── components/            # React bileşenleri
│   │   ├── services/              # API servisleri
│   │   ├── App.jsx                # Ana uygulama
│   │   └── main.jsx
│   ├── package.json
│   └── README.md
│
├── cat_breed_info.json            # 59 ırk detaylı bilgi
├── best_cat_breed_model_resnet50_v2.pth
├── MobileNetV3_best.pth
├── EfficientNetB0_best.pth
├── optimal_ensemble_final.pth
├── yolo11n.pt
├── README.md                      # Bu dosya
└── MULTI_MODEL_GUIDE.md          # Multi-model sistem rehberi
```

## 🎯 Kullanım

### Web Arayüzü

1. **Tarayıcınızda açın**: `http://localhost:5173`

2. **Model seçimi yapın**:
   - Tek model: Dropdown'dan bir model seçin
   - Karşılaştırma: "Tüm Modelleri Karşılaştır" seçeneğini işaretleyin

3. **Görüntü yükleyin**:
   - Sürükle-bırak ile
   - "Dosya Seç" butonu ile

4. **Tahmin et**: "Cinsi Tahmin Et" butonuna tıklayın

5. **Sonuçları inceleyin**:
   - En olası cins ve güven skoru
   - Diğer olasılıklar
   - Model karşılaştırması (compare mode'da)
   - Detaylı ırk bilgileri

### API Kullanımı

#### Mevcut Modelleri Listele
```bash
curl http://localhost:8000/models
```

#### Tek Model ile Tahmin
```bash
curl -X POST "http://localhost:8000/predict?model=resnet50" \
  -F "file=@kedi.jpg"
```

#### Tüm Modellerle Tahmin
```bash
curl -X POST "http://localhost:8000/predict-all" \
  -F "file=@kedi.jpg"
```

#### Kedi Tespiti
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@kedi.jpg"
```

#### Irk Bilgisi
```bash
curl http://localhost:8000/breeds/British%20Shorthair
```

## 🔧 API Endpoint'leri

### Model Yönetimi
- `GET /` - API bilgisi ve yüklü modeller
- `GET /models` - Kullanılabilir modeller
- `GET /health` - Sistem sağlık kontrolü

### Tahmin
- `POST /predict` - Tek model ile tahmin
  - Parameters: `model` (str), `skip_detection` (bool), `top_k` (int)
- `POST /predict-all` - Tüm modellerle karşılaştırmalı tahmin
  - Parameters: `skip_detection` (bool), `top_k` (int)

### Tespit
- `POST /detect` - YOLO11n ile kedi tespiti

### Irk Bilgisi
- `GET /breeds` - Tüm ırkları listele
- `GET /breeds/{name}` - Belirli bir ırk hakkında detaylı bilgi

## 📊 Model Performansları

| Model | Doğruluk | Parametreler | Çıkarım Hızı | Kullanım Alanı |
|-------|----------|--------------|--------------|----------------|
| ResNet-50 v2 | 64.67% | 25.6M | ~0.15s | Genel kullanım, baseline |
| EfficientNetB0 | TBD | 5.3M | ~0.10s | Verimli, hızlı |
| MobileNetV3 | TBD | 4.2M | ~0.08s | Mobil, IoT |
| Optimal Ensemble | TBD | Combined | ~0.40s | Maksimum doğruluk |

*Çıkarım hızları CPU (Intel i7) üzerinde ölçülmüştür.*

## 🧠 Vahşi Kedi Tespiti

Sistem, Shannon entropisi kullanarak vahşi kedi türlerini (aslan, kaplan, vaşak vb.) tespit eder:

- **Entropi Threshold**: 2.5
- **Metod**: Softmax çıktısının entropi analizi
- **Sonuç**: Yüksek entropi → Belirsiz tahmin → Muhtemel vahşi kedi

Vahşi kedi tespit edildiğinde:
- ⚠️ Uyarı gösterilir
- Tahmin güvenilirliği düşük kabul edilir
- Entropi değeri raporlanır

## 🐈 Desteklenen Kedi Irkları (59 Adet)

<details>
<summary>Tüm ırkları görmek için tıklayın</summary>

- Abyssinian
- American Bobtail
- American Curl
- American Shorthair
- American Wirehair
- Applehead Siamese
- Balinese
- Bengal
- Birman
- Bombay
- British Shorthair
- Burmese
- Burmilla
- Calico
- Canadian Hairless (Sphynx)
- Chartreux
- Chausie
- Chinchilla
- Cornish Rex
- Cymric
- Devon Rex
- Dilute Calico
- Dilute Tortoiseshell
- Domestic Long Hair
- Domestic Medium Hair
- Domestic Short Hair
- Egyptian Mau
- Exotic Shorthair
- Extra-Toes Cat (Hemingway Polydactyl)
- Havana
- Himalayan
- Japanese Bobtail
- Javanese
- Korat
- LaPerm
- Maine Coon
- Manx
- Munchkin
- Nebelung
- Norwegian Forest Cat
- Ocicat
- Oriental Long Hair
- Oriental Short Hair
- Oriental Tabby
- Persian
- Pixiebob
- Ragamuffin
- Ragdoll
- Russian Blue
- Scottish Fold
- Selkirk Rex
- Siamese
- Siberian
- Singapura
- Snowshoe
- Somali
- Sphynx
- Tabby
- Turkish Angora

</details>

## 🔬 Teknik Detaylar

### Backend (FastAPI)

#### Model Yükleme Sistemi
```python
# backend/config/settings.py
AVAILABLE_MODELS = {
    'resnet50': 'best_cat_breed_model_resnet50_v2.pth',
    'efficientnet': 'EfficientNetB0_best.pth',
    'mobilenet': 'MobileNetV3_best.pth',
    'ensemble': 'optimal_ensemble_final.pth'
}
```

#### Model Cache
- Tüm modeller başlangıçta yüklenir
- Memory'de cache edilir
- Her istek için yeniden yükleme gerekmez

#### Inference Pipeline
1. Image preprocessing (224x224, normalize)
2. YOLO cat detection (optional)
3. Model forward pass
4. Softmax + Top-K predictions
5. Entropy analysis
6. Breed info retrieval

### Frontend (React + Vite)

#### State Management
- React hooks (useState, useEffect)
- No external state library
- Simple and performant

#### API Communication
- Fetch API
- FormData for file upload
- Async/await pattern

#### UI Components
- ImageUploader: Drag & drop, file select
- PredictionResults: Single & compare modes
- App: Main orchestrator

## 🚨 Troubleshooting

### Backend başlamıyor
```
ModuleNotFoundError: No module named 'fastapi'
```
**Çözüm**: `pip install -r backend/requirements.txt`

### Model yüklenemiyor
```
FileNotFoundError: [Errno 2] No such file or directory: 'best_cat_breed_model_resnet50_v2.pth'
```
**Çözüm**: Model dosyalarının proje kök dizininde olduğundan emin olun

### Frontend backend'e bağlanamıyor
```
Error: Failed to fetch
```
**Çözüm**: 
1. Backend'in çalıştığını kontrol edin: `http://localhost:8000`
2. `.env` dosyasını kontrol edin
3. CORS ayarlarını gözden geçirin

### CUDA hatası
```
RuntimeError: CUDA out of memory
```
**Çözüm**: CPU moduna geçin veya batch size azaltın

## 📈 Performans Optimizasyonu

### Backend
- Model caching (startup'ta yükleme)
- Async FastAPI endpoints
- Efficient image preprocessing
- PyTorch inference mode

### Frontend
- Vite fast HMR
- Lazy loading
- Optimized images
- Minimal re-renders

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'feat: Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📝 Changelog

### v2.0.0 (Multi-Model Release)
- ✨ 4 farklı model desteği eklendi
- ✨ Model karşılaştırma modu
- ✨ Konsensüs tahmini
- ✨ Model çalışma süreleri
- ✨ Backend/Frontend ayrımı
- ✨ Modern React UI
- 🔧 Model caching sistemi
- 🔧 FastAPI async endpoints

### v1.0.0 (Initial Release)
- ✅ ResNet-50 v2 model
- ✅ YOLO11n kedi tespiti
- ✅ Entropi analizi
- ✅ 59 ırk desteği
- ✅ Streamlit UI

## 📄 Lisans

MIT License

## 🙏 Teşekkürler

- **PyTorch Team**: Deep learning framework
- **FastAPI**: Modern web framework
- **React Team**: UI library
- **Ultralytics**: YOLO implementation
- **Oxford-IIIT Pet Dataset**: Training data

## 📧 İletişim

Sorularınız veya önerileriniz için issue açabilirsiniz.

---

**Happy Cat Breeding! 🐱✨**
