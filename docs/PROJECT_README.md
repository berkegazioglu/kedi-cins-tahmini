# 🐱 Kedi Cinsi Tanıma Sistemi v2.0

ResNet-50 ve YOLO11 ile güçlendirilmiş, entropi tabanlı vahşi kedi tespiti özellikli AI kedi cinsi tanıma sistemi.

## 🏗️ Proje Yapısı

```
kedi-cins-tahmini/
├── backend/                    # FastAPI Backend
│   ├── api/                    # API endpoints
│   ├── config/                 # Konfigürasyon
│   ├── models/                 # Model yükleme
│   ├── utils/                  # Yardımcı fonksiyonlar
│   └── requirements.txt        # Backend bağımlılıkları
│
├── frontend-react/             # React Frontend (NEW)
│   ├── src/
│   │   ├── components/        # React bileşenleri
│   │   ├── services/          # API servisleri
│   │   └── styles/            # CSS dosyaları
│   └── package.json
│
├── cat_breed_info.json        # 59 cins detaylı bilgi
├── runs/                       # Eğitilmiş modeller
│   └── resnet50_v2/
│       └── weights/
│           └── best.pth
├── yolo11n.pt                 # YOLO11 model
├── images_split/              # Veri seti
├── legacy/                    # Eski dosyalar (Streamlit vb.)
└── README.md
```

## ✨ Özellikler

### Backend (FastAPI)
- ✅ **REST API** - Modern FastAPI tabanlı
- ✅ **Modüler Yapı** - Temiz kod organizasyonu
- ✅ **ResNet-50** - Transfer learning ile eğitilmiş
- ✅ **YOLO11** - Kedi tespiti
- ✅ **Entropi Analizi** - Vahşi kedi tespiti
- ✅ **59 Cins** - Detaylı bilgi kartları
- ✅ **CORS Desteği** - Frontend entegrasyonu
- ✅ **Auto Docs** - Swagger UI & ReDoc

### Frontend (React)
- ✅ **Modern UI** - React + Vite
- ✅ **Drag & Drop** - Kolay yükleme
- ✅ **Real-time** - Anında sonuçlar
- ✅ **Responsive** - Mobil uyumlu
- ✅ **Detaylı Bilgi** - Tab tabanlı görünüm

### AI Modeli
- ✅ **Top-1 Accuracy**: 56.95%
- ✅ **Top-3 Accuracy**: 75.05%
- ✅ **Top-5 Accuracy**: 83.35%
- ✅ **Vahşi Kedi Tespiti**: Entropi analizi

## 🚀 Hızlı Başlangıç

### Backend

```bash
# Backend dizinine git
cd backend

# Sanal ortam oluştur
python -m venv venv
venv\Scripts\activate  # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt

# API'yi başlat
cd api
python main.py
```

Backend: http://localhost:8000
API Docs: http://localhost:8000/docs

### Frontend

```bash
# Frontend dizinine git
cd frontend-react

# Bağımlılıkları yükle
npm install

# Development sunucusunu başlat
npm run dev
```

Frontend: http://localhost:5173

## 📖 Kullanım

### 1. Backend API

```python
import requests

# Tahmin
with open('kedi.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f}
    )
    print(response.json())
```

### 2. React Frontend

```javascript
import ApiService from './services/api';

// Tahmin
const result = await ApiService.predictBreed(file, {
  topK: 5,
  skipDetection: false
});
```

### 3. cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@kedi.jpg"
```

## 🎯 API Endpoints

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| GET | `/` | API bilgileri |
| GET | `/health` | Sağlık kontrolü |
| GET | `/breeds` | Tüm cinsler |
| GET | `/breeds/{name}` | Cins detayları |
| POST | `/predict` | Tahmin yap |
| POST | `/detect` | Kedi tespiti |

## 🔧 Konfigürasyon

### Backend (.env)
```bash
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
USE_GPU=false
```

### Frontend (.env)
```bash
VITE_API_URL=http://localhost:8000
```

## 📊 Model Bilgileri

### ResNet-50 v2
- **Epoch**: 23
- **Val Accuracy**: 64.67%
- **Architecture**: Transfer Learning
- **Dataset**: 59 cat breeds
- **Training**: ImageNet pretrained

### YOLO11n
- **Task**: Cat detection
- **Model**: YOLOv11 nano
- **COCO Class**: 15 (cat)
- **Threshold**: 0.15

## 🦁 Vahşi Kedi Tespiti

Sistem, entropi analizi kullanarak vahşi kedi türlerini tespit eder:

- **Aslan, Kaplan, Leopar, Çita**
- **Vaşak, Puma, Jaguar**
- **Serval, Oselot, Caracal**

**Nasıl Çalışır?**
1. Model tahmin yapar
2. Shannon entropisi hesaplanır
3. Entropi > 2.5 ve güven < %40 ise vahşi kedi uyarısı verilir

## 📚 Cins Bilgileri

Her cins için:
- ✅ Genel bilgi (menşei, boyut, ağırlık)
- ✅ Sağlık özellikleri
- ✅ Beslenme önerileri
- ✅ Bakım gereksinimleri
- ✅ Karakter özellikleri
- ✅ Uyumluluk metrikleri

## 🛠️ Geliştirme

### Backend Test
```bash
cd backend/api
python -c "from main import app; print('Backend OK')"
```

### Frontend Test
```bash
cd frontend-react
npm run lint
npm run build
```

### Yeni Özellik Ekleme

1. **Backend**: `backend/api/main.py` - Yeni endpoint ekle
2. **Frontend**: `src/services/api.js` - API fonksiyonu ekle
3. **Component**: `src/components/` - UI bileşeni oluştur

## 📝 Değişiklik Geçmişi

### v2.0.0 (2026-02-19)
- ✨ Backend/Frontend ayrımı
- ✨ FastAPI REST API
- ✨ React Frontend
- ✨ Modüler kod yapısı
- ✨ Entropi tabanlı vahşi kedi tespiti
- ✨ 59 cins detaylı bilgi kartları
- ❌ Streamlit kaldırıldı (legacy'ye taşındı)

### v1.0.0
- 🎉 İlk sürüm
- ResNet-50 model
- YOLO11 entegrasyonu

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun
3. Değişikliklerinizi commit edin
4. Branch'inizi push edin
5. Pull Request açın

## 📄 Lisans

MIT License

## 👤 Geliştirici

**Berke Gazioğlu**
- GitHub: [@berkegazioglu](https://github.com/berkegazioglu)

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
