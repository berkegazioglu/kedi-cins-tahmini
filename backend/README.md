# Backend API - Kedi Cinsi Tanıma Sistemi

FastAPI tabanlı REST API backend servisi.

## 🏗️ Yapı

```
backend/
├── api/
│   └── main.py              # FastAPI uygulaması ve endpoint'ler
├── config/
│   └── settings.py          # Konfigürasyon ayarları
├── models/
│   └── loader.py            # Model yükleme ve yönetimi
├── utils/
│   └── inference.py         # Tahmin ve analiz fonksiyonları
└── requirements.txt         # Backend bağımlılıkları
```

## 🚀 Kurulum

1. **Sanal ortam oluştur:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

2. **Bağımlılıkları yükle:**
```bash
pip install -r requirements.txt
```

3. **Çevre değişkenleri (opsiyonel):**
```bash
# .env dosyası oluştur
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
USE_GPU=false
```

## 🎯 Kullanım

### Geliştirme Modu (Hot Reload)
```bash
cd backend/api
python main.py
```

### Production Modu
```bash
cd backend/api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

API şu adreste çalışacak: `http://localhost:8000`

## 📚 API Endpoints

### Genel

- `GET /` - API bilgileri
- `GET /health` - Sağlık kontrolü
- `GET /docs` - Swagger UI (otomatik dokümantasyon)
- `GET /redoc` - ReDoc dokümantasyonu

### Cinsler

- `GET /breeds` - Tüm kedi cinslerini listele
- `GET /breeds/{breed_name}` - Belirli bir cins hakkında detaylı bilgi

### Tahmin

- `POST /predict` - Kedi cinsi tahmini
  - Body: `multipart/form-data`
  - Field: `file` (image file)
  - Optional: `skip_detection` (bool)
  - Optional: `top_k` (int, default: 5)

- `POST /detect` - Sadece kedi tespiti
  - Body: `multipart/form-data`
  - Field: `file` (image file)

## 📖 Örnek Kullanım

### Python ile
```python
import requests

# Tahmin
with open('kedi.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict',
        files={'file': f},
        params={'top_k': 5}
    )
    result = response.json()
    print(result['predictions'])
```

### cURL ile
```bash
# Tahmin
curl -X POST "http://localhost:8000/predict" \
  -F "file=@kedi.jpg" \
  -F "top_k=5"

# Cins listesi
curl "http://localhost:8000/breeds"

# Cins bilgisi
curl "http://localhost:8000/breeds/Persian"
```

### JavaScript/Fetch ile
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

## 🔧 Özellikler

- ✅ ResNet-50 tabanlı cins tahmini
- ✅ YOLO11 ile kedi tespiti
- ✅ Entropi analizi ile vahşi kedi tespiti
- ✅ 59 farklı kedi cinsi desteği
- ✅ Detaylı cins bilgileri (sağlık, beslenme, bakım vb.)
- ✅ CORS desteği
- ✅ Otomatik API dokümantasyonu
- ✅ Hata yönetimi
- ✅ Dosya boyutu ve tip validasyonu

## 📊 Response Formatı

### Başarılı Tahmin
```json
{
  "success": true,
  "predictions": [
    {
      "breed": "Persian",
      "confidence": 85.32
    }
  ],
  "entropy": 2.15,
  "is_wild_cat": false,
  "wild_warning": null,
  "top_breed": "Persian",
  "top_confidence": 85.32,
  "detection": {
    "cat_found": true,
    "confidence": 0.95,
    "message": "Cat detected",
    "skipped": false
  },
  "breed_info": {
    "name_tr": "İran Kedisi",
    "origin": "İran",
    ...
  }
}
```

### Kedi Tespit Edilemedi
```json
{
  "success": false,
  "error": "No cat detected in image",
  "detection": {
    "cat_found": false,
    "confidence": 0.0,
    "message": "No objects detected"
  }
}
```

## ⚙️ Konfigürasyon

`config/settings.py` dosyasından ayarlar yapılabilir:

- Model yolları
- API ayarları
- Cihaz seçimi (CPU/GPU)
- Eşik değerleri
- Maksimum dosya boyutu

## 🐛 Hata Ayıklama

Debug modu için:
```bash
cd backend/api
python main.py --reload --log-level debug
```

## 📝 Notlar

- Model dosyaları `runs/resnet50_v2/weights/best.pth` konumunda olmalı
- YOLO modeli `yolo11n.pt` kök dizinde olmalı
- Cins bilgileri `cat_breed_info.json` kök dizinde olmalı
