# 🐱 Kedi Cinsi Tahmin Sistemi - Frontend

Modern React + Vite tabanlı web arayüzü ile kedi cinslerini tahmin edin.

## ✨ Özellikler

### 🎯 Model Seçimi
- **ResNet-50**: Derin öğrenme modeli (64.67% doğruluk)
- **EfficientNetB0**: Verimli ve hafif model
- **MobileNetV3**: Mobil cihazlar için optimize edilmiş
- **Optimal Ensemble**: Birleştirilmiş model sistemi

### 🔥 Temel Özellikler
- ✅ Tek model ile tahmin
- ✅ Tüm modelleri karşılaştırma modu
- ✅ YOLO11n ile kedi tespiti
- ✅ Entropi tabanlı vahşi kedi analizi
- ✅ 59 farklı kedi ırkı desteği
- ✅ Detaylı ırk bilgileri (sağlık, beslenme, bakım, karakter)
- ✅ Sürükle-bırak görüntü yükleme
- ✅ Responsive tasarım

### 📊 Karşılaştırma Özellikleri
- Her model için ayrı tahmin sonuçları
- Konsensüs (en yaygın tahmin)
- Model çalışma süreleri
- Yan yana görsel karşılaştırma

## 🚀 Kurulum

### Gereksinimler
- Node.js 16+ 
- npm veya yarn

### Adımlar

1. **Bağımlılıkları yükleyin:**
```bash
cd frontend
npm install
```

2. **Ortam değişkenlerini ayarlayın:**

`.env` dosyası zaten yapılandırılmış:
```env
VITE_API_URL=http://localhost:8000
```

Farklı bir backend URL'si kullanmak için bu dosyayı düzenleyin.

3. **Geliştirme sunucusunu başlatın:**
```bash
npm run dev
```

Uygulama varsayılan olarak `http://localhost:5173` adresinde çalışacaktır.

## 📦 Production Build

Production için optimize edilmiş build oluşturmak için:

```bash
npm run build
```

Build çıktısı `dist/` klasöründe oluşturulur.

Preview için:
```bash
npm run preview
```

## 🏗️ Proje Yapısı

```
frontend/
├── src/
│   ├── components/          # React bileşenleri
│   │   ├── ImageUploader.jsx       # Görüntü yükleme bileşeni
│   │   ├── ImageUploader.css
│   │   ├── PredictionResults.jsx   # Sonuç gösterimi
│   │   └── PredictionResults.css
│   ├── services/            # API servisleri
│   │   └── api.js          # Backend API entegrasyonu
│   ├── App.jsx             # Ana uygulama
│   ├── App.css
│   ├── main.jsx            # React giriş noktası
│   └── index.css
├── public/                 # Statik dosyalar
├── .env                    # Ortam değişkenleri
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## 🎨 Kullanım

### Tek Model ile Tahmin

1. Model seçin (ResNet-50, EfficientNetB0, MobileNetV3, Ensemble)
2. Kedi fotoğrafını yükleyin (sürükle-bırak veya dosya seç)
3. "Cinsi Tahmin Et" butonuna tıklayın
4. Sonuçları inceleyin:
   - En olası cins ve güven skoru
   - Diğer olası cinsler
   - Detaylı ırk bilgileri (5 sekme)

### Tüm Modelleri Karşılaştır

1. "Tüm Modelleri Karşılaştır" seçeneğini işaretleyin
2. Kedi fotoğrafını yükleyin
3. "Cinsi Tahmin Et" butonuna tıklayın
4. Tüm modellerin sonuçlarını karşılaştırın:
   - Konsensüs tahmini
   - Her model için ayrı tahmin ve güven skorları
   - Çalışma süreleri

### Kedi Tespitini Atlama

"Kedi Tespitini Atla (YOLO)" seçeneğini işaretleyerek YOLO11n kedi tespitini devre dışı bırakabilirsiniz. Bu, zaten kedi olduğunu bildiğiniz görüntüler için işlem süresini kısaltır.

## 🔧 API Entegrasyonu

### API Servisi (api.js)

Backend ile iletişim için `ApiService` sınıfı kullanılır:

```javascript
import ApiService from './services/api';

// Model listesini al
const models = await ApiService.getModels();

// Tek model ile tahmin
const result = await ApiService.predictBreed(imageFile, 'resnet50');

// Tüm modellerle tahmin
const comparison = await ApiService.predictAllModels(imageFile);

// Kedi tespiti
const detection = await ApiService.detectCat(imageFile);

// Irk bilgisi
const breedInfo = await ApiService.getBreedInfo('British Shorthair');
```

### Endpoints

- `GET /models` - Kullanılabilir modelleri listele
- `POST /predict` - Tek model ile tahmin
- `POST /predict-all` - Tüm modellerle tahmin
- `POST /detect` - YOLO kedi tespiti
- `GET /breeds` - Tüm ırkları listele
- `GET /breeds/{name}` - Irk detayları
- `GET /health` - Backend sağlık kontrolü

## 🎯 Bileşenler

### ImageUploader
Görüntü yükleme bileşeni:
- Sürükle-bırak desteği
- Dosya seçici
- Görüntü önizleme
- Yükleme durumu göstergesi

### PredictionResults
Tahmin sonuçları bileşeni:
- Tek model modu
- Karşılaştırma modu
- Vahşi kedi uyarıları
- Detaylı ırk bilgileri (5 sekme)
- İnteraktif güven çubukları

### App
Ana uygulama bileşeni:
- Model seçimi
- Kontrol paneli
- Bileşen koordinasyonu
- Hata yönetimi

## 🌐 Tarayıcı Desteği

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Opera 76+

## 📝 Geliştirme Notları

### Hot Module Replacement (HMR)
Vite, geliştirme sırasında hızlı yenileme için HMR kullanır. Değişiklikler anında tarayıcıda görünür.

### CSS Modülleri
Her bileşenin kendi CSS dosyası vardır. Global stiller için `index.css` kullanılır.

### Ortam Değişkenleri
- `VITE_API_URL`: Backend API URL'si
- Tüm Vite ortam değişkenleri `VITE_` ile başlamalıdır

## 🐛 Hata Ayıklama

### Backend bağlantı hatası
```
Error: Failed to fetch
```

**Çözüm:**
1. Backend'in çalıştığından emin olun: `http://localhost:8000`
2. `.env` dosyasındaki `VITE_API_URL` doğru mu kontrol edin
3. CORS ayarlarını kontrol edin

### Model yüklenemedi
```
Model 'xxx' not available
```

**Çözüm:**
1. Backend'de modellerin yüklendiğini kontrol edin
2. `GET /models` endpoint'ini çağırarak kullanılabilir modelleri görün
3. Backend loglarını inceleyin

## 📚 Teknolojiler

- **React 18**: UI kütüphanesi
- **Vite**: Build tool ve dev server
- **CSS3**: Styling (Gradients, Flexbox, Grid)
- **Fetch API**: HTTP istekleri
- **ES6+**: Modern JavaScript

## 🤝 Katkıda Bulunma

1. Yeni özellikler için branch oluşturun
2. Kodunuzu test edin
3. Commit mesajlarını anlamlı yazın
4. Pull request gönderin

## 📄 Lisans

MIT License

## 🙏 Teşekkürler

- PyTorch ve FastAPI ekiplerine
- Oxford-IIIT Pet Dataset
- React ve Vite topluluğuna

---

**Not:** Backend'in çalışır durumda olduğundan emin olun. Backend başlatma talimatları için ana `README.md` dosyasına bakın.
