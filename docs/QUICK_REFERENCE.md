# 🎯 Multi-Model System Quick Reference

## 📋 Quick Commands

### Start Everything
```powershell
# Interactive menu
.\start.ps1

# Or manually:
# Terminal 1 - Backend
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend  
cd frontend
npm run dev
```

### Run Tests
```bash
python test_multi_model.py
```

## 🔌 API Quick Reference

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 📊 System Info
```bash
# Root - API info
curl http://localhost:8000/

# Health check
curl http://localhost:8000/health

# Available models
curl http://localhost:8000/models
```

#### 🔮 Predictions

**Single Model**
```bash
# With ResNet-50
curl -X POST "http://localhost:8000/predict?model=resnet50&skip_detection=false&top_k=5" \
  -F "file=@cat.jpg"

# With EfficientNetB0
curl -X POST "http://localhost:8000/predict?model=efficientnet" \
  -F "file=@cat.jpg"

# With MobileNetV3
curl -X POST "http://localhost:8000/predict?model=mobilenet" \
  -F "file=@cat.jpg"

# With Ensemble
curl -X POST "http://localhost:8000/predict?model=ensemble" \
  -F "file=@cat.jpg"
```

**All Models (Compare)**
```bash
curl -X POST "http://localhost:8000/predict-all?skip_detection=true&top_k=5" \
  -F "file=@cat.jpg"
```

#### 🐱 Detection & Breeds
```bash
# YOLO cat detection
curl -X POST "http://localhost:8000/detect" \
  -F "file=@image.jpg"

# List all breeds
curl http://localhost:8000/breeds

# Get breed info
curl http://localhost:8000/breeds/British%20Shorthair
```

## 🎨 Frontend URLs

- **Main App**: http://localhost:5173
- **API Proxy**: http://localhost:5173/api/* → http://localhost:8000/*

## 🤖 Available Models

| Model | Key | File | Use Case |
|-------|-----|------|----------|
| ResNet-50 v2 | `resnet50` | `best_cat_breed_model_resnet50_v2.pth` | General purpose, baseline |
| EfficientNetB0 | `efficientnet` | `EfficientNetB0_best.pth` | Efficient, fast |
| MobileNetV3 | `mobilenet` | `MobileNetV3_best.pth` | Mobile, IoT |
| Optimal Ensemble | `ensemble` | `optimal_ensemble_final.pth` | Best accuracy |

## 📁 Important Files

### Backend Configuration
```
backend/config/settings.py     - All configuration
backend/models/loader.py       - Model loading
backend/utils/inference.py     - Prediction logic
backend/api/main.py           - API endpoints
```

### Frontend
```
frontend/src/services/api.js           - API client
frontend/src/components/ImageUploader.jsx  - Upload UI
frontend/src/components/PredictionResults.jsx - Results UI
frontend/src/App.jsx                   - Main app
```

### Data
```
cat_breed_info.json           - 59 breeds info
yolo11n.pt                    - YOLO detection model
*.pth                         - PyTorch classification models
```

## 🔧 Configuration

### Backend Settings (backend/config/settings.py)
```python
# Model paths
AVAILABLE_MODELS = {
    'resnet50': 'best_cat_breed_model_resnet50_v2.pth',
    'efficientnet': 'EfficientNetB0_best.pth',
    'mobilenet': 'MobileNetV3_best.pth',
    'ensemble': 'optimal_ensemble_final.pth'
}

# Detection settings
YOLO_MODEL_PATH = 'yolo11n.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.5
ENTROPY_THRESHOLD = 2.5

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
```

### Frontend Settings (frontend/.env)
```env
VITE_API_URL=http://localhost:8000
```

## 📊 Response Formats

### Single Model Prediction
```json
{
  "success": true,
  "top_breed": "British Shorthair",
  "top_confidence": 0.8542,
  "predictions": [
    {"breed": "British Shorthair", "confidence": 0.8542},
    {"breed": "Russian Blue", "confidence": 0.0823}
  ],
  "is_wild_cat": false,
  "entropy": 1.234,
  "model_used": "resnet50",
  "detection": {
    "cat_found": true,
    "confidence": 0.95,
    "message": "Cat detected",
    "skipped": false
  },
  "breed_info": {
    "name_tr": "British Shorthair",
    "origin": "İngiltere",
    "size": "Orta-Büyük",
    ...
  }
}
```

### All Models Comparison
```json
{
  "success": true,
  "consensus": "British Shorthair",
  "predictions": {
    "resnet50": {
      "top_breed": "British Shorthair",
      "top_confidence": 0.8542,
      "inference_time": 0.145,
      ...
    },
    "efficientnet": {...},
    "mobilenet": {...},
    "ensemble": {...}
  },
  "timing": {
    "resnet50": 0.145,
    "efficientnet": 0.098,
    "mobilenet": 0.082,
    "ensemble": 0.412
  },
  "models_used": ["resnet50", "efficientnet", "mobilenet", "ensemble"]
}
```

## 🚨 Common Issues

### "Model not loaded"
**Solution**: Check that .pth files are in project root

### "CUDA out of memory"  
**Solution**: Models automatically use CPU if CUDA unavailable

### "No module named 'fastapi'"
**Solution**: `pip install -r backend/requirements.txt`

### "Cannot GET /"
**Solution**: Backend not running, start with `uvicorn backend.api.main:app`

### Frontend can't connect to backend
**Solution**: Check CORS settings and `.env` file

## 📈 Performance Tips

### Backend
- Use `skip_detection=true` to bypass YOLO (faster)
- Use `top_k=3` instead of 5 for less data
- Single model faster than compare mode

### Frontend  
- Resize large images before upload
- Use WebP format for smaller size
- Enable browser caching

## 🧪 Testing Checklist

- [ ] Backend starts without errors
- [ ] All 4 models load successfully
- [ ] `/health` returns healthy status
- [ ] `/models` lists all 4 models
- [ ] Single prediction works with each model
- [ ] Compare mode works
- [ ] YOLO detection works
- [ ] Breed info retrieval works
- [ ] Frontend connects to backend
- [ ] Image upload works
- [ ] Model selector works
- [ ] Compare mode toggle works

## 📚 Documentation

- **Main README**: `MULTI_MODEL_GUIDE.md`
- **Frontend README**: `frontend/README.md`
- **API Docs**: http://localhost:8000/docs (Swagger)
- **ReDoc**: http://localhost:8000/redoc

## 🎓 Learning Resources

### FastAPI
- https://fastapi.tiangolo.com/

### React + Vite
- https://react.dev/
- https://vitejs.dev/

### PyTorch
- https://pytorch.org/docs/

### YOLO
- https://docs.ultralytics.com/

## 💡 Tips & Tricks

### Quick Model Comparison
```python
import requests

with open('cat.jpg', 'rb') as f:
    r = requests.post('http://localhost:8000/predict-all', 
                      files={'file': f})
    
for model, pred in r.json()['predictions'].items():
    print(f"{model}: {pred['top_breed']} ({pred['top_confidence']*100:.1f}%)")
```

### Batch Processing
```python
import os
import requests

for img in os.listdir('cats/'):
    with open(f'cats/{img}', 'rb') as f:
        r = requests.post('http://localhost:8000/predict?model=resnet50',
                         files={'file': f})
        print(f"{img}: {r.json()['top_breed']}")
```

### Health Monitoring
```bash
# Check every 5 seconds
while true; do
  curl -s http://localhost:8000/health | jq '.status'
  sleep 5
done
```

## 🎯 Production Deployment

### Backend
```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn backend.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
# Build for production
cd frontend
npm run build

# Serve dist/ with nginx, apache, or any static server
```

### Docker (Optional)
```dockerfile
# Dockerfile for backend
FROM python:3.9-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

**Need help? Check the main documentation or open an issue!** 🚀
