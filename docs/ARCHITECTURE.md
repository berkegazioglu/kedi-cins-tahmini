# 🏗️ System Architecture

## 📊 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER BROWSER                            │
│                    http://localhost:5173                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             │ HTTP/AJAX
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                       REACT FRONTEND                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ ImageUploader│  │PredictionView│  │ ModelSelector│          │
│  │              │  │              │  │              │          │
│  │ - Drag&Drop  │  │ - Results    │  │ - Dropdown   │          │
│  │ - Preview    │  │ - Compare    │  │ - Compare    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│                    ┌──────────────┐                             │
│                    │  API Service │                             │
│                    │              │                             │
│                    │ - api.js     │                             │
│                    └──────┬───────┘                             │
└───────────────────────────┼─────────────────────────────────────┘
                            │ REST API
                            │ JSON
┌───────────────────────────▼─────────────────────────────────────┐
│                    FASTAPI BACKEND                              │
│                 http://localhost:8000                           │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    API ENDPOINTS                        │   │
│  │                                                          │   │
│  │  GET  /              - API Info                         │   │
│  │  GET  /health        - Health Check                     │   │
│  │  GET  /models        - List Models                      │   │
│  │  GET  /breeds        - List Breeds                      │   │
│  │  GET  /breeds/{name} - Breed Info                       │   │
│  │  POST /predict       - Single Model Prediction          │   │
│  │  POST /predict-all   - Multi-Model Comparison           │   │
│  │  POST /detect        - YOLO Cat Detection               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────────┐     │
│  │               BUSINESS LOGIC LAYER                     │     │
│  │                                                         │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │     │
│  │  │   Model      │  │   Inference  │  │   Cat       │ │     │
│  │  │   Loader     │  │   Utils      │  │   Detector  │ │     │
│  │  │              │  │              │  │             │ │     │
│  │  │ - Load Model │  │ - Preprocess │  │ - YOLO      │ │     │
│  │  │ - Cache      │  │ - Predict    │  │ - Threshold │ │     │
│  │  │ - Validate   │  │ - Entropy    │  │ - Validate  │ │     │
│  │  └──────────────┘  └──────────────┘  └─────────────┘ │     │
│  └───────────────────────────────────────────────────────┘     │
│                            │                                     │
│  ┌─────────────────────────▼─────────────────────────────┐     │
│  │                MODEL CACHE LAYER                       │     │
│  │                                                         │     │
│  │  models = {                                            │     │
│  │    'resnet50': <model>,                                │     │
│  │    'efficientnet': <model>,                            │     │
│  │    'mobilenet': <model>,                               │     │
│  │    'ensemble': <model>                                 │     │
│  │  }                                                      │     │
│  │                                                         │     │
│  │  breed_predictors = {                                  │     │
│  │    'resnet50': <predictor>,                            │     │
│  │    'efficientnet': <predictor>,                        │     │
│  │    'mobilenet': <predictor>,                           │     │
│  │    'ensemble': <predictor>                             │     │
│  │  }                                                      │     │
│  └───────────────────────────────────────────────────────┘     │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               │ PyTorch Load
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                        MODEL FILES                              │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  best_cat_breed_model_resnet50_v2.pth  (ResNet-50)     │   │
│  │  - 25.6M parameters                                      │   │
│  │  - 64.67% accuracy                                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  EfficientNetB0_best.pth  (EfficientNetB0)             │   │
│  │  - 5.3M parameters                                       │   │
│  │  - Efficient architecture                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  MobileNetV3_best.pth  (MobileNetV3)                   │   │
│  │  - 4.2M parameters                                       │   │
│  │  - Mobile optimized                                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  optimal_ensemble_final.pth  (Ensemble)                │   │
│  │  - Combined model                                        │   │
│  │  - Best accuracy                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  yolo11n.pt  (YOLO11n)                                 │   │
│  │  - Cat detection                                         │   │
│  │  - 80 classes                                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
```

## 🔄 Request Flow

### Single Model Prediction

```
USER → Frontend → API Service → POST /predict?model=resnet50
                                        ↓
                                 Validate Model
                                        ↓
                            Get breed_predictors['resnet50']
                                        ↓
                         ┌──────────────┴──────────────┐
                         ↓                             ↓
                    YOLO Detection              Preprocess Image
                  (if not skipped)                    ↓
                         ↓                      ResNet-50 Forward
                    Cat Found?                         ↓
                         ↓                      Softmax + Top-K
                    ┌────┴────┐                        ↓
                 YES         NO                 Entropy Analysis
                    ↓          ↓                        ↓
               Continue    Return Error          Wild Cat Check
                    ↓                                   ↓
                    └────────────┬──────────────────────┘
                                 ↓
                          Get Breed Info
                                 ↓
                         Add model_used field
                                 ↓
                           Return JSON
                                 ↓
                           API Service
                                 ↓
                          Update UI State
                                 ↓
                       Display Results
```

### Multi-Model Comparison

```
USER → Frontend → API Service → POST /predict-all
                                        ↓
                                 YOLO Detection
                                   (optional)
                                        ↓
                                  Cat Found?
                                        ↓
                      ┌─────────────────┼─────────────────┐
                      ↓                 ↓                 ↓
              Model: resnet50   Model: efficientnet   Model: mobilenet
                      ↓                 ↓                 ↓
              Start Timer       Start Timer         Start Timer
                      ↓                 ↓                 ↓
              Preprocess        Preprocess          Preprocess
                      ↓                 ↓                 ↓
              Forward Pass      Forward Pass        Forward Pass
                      ↓                 ↓                 ↓
              Top-K + Entropy   Top-K + Entropy     Top-K + Entropy
                      ↓                 ↓                 ↓
              Stop Timer        Stop Timer          Stop Timer
                      ↓                 ↓                 ↓
              Store Result      Store Result        Store Result
                      └─────────────────┴─────────────────┘
                                        ↓
                      ┌─────────────────┼─────────────────┐
                      ↓                                   ↓
              Model: ensemble                    Calculate Consensus
                      ↓                              (most common)
              [Same process]                              ↓
                      ↓                                   ↓
              Store Result                         Aggregate Timing
                      └───────────────┬─────────────────────┘
                                      ↓
                              Return All Results
                                      ↓
                                 API Service
                                      ↓
                              Update UI State
                                      ↓
                        Display Comparison View
```

## 📦 Component Dependencies

### Backend Dependencies

```
FastAPI App (main.py)
    │
    ├── Config (settings.py)
    │   ├── AVAILABLE_MODELS
    │   ├── DEFAULT_MODEL
    │   ├── ENTROPY_THRESHOLD
    │   ├── YOLO_CONFIDENCE_THRESHOLD
    │   └── API_SETTINGS
    │
    ├── Model Loader (loader.py)
    │   ├── load_resnet50()
    │   ├── load_efficientnet()
    │   ├── load_mobilenet()
    │   ├── load_ensemble()
    │   └── load_model()  [Generic]
    │
    └── Inference Utils (inference.py)
        ├── ImagePreprocessor
        │   └── preprocess_for_resnet()
        │
        ├── EntropyAnalyzer
        │   ├── calculate_entropy()
        │   └── detect_wild_cat()
        │
        ├── CatDetector
        │   └── detect_cat()
        │
        └── BreedPredictor
            └── predict()
```

### Frontend Dependencies

```
App.jsx (Main Component)
    │
    ├── State Management
    │   ├── selectedFile
    │   ├── results
    │   ├── isLoading
    │   ├── error
    │   ├── skipDetection
    │   ├── selectedModel
    │   ├── compareMode
    │   └── availableModels
    │
    ├── API Service (api.js)
    │   ├── getModels()
    │   ├── predictBreed()
    │   ├── predictAllModels()
    │   ├── detectCat()
    │   ├── getBreeds()
    │   ├── getBreedInfo()
    │   └── healthCheck()
    │
    ├── ImageUploader Component
    │   ├── Props: onImageSelect, onPredict, isLoading
    │   ├── Features: Drag&Drop, Preview, File Select
    │   └── CSS: ImageUploader.css
    │
    └── PredictionResults Component
        ├── Props: results, compareMode
        ├── Modes: Single, Compare
        ├── Features: Tabs, Consensus, Timing
        └── CSS: PredictionResults.css
```

## 🗄️ Data Flow

### Model Loading (Startup)

```
Backend Startup
      ↓
Read AVAILABLE_MODELS from settings.py
      ↓
For each model in AVAILABLE_MODELS:
      ↓
  Check if .pth file exists
      ↓
      YES → Load model architecture
      ↓
      Load checkpoint weights
      ↓
      Set to eval mode
      ↓
      Cache in models dict
      ↓
      Create BreedPredictor
      ↓
      Cache in breed_predictors dict
      ↓
      Log success
      ↓
      NO → Log warning, skip model
      ↓
All models loaded
      ↓
API Ready to serve requests
```

### Prediction Flow (Single Model)

```
User uploads image
      ↓
Frontend: Create FormData
      ↓
Frontend: Add file + parameters (model, skip_detection, top_k)
      ↓
Frontend: POST to /predict
      ↓
Backend: Receive request
      ↓
Backend: Validate file type
      ↓
Backend: Validate model exists
      ↓
Backend: Get breed_predictor for selected model
      ↓
Backend: YOLO detection (if not skipped)
      ↓
      Cat found? → YES
      ↓
Backend: Preprocess image (resize, normalize)
      ↓
Backend: Model forward pass
      ↓
Backend: Apply softmax
      ↓
Backend: Get top-K predictions
      ↓
Backend: Calculate entropy
      ↓
Backend: Check if wild cat (entropy > threshold)
      ↓
Backend: Retrieve breed info from JSON
      ↓
Backend: Build response JSON
      ↓
Backend: Add model_used field
      ↓
Backend: Return response
      ↓
Frontend: Parse JSON
      ↓
Frontend: Update results state
      ↓
Frontend: Render PredictionResults
      ↓
User sees results
```

## 🎨 UI Component Tree

```
App
 ├─ Header
 │   ├─ Title: "🐱 Kedi Cinsi Tahmin Sistemi"
 │   └─ Subtitle
 │
 ├─ Controls Panel
 │   ├─ Skip Detection Checkbox
 │   ├─ Compare Mode Checkbox
 │   └─ Model Selector (if not compare mode)
 │       └─ Dropdown with available models
 │
 ├─ ImageUploader
 │   ├─ Upload Area (drag & drop)
 │   ├─ File Input (hidden)
 │   ├─ Preview Image (if uploaded)
 │   └─ Predict Button
 │
 ├─ Error Message (if error)
 │
 ├─ PredictionResults (if results)
 │   │
 │   ├─ Single Model Mode
 │   │   ├─ Wild Cat Warning (if applicable)
 │   │   ├─ Model Badge
 │   │   ├─ Top Prediction Card
 │   │   ├─ Other Predictions List
 │   │   └─ Breed Info Tabs
 │   │       ├─ General
 │   │       ├─ Health
 │   │       ├─ Nutrition
 │   │       ├─ Care
 │   │       └─ Character
 │   │
 │   └─ Compare Mode
 │       ├─ Consensus Section
 │       ├─ Models Comparison Grid
 │       │   ├─ Model Card 1 (resnet50)
 │       │   ├─ Model Card 2 (efficientnet)
 │       │   ├─ Model Card 3 (mobilenet)
 │       │   └─ Model Card 4 (ensemble)
 │       └─ Timing Summary
 │
 └─ Footer
     ├─ Technology Stack Info
     └─ Features Info
```

## 🔐 Security & Validation

### Backend Validation

```
Request Received
      ↓
┌─────────────────────────────────┐
│ File Validation                 │
│ - Check content_type            │
│ - Max size: 10MB                │
│ - Valid image format            │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ Parameter Validation            │
│ - model in AVAILABLE_MODELS     │
│ - skip_detection is boolean     │
│ - top_k is integer (1-59)       │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ Model Availability Check        │
│ - Model loaded?                 │
│ - Predictor exists?             │
└─────────────────────────────────┘
      ↓
┌─────────────────────────────────┐
│ Image Processing Safety         │
│ - Try/except PIL operations     │
│ - Handle corrupted images       │
│ - Validate dimensions           │
└─────────────────────────────────┘
      ↓
Process Request
```

## 📈 Performance Optimization

### Backend Caching Strategy

```
Application Start
      ↓
Load all models once
      ↓
Cache in memory
      ↓
Request comes in
      ↓
Use cached model (no reload)
      ↓
Fast inference
```

### Frontend Optimization

```
Initial Load
      ↓
Fetch available models once
      ↓
Cache in state
      ↓
User uploads image
      ↓
Show loading state immediately
      ↓
Send API request
      ↓
Update UI only when response received
      ↓
Re-render only changed components
```

## 🌐 Network Communication

### API Contract

```
Request Format:
POST /predict
Content-Type: multipart/form-data
Parameters:
  - file: <binary image data>
  - model: string (default: "resnet50")
  - skip_detection: boolean (default: false)
  - top_k: integer (default: 5)

Response Format:
Content-Type: application/json
{
  "success": boolean,
  "top_breed": string,
  "top_confidence": float,
  "predictions": array,
  "is_wild_cat": boolean,
  "entropy": float,
  "model_used": string,
  "detection": object,
  "breed_info": object
}
```

---

**This architecture supports:**
- ✅ Scalability (add more models easily)
- ✅ Maintainability (clear separation of concerns)
- ✅ Performance (model caching, async operations)
- ✅ Flexibility (model selection, compare mode)
- ✅ Reliability (error handling at each layer)
