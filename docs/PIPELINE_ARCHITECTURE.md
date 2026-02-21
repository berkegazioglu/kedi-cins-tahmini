# 🏗️ 2-Stage Pipeline Architecture

## 📊 Overview

Bu proje, kedi cinsi sınıflandırması için iki aşamalı bir pipeline mimarisi kullanır:

```
┌──────────────────────────────────────────────────────────────┐
│                    Input Image (any size)                    │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    Stage 1: YOLO11n Detection   │
        │         (2.6M params)            │
        │   Cat Detection? (Conf > 0.25)  │
        └────────────────┬─────────────────┘
                         │
                    ┌────┴────┐
                  NO│         │YES
                    │         │
          ┌─────────▼──┐    ┌─▼──────────────────────────────┐
          │  Reject    │    │ Stage 2: Optimal 3-Model       │
          │  Image     │    │        Ensemble                │
          └────────────┘    │                                │
                            │ ┌────────────────────────┐     │
                            │ │   Base Models (3)      │     │
                            │ │                        │     │
                            │ │ • ResNet-50 (24.6M)    │     │
                            │ │   Accuracy: 64.67%     │     │
                            │ │                        │     │
                            │ │ • EfficientNetB0 (5.3M)│     │
                            │ │   Accuracy: 60.66%     │     │
                            │ │                        │     │
                            │ │ • MobileNetV3 (5.4M)   │     │
                            │ │   Accuracy: 60.06%     │     │
                            │ └───────────┬────────────┘     │
                            │             │                  │
                            │ ┌───────────▼────────────┐     │
                            │ │  Meta-Learner (FC)     │     │
                            │ │  Combines outputs      │     │
                            │ └───────────┬────────────┘     │
                            │             │                  │
                            │     ┌───────▼────────┐         │
                            │     │ Final Prediction│         │
                            │     │   63.85% Acc    │         │
                            │     └────────────────┘         │
                            └────────────────────────────────┘
```

## 🎯 Stage 1: YOLO11n Cat Detection

### Purpose
- Görüntüde kedi olup olmadığını tespit etmek
- Yanlış sınıflandırmaları önlemek (örn: köpek, kaplumbağa vs.)
- İşlem süresini optimize etmek

### Specifications
- **Model**: YOLO11n
- **Parameters**: 2.6M
- **Pre-trained**: COCO dataset (80 classes)
- **Cat Class ID**: 15
- **Confidence Threshold**: 0.25 (configurable)
- **Input**: Any size image
- **Output**: Boolean (cat detected or not) + confidence score

### Features
- Fast inference (~50ms on CPU)
- Bounding box detection
- Multiple cats support
- Skip detection option (for trusted sources)

### Files
- `backend/models/stage1_detection.py` - Detection implementation
  - `YOLOCatDetector` - Core YOLO detector
  - `DetectionPipeline` - Complete detection pipeline

## 🧠 Stage 2: Optimal 3-Model Ensemble

### Architecture

#### Base Models (Pre-trained)

1. **ResNet-50** (24.6M parameters)
   - Accuracy: 64.67%
   - Deep residual network
   - Strong feature extraction
   - Best individual performance

2. **EfficientNetB0** (5.3M parameters)
   - Accuracy: 60.66%
   - Compound scaling
   - Efficient architecture
   - Good speed/accuracy trade-off

3. **MobileNetV3-Large** (5.4M parameters)
   - Accuracy: 60.06%
   - Mobile-optimized
   - Inverted residuals
   - Fast inference

#### Meta-Learner (Trainable)

```
Input: [ResNet50_output (59), EfficientNet_output (59), MobileNet_output (59)]
       = 177 features

Layer 1: Linear(177 → 256) + ReLU + Dropout(0.3)
Layer 2: Linear(256 → 128) + ReLU + Dropout(0.2)
Layer 3: Linear(128 → 59)

Output: Final prediction (59 classes)
```

### Training Strategies

#### 1. End-to-End Training
```python
model = OptimalEnsembleModel(num_classes=59, freeze_base_models=False)
# Train all models together
```
- Pros: Best potential accuracy
- Cons: Requires more compute, risk of overfitting

#### 2. Freeze Base Models
```python
model = OptimalEnsembleModel(num_classes=59, freeze_base_models=True)
# Only train meta-learner
```
- Pros: Fast training, uses pre-trained knowledge
- Cons: Limited adaptation to specific dataset

#### 3. Two-Stage Training (Recommended)
```python
# Stage 1: Train meta-learner only (freeze base models)
trainer = EnsembleTrainer(..., freeze_base_models=True)
trainer.train(num_epochs=10)

# Stage 2: Fine-tune everything
trainer.freeze_base_models = False
trainer.train(num_epochs=5)
```

### Performance

- **Final Accuracy**: 63.85%
- **Inference Time**: ~0.4s (CPU) / ~0.15s (GPU)
- **Memory**: ~6GB RAM (all models loaded)

### Features

- Wild cat detection (entropy analysis)
- Top-K predictions with confidence
- Base model comparison
- Batch prediction support
- Model ensemble visualization

### Files
- `backend/models/ensemble_model.py` - Ensemble architecture
  - `OptimalEnsembleModel` - Full ensemble with meta-learner
  - `SimpleEnsembleModel` - Simple averaging
  - `WeightedEnsembleModel` - Weighted combination
- `backend/models/stage2_classification.py` - Classification pipeline
  - `BreedClassificationPipeline` - Main classification logic
  - `ModelComparison` - Compare base models

## 🔄 Complete Pipeline

### Files
- `backend/models/pipeline.py` - Complete 2-stage pipeline
  - `TwoStagePipeline` - Combines both stages
  - `PipelineVisualizer` - Result visualization

### Usage Example

```python
from backend.models.pipeline import TwoStagePipeline
from backend.models.ensemble_model import OptimalEnsembleModel
from PIL import Image
import torch

# Load models
ensemble_model = OptimalEnsembleModel(num_classes=59)
ensemble_model.load_state_dict(torch.load('optimal_ensemble_final.pth'))

# Create pipeline
pipeline = TwoStagePipeline(
    yolo_model_path='yolo11n.pt',
    ensemble_model=ensemble_model,
    class_names=breed_names,
    device=torch.device('cuda'),
    yolo_confidence=0.25,
    entropy_threshold=2.5
)

# Process image
image = Image.open('cat.jpg')
result = pipeline.process(image, skip_detection=False, top_k=5)

# Result structure:
{
    'success': True,
    'stage': 'complete',
    'detection': {
        'cat_detected': True,
        'confidence': 0.95,
        'message': 'Cat detected'
    },
    'classification': {
        'top_breed': 'British Shorthair',
        'top_confidence': 0.854,
        'predictions': [...],
        'is_wild_cat': False,
        'entropy': 1.23
    },
    'timing': {
        'detection': 0.05,
        'classification': 0.15,
        'total': 0.20
    }
}
```

## 🚀 Training

### Train Ensemble Model

```bash
python train_ensemble_pipeline.py
```

Configuration in `train_ensemble_pipeline.py`:
```python
data_dir = 'images_split'  # train/ and val/ subdirectories
batch_size = 32
num_epochs = 20
learning_rate = 0.001
device = 'cuda'  # or 'cpu'
freeze_base_models = False  # Set True for faster training
```

### Expected Results

| Training Strategy | Epochs | Val Accuracy | Training Time (GPU) |
|-------------------|--------|--------------|---------------------|
| Freeze base models | 10 | ~62% | ~2 hours |
| End-to-end | 20 | ~64% | ~8 hours |
| Two-stage | 10+5 | ~63.85% | ~4 hours |

## 🧪 Testing

### Test Pipeline Architecture

```bash
python test_pipeline_architecture.py
```

This will test:
- ✅ Stage 1: YOLO detection
- ✅ Stage 2: Ensemble classification
- ✅ Complete pipeline
- ✅ Real image processing (if available)

### Test Individual Components

```bash
# Test Stage 1 only
python backend/models/stage1_detection.py

# Test Stage 2 only
python backend/models/stage2_classification.py

# Test ensemble model
python backend/models/ensemble_model.py

# Test complete pipeline
python backend/models/pipeline.py
```

## 📊 Model Comparison

### Individual vs Ensemble

| Model | Parameters | Accuracy | Inference Time |
|-------|-----------|----------|----------------|
| ResNet-50 | 24.6M | 64.67% | ~0.15s |
| EfficientNetB0 | 5.3M | 60.66% | ~0.10s |
| MobileNetV3 | 5.4M | 60.06% | ~0.08s |
| **Ensemble** | **35.3M** | **63.85%** | **~0.40s** |

### Why Ensemble?

- **Robustness**: Combines strengths of multiple models
- **Reduced variance**: Less sensitive to individual model errors
- **Better generalization**: Meta-learner learns optimal combination
- **Wild cat detection**: Entropy analysis from multiple predictions

## 📁 Project Structure

```
kedi-cins-tahmini/
├── backend/
│   └── models/
│       ├── ensemble_model.py        # Ensemble architectures
│       ├── stage1_detection.py      # YOLO detection
│       ├── stage2_classification.py # Classification pipeline
│       ├── pipeline.py              # Complete 2-stage pipeline
│       └── loader.py                # Model loading utilities
│
├── train_ensemble_pipeline.py       # Training script
├── test_pipeline_architecture.py    # Testing script
│
├── yolo11n.pt                       # YOLO weights
├── optimal_ensemble_final.pth       # Trained ensemble
│
└── images_split/                    # Dataset
    ├── train/
    └── val/
```

## 🔧 Configuration

### Detection Settings
```python
YOLO_MODEL_PATH = 'yolo11n.pt'
YOLO_CONFIDENCE_THRESHOLD = 0.25
```

### Classification Settings
```python
ENTROPY_THRESHOLD = 2.5  # Wild cat detection
NUM_CLASSES = 59
TOP_K = 5  # Number of predictions
```

### Training Settings
```python
LEARNING_RATE = 0.001
BATCH_SIZE = 32
NUM_EPOCHS = 20
OPTIMIZER = 'Adam'
SCHEDULER = 'ReduceLROnPlateau'
```

## 📈 Performance Optimization

### Memory
- Load models once at startup
- Use model caching
- Batch processing for multiple images

### Speed
- Skip detection for trusted sources
- Use GPU when available
- Optimize image preprocessing
- Use smaller models for mobile

### Accuracy
- Ensemble multiple models
- Use entropy for uncertainty detection
- Fine-tune on domain-specific data
- Data augmentation during training

## 🎯 Future Improvements

1. **Add more base models**: Increase ensemble diversity
2. **Attention mechanism**: Weight different regions of image
3. **Self-training**: Use confident predictions for semi-supervised learning
4. **Model distillation**: Create smaller student model
5. **Active learning**: Select most informative samples for labeling

## 📚 References

- YOLO: https://github.com/ultralytics/ultralytics
- ResNet: https://arxiv.org/abs/1512.03385
- EfficientNet: https://arxiv.org/abs/1905.11946
- MobileNetV3: https://arxiv.org/abs/1905.02244
- Ensemble Learning: https://en.wikipedia.org/wiki/Ensemble_learning

---

**Ready to use! 🚀** Test the pipeline with:
```bash
python test_pipeline_architecture.py
```
