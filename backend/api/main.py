"""
FastAPI Backend for Cat Breed Classification
2-Stage Pipeline: YOLO11n Detection → Optimal Ensemble Classification
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io
import json
import torch
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="Cat Breed Classification API",
    description="AI-powered cat breed recognition with 2-stage pipeline (YOLO + Ensemble)",
    version="3.0.0"
)

# API router — tüm endpoint'ler /api prefix ile de erişilebilir
api_router = APIRouter(prefix="/api")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
pipeline = None
breed_info = None
class_names = None

# Model paths
YOLO_MODEL_PATH = Path("/app/yolo11n.pt")
RESNET50_MODEL_PATH = Path("/app/resnet50_best.pth")
ENSEMBLE_MODEL_PATH = Path("/app/optimal_ensemble_final.pth")
BREED_INFO_PATH = Path("/app/cat_breed_info.json")


@app.on_event("startup")
async def startup_event():
    """Load 2-stage pipeline on startup"""
    global pipeline, breed_info, class_names
    
    print("="*50)
    print("Loading 2-Stage Pipeline...")
    print("="*50)
    
    # Import here to avoid circular imports
    from backend.models.pipeline import TwoStagePipeline
    from backend.models.loader import ModelLoader
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Try ResNet50 first (best standalone model), fallback to ensemble
    classification_model = None
    print("\n[Stage 2] Loading Classification Model...")
    
    if RESNET50_MODEL_PATH.exists():
        try:
            print("  Trying ResNet50 (best standalone model)...")
            classification_model, loaded_class_names, device = ModelLoader.load_resnet50(
                RESNET50_MODEL_PATH,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            class_names = loaded_class_names
            print(f"✓ ResNet50 loaded: {len(class_names)} breeds on {device}")
        except Exception as e:
            print(f"⚠ ResNet50 failed: {e}, falling back to ensemble...")
            classification_model = None
    
    if classification_model is None:
        try:
            print("  Loading Optimal Ensemble Model...")
            classification_model, loaded_class_names, device = ModelLoader.load_ensemble(
                ENSEMBLE_MODEL_PATH,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            class_names = loaded_class_names
            print(f"✓ Ensemble loaded: {len(class_names)} breeds on {device}")
        except Exception as e:
            print(f"✗ Failed to load any classification model: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Classification model loading failed: {e}")
    
    ensemble_model = classification_model
    
    # Initialize 2-stage pipeline
    print(f"\n[Stage 1] Initializing YOLO11n Detection...")
    try:
        pipeline = TwoStagePipeline(
            yolo_model_path=str(YOLO_MODEL_PATH),
            ensemble_model=ensemble_model,
            class_names=class_names,
            device=device,
            yolo_confidence=0.25,
            entropy_threshold=2.5
        )
        print("✓ YOLO11n loaded")
    except Exception as e:
        print(f"✗ YOLO loading failed: {e}")
        raise RuntimeError(f"YOLO model loading failed: {e}")
    
    # Load breed info
    print("\n[Additional] Loading breed information...")
    try:
        with open(BREED_INFO_PATH, 'r', encoding='utf-8') as f:
            breed_info = json.load(f)
        print(f"✓ Loaded info for {len(breed_info)} breeds")
    except Exception as e:
        print(f"⚠ Breed info not loaded: {e}")
        breed_info = {}
    
    print("\n" + "="*50)
    print("✓ 2-Stage Pipeline Ready!")
    print("="*50)


@api_router.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Cat Breed Classification API",
        "version": "3.0.0",
        "status": "running",
        "pipeline": {
            "stage1": "YOLO11n Detection (2.6M params)",
            "stage2": "Optimal Ensemble (ResNet50 + EfficientNet + MobileNet)",
            "accuracy": "63.85%",
            "breeds": len(class_names) if class_names else 0
        }
    }


@api_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_loaded": pipeline is not None,
        "breeds_loaded": len(class_names) if class_names else 0,
        "breed_info_loaded": breed_info is not None and len(breed_info) > 0
    }


@api_router.get("/info")
async def get_info():
    """Get API information"""
    return {
        "api": "Cat Breed Classification",
        "version": "3.0.0",
        "pipeline": {
            "stage1": {
                "model": "YOLO11n",
                "params": "2.6M",
                "purpose": "Cat detection and localization"
            },
            "stage2": {
                "model": "Optimal Ensemble",
                "architecture": "ResNet50 + EfficientNet-B0 + MobileNet-V3",
                "meta_learner": "LogisticRegression",
                "purpose": "Breed classification"
            }
        },
        "performance": {
            "overall_accuracy": "63.85%",
            "breeds_supported": len(class_names) if class_names else 0
        },
        "endpoints": {
            "/": "API root",
            "/health": "Health check",
            "/info": "API information",
            "/breeds": "List all breeds",
            "/breeds/{breed_name}": "Get breed details",
            "/predict": "Predict breed from image"
        }
    }


@api_router.get("/breeds")
async def list_breeds():
    """List all available cat breeds"""
    if class_names is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    return {
        "total": len(class_names),
        "breeds": class_names
    }


@api_router.get("/breeds/{breed_name}")
async def get_breed_info_endpoint(breed_name: str):
    """Get detailed information about a specific breed"""
    if breed_info is None:
        raise HTTPException(status_code=503, detail="Breed information not loaded")
    
    if breed_name not in breed_info:
        raise HTTPException(status_code=404, detail=f"Breed '{breed_name}' not found")
    
    return {
        "breed": breed_name,
        "info": breed_info[breed_name]
    }


@api_router.post("/predict")
async def predict_breed(
    file: UploadFile = File(...),
    skip_detection: bool = False,
    top_k: int = 5,
    return_base_predictions: bool = False
):
    """
    Predict cat breed from uploaded image using 2-stage pipeline
    
    Parameters:
    - file: Image file (JPG, JPEG, PNG)
    - skip_detection: Skip YOLO cat detection (default: False)
    - top_k: Number of top predictions to return (default: 5)
    - return_base_predictions: Include individual model predictions (default: False)
    
    Returns:
    - Complete pipeline result with breed predictions
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded")
    
    # Validate file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(status_code=400, detail=f"Image too large (max 10MB)")
        
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Run 2-stage pipeline
        result = pipeline.process(
            image=image,
            skip_detection=skip_detection,
            top_k=top_k,
            return_base_predictions=return_base_predictions
        )
        
        # Add breed info to top predictions
        if result.get('success') and 'predictions' in result:
            for pred in result['predictions']:
                breed = pred['breed']
                if breed in breed_info:
                    pred['info'] = breed_info[breed]
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ── API Router'ı kaydet ─────────────────────────────────────────────────────
app.include_router(api_router)

# ── React Frontend Static Dosya Servisi ──────────────────────────────────────
# Docker build'de React dist/ klasörü /app/frontend-react/dist'e kopyalanır.
# api_router zaten /api/* path'lerini alıyor; catch-all sadece geri kalanı yakalar.
FRONTEND_DIST = Path("/app/frontend-react/dist")

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str):
    """React Router desteği: bilinmeyen path'lerde index.html döndür"""
    if FRONTEND_DIST.exists():
        static_file = FRONTEND_DIST / full_path
        if static_file.exists() and static_file.is_file():
            return FileResponse(str(static_file))
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
    # Frontend build yok — API bilgisi göster
    return JSONResponse({
        "message": "Cat Breed Classification API",
        "version": "3.0.0",
        "status": "running",
        "note": "Frontend build bulunamadı. /api/ prefix ile API'ye erişin."
    })

@app.get("/", include_in_schema=False)
async def serve_root():
    """Root path → React frontend index.html"""
    if FRONTEND_DIST.exists():
        index = FRONTEND_DIST / "index.html"
        if index.exists():
            return FileResponse(str(index))
    return JSONResponse({
        "message": "Cat Breed Classification API",
        "version": "3.0.0",
        "status": "running",
        "api_docs": "/docs"
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
