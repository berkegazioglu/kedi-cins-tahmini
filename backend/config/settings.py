"""
Backend Configuration Settings
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
MODELS_DIR = BASE_DIR / "runs"
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"

# Model paths
RESNET50_MODEL_PATH = MODELS_DIR / "resnet50_v2" / "weights" / "best.pth"
EFFICIENTNET_MODEL_PATH = BASE_DIR / "EfficientNetB0_best.pth"
MOBILENET_MODEL_PATH = BASE_DIR / "MobileNetV3_best.pth"
ENSEMBLE_MODEL_PATH = BASE_DIR / "optimal_ensemble_final.pth"
YOLO_MODEL_PATH = BASE_DIR / "yolo11n.pt"
BREED_INFO_PATH = BASE_DIR / "cat_breed_info.json"
CLASS_NAMES_PATH = MODELS_DIR / "resnet50_v2" / "class_names.json"

# Available models
AVAILABLE_MODELS = {
    "resnet50": RESNET50_MODEL_PATH,
    "efficientnet": EFFICIENTNET_MODEL_PATH,
    "mobilenet": MOBILENET_MODEL_PATH,
    "ensemble": ENSEMBLE_MODEL_PATH
}

DEFAULT_MODEL = "resnet50"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
API_WORKERS = int(os.getenv("API_WORKERS", 4))

# Model settings
DEVICE = "cuda" if os.getenv("USE_GPU", "false").lower() == "true" else "cpu"
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Detection settings
YOLO_CONFIDENCE_THRESHOLD = 0.15
ENTROPY_THRESHOLD = 2.5
TOP_K_PREDICTIONS = 5

# Wild cat species
WILD_CATS = [
    "aslan", "kaplan", "leopar", "çita", "jaguar",
    "puma", "vaşak", "çöl kedisi", "serval", "oselot",
    "kara ayak kedi", "pallas kedisi", "balıkçı kedisi",
    "lion", "tiger", "leopard", "cheetah", "jaguar",
    "puma", "lynx", "serval", "ocelot", "caracal",
    "sand cat", "fishing cat", "pallas cat", "black-footed cat"
]

# Ensure directories exist
UPLOADS_DIR.mkdir(exist_ok=True)
