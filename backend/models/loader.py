"""
Model loading and management utilities
"""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import json
from typing import Tuple, List, Optional, Dict

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ModelLoader:
    """Handles loading and caching of ML models"""
    
    _models = {}  # Cache for all models
    _yolo_model = None
    _class_names = None
    _breed_info = None
    
    @classmethod
    def load_resnet50(cls, model_path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str], torch.device]:
        """Load ResNet-50 model"""
        cache_key = f"resnet50_{device}"
        if cache_key in cls._models:
            return cls._models[cache_key], cls._class_names, torch.device(device)
        
        try:
            device = torch.device(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            num_classes = len(checkpoint['class_names'])
            cls._class_names = checkpoint['class_names']
            
            # Create model
            model = models.resnet50(pretrained=False)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_ftrs, num_classes)
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            cls._models[cache_key] = model
            return model, cls._class_names, device
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ResNet-50 model: {e}")
    
    @classmethod
    def load_efficientnet(cls, model_path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str], torch.device]:
        """Load EfficientNetB0 model"""
        cache_key = f"efficientnet_{device}"
        if cache_key in cls._models:
            return cls._models[cache_key], cls._class_names, torch.device(device)
        
        try:
            device = torch.device(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            num_classes = len(checkpoint['class_names'])
            if cls._class_names is None:
                cls._class_names = checkpoint['class_names']
            
            # Create model
            model = models.efficientnet_b0(pretrained=False)
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            cls._models[cache_key] = model
            return model, cls._class_names, device
            
        except Exception as e:
            raise RuntimeError(f"Failed to load EfficientNet model: {e}")
    
    @classmethod
    def load_mobilenet(cls, model_path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str], torch.device]:
        """Load MobileNetV3 model"""
        cache_key = f"mobilenet_{device}"
        if cache_key in cls._models:
            return cls._models[cache_key], cls._class_names, torch.device(device)
        
        try:
            device = torch.device(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            num_classes = len(checkpoint['class_names'])
            if cls._class_names is None:
                cls._class_names = checkpoint['class_names']
            
            # Create model
            model = models.mobilenet_v3_small(pretrained=False)
            num_ftrs = model.classifier[3].in_features
            model.classifier[3] = nn.Linear(num_ftrs, num_classes)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            
            cls._models[cache_key] = model
            return model, cls._class_names, device
            
        except Exception as e:
            raise RuntimeError(f"Failed to load MobileNet model: {e}")
    
    @classmethod
    def load_ensemble(cls, model_path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str], torch.device]:
        """Load optimal 3-model ensemble with meta-learner"""
        cache_key = f"ensemble_{device}"
        if cache_key in cls._models:
            return cls._models[cache_key], cls._class_names, torch.device(device)
        
        try:
            from .ensemble_model import OptimalEnsembleModel, SimpleEnsembleModel
            
            device = torch.device(device)
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            
            # Extract class names - if not in checkpoint, load from breed info
            if cls._class_names is None:
                cls._class_names = checkpoint.get('class_names', checkpoint.get('classes', None))
                
                # If still None, load from cat_breed_info.json
                if cls._class_names is None:
                    breed_info_path = Path(__file__).parent.parent.parent / 'cat_breed_info.json'
                    if breed_info_path.exists():
                        import json
                        with open(breed_info_path, 'r', encoding='utf-8') as f:
                            breed_info = json.load(f)
                            if isinstance(breed_info, dict):
                                cls._class_names = sorted(breed_info.keys())
                                print(f"✓ Loaded {len(cls._class_names)} class names from cat_breed_info.json")
                            else:
                                print("⚠ Invalid breed info format")
                    else:
                        print(f"⚠ cat_breed_info.json not found at {breed_info_path}")
            
            # Get state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Fix state dict keys if they have .model. prefix (e.g., resnet50.model.conv1.weight -> resnet50.conv1.weight)
            fixed_state_dict = {}
            for key, value in state_dict.items():
                # Remove .model. prefix from keys
                if '.model.' in key:
                    new_key = key.replace('.model.', '.')
                    fixed_state_dict[new_key] = value
                else:
                    fixed_state_dict[key] = value
            
            # Target num_classes from class_names
            num_classes = len(cls._class_names) if cls._class_names else 59
            print(f"✓ Target num_classes: {num_classes}")
            
            # Check for size mismatches and filter incompatible layers
            incompatible_keys = []
            for key in list(fixed_state_dict.keys()):
                tensor = fixed_state_dict[key]
                # Check classifier layers that might have wrong num_classes
                if 'classifier' in key or 'fc.weight' in key or 'fc.bias' in key:
                    if hasattr(tensor, 'shape') and len(tensor.shape) > 0:
                        # Check if first dimension (output classes) mismatches
                        if tensor.shape[0] != num_classes and 'weight' in key:
                            print(f"⚠ Skipping incompatible layer: {key} (shape: {tensor.shape}, expected: {num_classes})")
                            incompatible_keys.append(key)
                        elif tensor.shape[0] != num_classes and 'bias' in key:
                            print(f"⚠ Skipping incompatible layer: {key} (shape: {tensor.shape}, expected: {num_classes})")
                            incompatible_keys.append(key)
            
            # Remove incompatible layers from state_dict
            for key in incompatible_keys:
                del fixed_state_dict[key]
            
            print(f"✓ Filtered out {len(incompatible_keys)} incompatible layers")
            
            # Determine ensemble architecture from checkpoint keys
            checkpoint_keys = fixed_state_dict.keys()
            
            if 'meta_learner.0.weight' in checkpoint_keys or any('meta_learner' in str(k) for k in checkpoint_keys):
                # Full OptimalEnsembleModel with meta-learner
                model = OptimalEnsembleModel(num_classes=num_classes)
                # Use strict=False to allow partial loading
                missing_keys, unexpected_keys = model.load_state_dict(fixed_state_dict, strict=False)
                if missing_keys:
                    print(f"⚠ Missing keys: {len(missing_keys)} (expected for partially trained models)")
                if unexpected_keys:
                    print(f"⚠ Unexpected keys: {len(unexpected_keys)}")
                print(f"✓ Loaded OptimalEnsembleModel with meta-learner ({num_classes} classes)")
            elif 'resnet50.fc.weight' in checkpoint_keys or any('resnet50' in str(k) for k in checkpoint_keys):
                # SimpleEnsembleModel (3 models without meta-learner)
                model = SimpleEnsembleModel(num_classes=num_classes)
                model.load_state_dict(fixed_state_dict, strict=False)
                print(f"✓ Loaded SimpleEnsembleModel ({num_classes} classes)")
            else:
                # Fallback: Create OptimalEnsembleModel and try loading
                print("⚠ Unknown checkpoint format, attempting OptimalEnsembleModel...")
                model = OptimalEnsembleModel(num_classes=num_classes)
                try:
                    model.load_state_dict(fixed_state_dict, strict=False)
                except Exception as load_err:
                    print(f"⚠ Partial load: {load_err}")
            
            model = model.to(device)
            model.eval()
            
            cls._models[cache_key] = model
            return model, cls._class_names, device
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Failed to load Ensemble model: {e}")
    
    @classmethod
    def load_model(cls, model_name: str, model_path: Path, device: str = "cpu") -> Tuple[nn.Module, List[str], torch.device]:
        """Load any model by name"""
        loaders = {
            "resnet50": cls.load_resnet50,
            "efficientnet": cls.load_efficientnet,
            "mobilenet": cls.load_mobilenet,
            "ensemble": cls.load_ensemble
        }
        
        if model_name not in loaders:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(loaders.keys())}")
        
        return loaders[model_name](model_path, device)
    
    @classmethod
    def load_yolo(cls, model_path: Path) -> Optional[object]:
        """Load YOLO model for cat detection"""
        if cls._yolo_model is not None:
            return cls._yolo_model
        
        if not YOLO_AVAILABLE:
            return None
        
        try:
            cls._yolo_model = YOLO(str(model_path))
            return cls._yolo_model
        except Exception:
            return None
    
    @classmethod
    def load_breed_info(cls, info_path: Path) -> dict:
        """Load breed information from JSON"""
        if cls._breed_info is not None:
            return cls._breed_info
        
        try:
            with open(info_path, 'r', encoding='utf-8') as f:
                cls._breed_info = json.load(f)
            return cls._breed_info
        except Exception as e:
            print(f"Warning: Failed to load breed info: {e}")
            return {}
    
    @classmethod
    def get_class_names(cls) -> Optional[List[str]]:
        """Get class names"""
        return cls._class_names
