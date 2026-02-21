"""
Inference and prediction utilities
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from scipy.stats import entropy
from typing import Tuple, List, Dict, Optional


class ImagePreprocessor:
    """Handle image preprocessing for model input"""
    
    @staticmethod
    def preprocess_for_resnet(image: Image.Image) -> torch.Tensor:
        """Preprocess image for ResNet-50"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Convert RGBA to RGB if necessary
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        return transform(image).unsqueeze(0)


class EntropyAnalyzer:
    """Analyze prediction entropy for wild cat detection"""
    
    @staticmethod
    def calculate_entropy(probabilities: torch.Tensor) -> float:
        """Calculate Shannon entropy of probability distribution"""
        probs = probabilities.cpu().numpy()
        probs = probs / probs.sum()
        probs = probs + 1e-10
        return entropy(probs, base=2)
    
    @staticmethod
    def detect_wild_cat(
        probabilities: torch.Tensor,
        entropy_value: float,
        top_predictions: torch.Tensor,
        threshold: float = 2.5
    ) -> Tuple[bool, Optional[str], float]:
        """Detect if image might be a wild cat based on entropy"""
        is_high_entropy = entropy_value > threshold
        max_confidence = top_predictions[0].item() * 100
        is_low_confidence = max_confidence < 40
        
        is_wild = is_high_entropy and is_low_confidence
        
        warning_msg = None
        if is_wild:
            warning_msg = (
                f"⚠️ **Vahşi Kedi Uyarısı**: Bu görsel bir ev kedisi değil, "
                f"vahşi bir kedi türü (aslan, kaplan, vaşak, leopar vb.) olabilir. "
                f"Sistem sadece ev kedisi cinsleri için eğitilmiştir.\n\n"
                f"📊 Entropi: {entropy_value:.3f} (Eşik: {threshold})\n"
                f"📉 Maksimum Güven: %{max_confidence:.2f}"
            )
        elif is_high_entropy:
            warning_msg = (
                f"⚠️ **Yüksek Belirsizlik**: Model bu görselde kararsız. "
                f"Bu bir vahşi kedi, kedi olmayan bir hayvan veya belirsiz bir görsel olabilir.\n\n"
                f"📊 Entropi: {entropy_value:.3f} (Eşik: {threshold})"
            )
        
        return is_wild, warning_msg, entropy_value


class CatDetector:
    """YOLO-based cat detection"""
    
    @staticmethod
    def detect_cat(
        image: Image.Image,
        yolo_model: Optional[object],
        confidence_threshold: float = 0.15
    ) -> Tuple[bool, float, str]:
        """Detect if image contains a cat using YOLO"""
        if yolo_model is None:
            return True, 1.0, "YOLO not available - skipping detection"
        
        try:
            results = yolo_model(image, verbose=False)
            
            if len(results) == 0:
                return False, 0.0, "No objects detected"
            
            detected_objects = []
            cat_found = False
            max_cat_conf = 0.0
            
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue
                
                names = yolo_model.names if hasattr(yolo_model, 'names') else {}
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = names.get(cls, f"class_{cls}")
                    detected_objects.append((cls, class_name, conf))
                    
                    # Class 15 = cat in COCO
                    if cls == 15 and conf > confidence_threshold:
                        cat_found = True
                        max_cat_conf = max(max_cat_conf, conf)
            
            if cat_found:
                return True, max_cat_conf, f"Cat detected (class 15, conf {max_cat_conf:.2f})"
            
            if len(detected_objects) > 0:
                debug_info = ", ".join([f"{name}({cls}):{conf:.2f}" for cls, name, conf in detected_objects[:3]])
                debug_msg = f"No cat found. Detected: {debug_info}"
            else:
                debug_msg = "No objects detected"
                
            return False, 0.0, debug_msg
            
        except Exception as e:
            return True, 1.0, f"Detection error (proceeding anyway): {str(e)[:100]}"


class BreedPredictor:
    """Cat breed prediction with entropy analysis"""
    
    def __init__(self, model: torch.nn.Module, class_names: List[str], device: torch.device):
        self.model = model
        self.class_names = class_names
        self.device = device
        self.preprocessor = ImagePreprocessor()
        self.entropy_analyzer = EntropyAnalyzer()
    
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        entropy_threshold: float = 2.5
    ) -> Dict:
        """Predict cat breed with entropy-based wild cat detection"""
        try:
            # Preprocess
            image_tensor = self.preprocessor.preprocess_for_resnet(image)
            image_tensor = image_tensor.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, top_k)
            
            # Calculate entropy
            entropy_value = self.entropy_analyzer.calculate_entropy(probabilities[0])
            
            # Detect wild cat
            is_wild, wild_warning, _ = self.entropy_analyzer.detect_wild_cat(
                probabilities[0],
                entropy_value,
                top_probs[0],
                entropy_threshold
            )
            
            # Format results
            predictions = []
            for prob, idx in zip(top_probs[0], top_indices[0]):
                predictions.append({
                    'breed': self.class_names[idx.item()],
                    'confidence': prob.item() * 100
                })
            
            return {
                'success': True,
                'predictions': predictions,
                'entropy': entropy_value,
                'is_wild_cat': is_wild,
                'wild_warning': wild_warning,
                'top_breed': predictions[0]['breed'],
                'top_confidence': predictions[0]['confidence']
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
