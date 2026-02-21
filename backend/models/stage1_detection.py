"""
Stage 1: YOLO11n Cat Detection Pipeline
Implements the first stage of the cat breed classification system
"""
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np
from typing import Tuple, Optional


class YOLOCatDetector:
    """
    Stage 1: YOLO11n Object Detection
    Detects if image contains a cat before breed classification
    
    Architecture:
    - YOLO11n (2.6M parameters)
    - Pre-trained on COCO dataset
    - Confidence threshold: 0.25 (configurable)
    """
    
    def __init__(self, model_path: str = 'yolo11n.pt', confidence_threshold: float = 0.25):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO11n weights
            confidence_threshold: Minimum confidence for cat detection
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.cat_class_id = 15  # Cat class in COCO dataset
        
    def load_model(self):
        """Load YOLO11n model"""
        try:
            self.model = YOLO(self.model_path)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    def detect(self, image: Image.Image) -> Tuple[bool, float, str, Optional[dict]]:
        """
        Detect if image contains a cat
        
        Args:
            image: PIL Image
            
        Returns:
            cat_found: True if cat detected
            confidence: Detection confidence (0-1)
            message: Human-readable message
            detection_info: Detailed detection information
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Run YOLO detection
            results = self.model(image, verbose=False)
            
            # Check if any cat detected
            cat_detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == self.cat_class_id and conf >= self.confidence_threshold:
                        cat_detections.append({
                            'confidence': conf,
                            'bbox': box.xyxy[0].cpu().numpy().tolist(),
                            'class': 'cat'
                        })
            
            if cat_detections:
                # Get highest confidence detection
                best_detection = max(cat_detections, key=lambda x: x['confidence'])
                confidence = best_detection['confidence']
                
                return True, confidence, f"Cat detected with {confidence:.2%} confidence", {
                    'num_detections': len(cat_detections),
                    'best_detection': best_detection,
                    'all_detections': cat_detections
                }
            else:
                return False, 0.0, "No cat detected in image", None
                
        except Exception as e:
            return False, 0.0, f"Detection error: {str(e)}", None
    
    def detect_with_visualization(self, image: Image.Image) -> Tuple[bool, float, Image.Image]:
        """
        Detect cat and return image with bounding boxes
        
        Args:
            image: PIL Image
            
        Returns:
            cat_found: True if cat detected
            confidence: Detection confidence
            annotated_image: Image with bounding boxes drawn
        """
        if self.model is None:
            self.load_model()
        
        try:
            # Run detection
            results = self.model(image, verbose=False)
            
            # Get annotated image
            annotated = results[0].plot()  # Returns numpy array with boxes drawn
            annotated_image = Image.fromarray(annotated)
            
            # Check for cats
            cat_found = False
            max_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls == self.cat_class_id and conf >= self.confidence_threshold:
                        cat_found = True
                        max_confidence = max(max_confidence, conf)
            
            return cat_found, max_confidence, annotated_image
            
        except Exception as e:
            # Return original image on error
            return False, 0.0, image


class DetectionPipeline:
    """
    Complete detection pipeline for Stage 1
    Handles image preprocessing and post-processing
    """
    
    def __init__(self, yolo_model_path: str = 'yolo11n.pt', confidence_threshold: float = 0.25):
        """
        Initialize detection pipeline
        
        Args:
            yolo_model_path: Path to YOLO weights
            confidence_threshold: Detection threshold
        """
        self.detector = YOLOCatDetector(yolo_model_path, confidence_threshold)
        self.detector.load_model()
    
    def process(self, image: Image.Image, skip_detection: bool = False) -> dict:
        """
        Process image through detection pipeline
        
        Args:
            image: Input PIL Image
            skip_detection: If True, skip YOLO detection
            
        Returns:
            result: Dict with detection results
        """
        if skip_detection:
            return {
                'cat_detected': True,
                'confidence': 1.0,
                'message': 'Detection skipped by user',
                'skipped': True,
                'proceed_to_classification': True
            }
        
        # Run detection
        cat_found, confidence, message, details = self.detector.detect(image)
        
        return {
            'cat_detected': cat_found,
            'confidence': confidence,
            'message': message,
            'details': details,
            'skipped': False,
            'proceed_to_classification': cat_found
        }
    
    def get_cropped_cat(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Detect cat and crop to bounding box
        Useful for focusing classification on the cat region
        
        Args:
            image: Input PIL Image
            
        Returns:
            cropped_image: Cropped image containing cat, or None if no cat
        """
        cat_found, confidence, message, details = self.detector.detect(image)
        
        if not cat_found or details is None:
            return None
        
        # Get best detection bbox
        bbox = details['best_detection']['bbox']  # [x1, y1, x2, y2]
        
        # Crop image
        cropped = image.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])))
        
        return cropped


def test_detection_pipeline():
    """Test YOLO detection pipeline"""
    print("=" * 60)
    print("Testing Stage 1: YOLO11n Cat Detection")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DetectionPipeline()
    
    # Create dummy image
    dummy_image = Image.new('RGB', (640, 480), color=(73, 109, 137))
    
    # Test detection
    result = pipeline.process(dummy_image)
    
    print(f"\nDetection Result:")
    print(f"  Cat detected: {result['cat_detected']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Message: {result['message']}")
    print(f"  Proceed to classification: {result['proceed_to_classification']}")
    
    # Test skip detection
    result_skipped = pipeline.process(dummy_image, skip_detection=True)
    print(f"\nSkipped Detection Result:")
    print(f"  Cat detected: {result_skipped['cat_detected']}")
    print(f"  Skipped: {result_skipped['skipped']}")
    print(f"  Proceed to classification: {result_skipped['proceed_to_classification']}")
    
    print("\n" + "=" * 60)
    print("✅ Stage 1 test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_detection_pipeline()
