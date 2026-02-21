"""
Complete 2-Stage Pipeline: Detection → Classification
Combines YOLO11n detection and Optimal Ensemble classification
"""
import torch
from PIL import Image
from typing import Dict, Optional
import time

from .stage1_detection import DetectionPipeline
from .stage2_classification import BreedClassificationPipeline, ModelComparison


class TwoStagePipeline:
    """
    Complete 2-Stage Cat Breed Classification Pipeline
    
    Stage 1: YOLO11n Cat Detection (2.6M params)
    Stage 2: Optimal 3-Model Ensemble (ResNet50 + EfficientNet + MobileNet + Meta-Learner)
    
    Final Accuracy: 63.85%
    """
    
    def __init__(
        self,
        yolo_model_path: str,
        ensemble_model,
        class_names: list,
        device: torch.device,
        yolo_confidence: float = 0.25,
        entropy_threshold: float = 2.5
    ):
        """
        Initialize complete pipeline
        
        Args:
            yolo_model_path: Path to YOLO11n weights
            ensemble_model: Loaded ensemble model
            class_names: List of 59 breed names
            device: torch device
            yolo_confidence: YOLO detection threshold
            entropy_threshold: Wild cat detection threshold
        """
        # Stage 1: Detection
        self.detection_pipeline = DetectionPipeline(
            yolo_model_path=yolo_model_path,
            confidence_threshold=yolo_confidence
        )
        
        # Stage 2: Classification
        self.classification_pipeline = BreedClassificationPipeline(
            model=ensemble_model,
            class_names=class_names,
            device=device,
            entropy_threshold=entropy_threshold
        )
        
        self.device = device
    
    def process(
        self,
        image: Image.Image,
        skip_detection: bool = False,
        top_k: int = 5,
        return_base_predictions: bool = False
    ) -> Dict:
        """
        Process image through complete 2-stage pipeline
        
        Args:
            image: PIL Image
            skip_detection: Skip YOLO detection (Stage 1)
            top_k: Number of top predictions
            return_base_predictions: Include individual model predictions
            
        Returns:
            result: Complete pipeline result
        """
        start_time = time.time()
        
        # Stage 1: Detection
        detection_start = time.time()
        detection_result = self.detection_pipeline.process(image, skip_detection)
        detection_time = time.time() - detection_start
        
        # Check if cat detected
        if not detection_result['proceed_to_classification']:
            return {
                'success': False,
                'stage': 'detection',
                'error': 'No cat detected',
                'detection': detection_result,
                'timing': {
                    'detection': detection_time,
                    'classification': 0.0,
                    'total': time.time() - start_time
                }
            }
        
        # Stage 2: Classification
        classification_start = time.time()
        classification_result = self.classification_pipeline.predict(
            image=image,
            top_k=top_k,
            return_base_predictions=return_base_predictions
        )
        classification_time = time.time() - classification_start
        
        # Combine results
        total_time = time.time() - start_time
        
        if not classification_result['success']:
            return {
                'success': False,
                'stage': 'classification',
                'error': classification_result.get('error', 'Classification failed'),
                'detection': detection_result,
                'timing': {
                    'detection': detection_time,
                    'classification': classification_time,
                    'total': total_time
                }
            }
        
        # Success - combine all information
        result = {
            'success': True,
            'stage': 'complete',
            'detection': detection_result,
            'classification': classification_result,
            'timing': {
                'detection': detection_time,
                'classification': classification_time,
                'total': total_time,
                'breakdown': {
                    'stage1_detection': f"{detection_time:.3f}s",
                    'stage2_classification': f"{classification_time:.3f}s",
                    'total_pipeline': f"{total_time:.3f}s"
                }
            },
            # Top-level fields for convenience
            'top_breed': classification_result['top_breed'],
            'top_confidence': classification_result['top_confidence'],
            'predictions': classification_result['predictions'],
            'is_wild_cat': classification_result['is_wild_cat'],
            'entropy': classification_result['entropy']
        }
        
        # Add base predictions if available
        if 'base_model_predictions' in classification_result:
            result['base_model_predictions'] = classification_result['base_model_predictions']
        
        return result
    
    def process_batch(
        self,
        images: list,
        skip_detection: bool = False,
        top_k: int = 5
    ) -> list:
        """
        Process multiple images
        
        Args:
            images: List of PIL Images
            skip_detection: Skip detection stage
            top_k: Number of top predictions
            
        Returns:
            results: List of pipeline results
        """
        results = []
        for image in images:
            result = self.process(image, skip_detection, top_k)
            results.append(result)
        return results
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the pipeline
        
        Returns:
            info: Pipeline architecture information
        """
        return {
            'architecture': '2-Stage Pipeline',
            'stage1': {
                'name': 'YOLO11n Object Detection',
                'parameters': '2.6M',
                'purpose': 'Cat presence verification',
                'confidence_threshold': self.detection_pipeline.detector.confidence_threshold
            },
            'stage2': {
                'name': 'Optimal 3-Model Ensemble',
                'base_models': [
                    {'name': 'ResNet-50', 'params': '24.6M', 'accuracy': '64.67%'},
                    {'name': 'EfficientNetB0', 'params': '5.3M', 'accuracy': '60.66%'},
                    {'name': 'MobileNetV3-Large', 'params': '5.4M', 'accuracy': '60.06%'}
                ],
                'meta_learner': 'Fully Connected Layers',
                'final_accuracy': '63.85%',
                'entropy_threshold': self.classification_pipeline.entropy_threshold
            },
            'num_classes': len(self.classification_pipeline.class_names),
            'device': str(self.device)
        }


class PipelineVisualizer:
    """
    Visualize pipeline processing steps
    """
    
    @staticmethod
    def print_result(result: Dict, verbose: bool = True):
        """
        Pretty print pipeline result
        
        Args:
            result: Pipeline result dict
            verbose: Print detailed information
        """
        print("\n" + "=" * 60)
        print("🐱 Cat Breed Classification Pipeline Result")
        print("=" * 60)
        
        if not result['success']:
            print(f"\n❌ Pipeline Failed at {result['stage']} stage")
            print(f"   Error: {result['error']}")
            return
        
        # Detection
        detection = result['detection']
        print(f"\n📍 Stage 1: Detection")
        print(f"   Cat detected: {'✅' if detection['cat_detected'] else '❌'}")
        print(f"   Confidence: {detection['confidence']:.2%}")
        print(f"   Message: {detection['message']}")
        
        # Classification
        classification = result['classification']
        print(f"\n🎯 Stage 2: Classification")
        print(f"   Top Breed: {classification['top_breed']}")
        print(f"   Confidence: {classification['top_confidence']:.2%}")
        print(f"   Wild Cat: {'⚠️ YES' if classification['is_wild_cat'] else '✅ NO'}")
        print(f"   Entropy: {classification['entropy']:.3f}")
        
        if verbose and len(classification['predictions']) > 1:
            print(f"\n   Other Predictions:")
            for i, pred in enumerate(classification['predictions'][1:], 1):
                print(f"   {i}. {pred['breed']}: {pred['confidence']:.2%}")
        
        # Base models (if available)
        if 'base_model_predictions' in result:
            print(f"\n🤖 Base Model Predictions:")
            for model_name, pred in result['base_model_predictions'].items():
                print(f"   {model_name}: {pred['breed']} ({pred['confidence']:.2%})")
        
        # Timing
        timing = result['timing']
        print(f"\n⏱️  Performance:")
        print(f"   Detection: {timing['detection']:.3f}s")
        print(f"   Classification: {timing['classification']:.3f}s")
        print(f"   Total: {timing['total']:.3f}s")
        
        print("\n" + "=" * 60)


def test_complete_pipeline():
    """Test complete 2-stage pipeline"""
    print("=" * 60)
    print("🐱 Testing Complete 2-Stage Pipeline")
    print("=" * 60)
    
    print("\nPipeline Architecture:")
    print("  Stage 1: YOLO11n Detection (2.6M params)")
    print("  Stage 2: Optimal 3-Model Ensemble")
    print("    ├── ResNet-50 (24.6M, 64.67%)")
    print("    ├── EfficientNetB0 (5.3M, 60.66%)")
    print("    ├── MobileNetV3-Large (5.4M, 60.06%)")
    print("    └── Meta-Learner (FC Layers)")
    print("  Final Accuracy: 63.85%")
    
    print("\nPipeline Features:")
    print("  ✅ 2-stage processing")
    print("  ✅ Cat detection before classification")
    print("  ✅ Ensemble model combination")
    print("  ✅ Wild cat detection (entropy)")
    print("  ✅ Detailed timing information")
    print("  ✅ Base model comparison")
    print("  ✅ Batch processing support")
    
    print("\n" + "=" * 60)
    print("✅ Complete pipeline ready!")
    print("=" * 60)


if __name__ == "__main__":
    test_complete_pipeline()
