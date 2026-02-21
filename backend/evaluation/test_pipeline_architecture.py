"""
Test script for 2-stage pipeline architecture
Tests both detection and classification stages
"""
import torch
from PIL import Image
from pathlib import Path

# Import pipeline components
from backend.models.stage1_detection import YOLOCatDetector, DetectionPipeline
from backend.models.stage2_classification import BreedClassificationPipeline
from backend.models.pipeline import TwoStagePipeline, PipelineVisualizer
from backend.models.ensemble_model import OptimalEnsembleModel


def test_stage1_detection():
    """Test Stage 1: YOLO Detection"""
    print("\n" + "="*60)
    print("Testing Stage 1: YOLO11n Cat Detection")
    print("="*60)
    
    try:
        detector = YOLOCatDetector(model_path='yolo11n.pt', confidence_threshold=0.25)
        detector.load_model()
        print("✅ YOLO11n model loaded successfully")
        
        # Test with dummy image
        dummy_image = Image.new('RGB', (640, 480), color='blue')
        cat_found, confidence, message, details = detector.detect(dummy_image)
        
        print(f"\nTest Detection Result:")
        print(f"  Cat found: {cat_found}")
        print(f"  Confidence: {confidence:.2%}")
        print(f"  Message: {message}")
        
        return True
    except Exception as e:
        print(f"❌ Stage 1 test failed: {e}")
        return False


def test_stage2_classification():
    """Test Stage 2: Ensemble Classification"""
    print("\n" + "="*60)
    print("Testing Stage 2: Optimal 3-Model Ensemble")
    print("="*60)
    
    try:
        # Create ensemble model
        model = OptimalEnsembleModel(num_classes=59)
        print("✅ Ensemble model created successfully")
        
        # Get model info
        info = model.get_model_info()
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {info['total_parameters']:,}")
        print(f"  Base models: {info['base_models']}")
        print(f"  ResNet-50: {info['resnet50_parameters']:,} params")
        print(f"  EfficientNet: {info['efficientnet_parameters']:,} params")
        print(f"  MobileNet: {info['mobilenet_parameters']:,} params")
        print(f"  Meta-Learner: {info['meta_learner_parameters']:,} params")
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            output, base_outputs = model(dummy_input)
        
        print(f"\nForward Pass Test:")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Base models outputs: {len(base_outputs)}")
        for name, out in base_outputs.items():
            if isinstance(out, torch.Tensor):
                print(f"    {name}: {out.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Stage 2 test failed: {e}")
        return False


def test_complete_pipeline():
    """Test Complete 2-Stage Pipeline"""
    print("\n" + "="*60)
    print("Testing Complete 2-Stage Pipeline")
    print("="*60)
    
    try:
        # Create models
        ensemble_model = OptimalEnsembleModel(num_classes=59)
        class_names = [f"Breed_{i}" for i in range(59)]  # Dummy class names
        device = torch.device('cpu')
        
        # Create pipeline
        pipeline = TwoStagePipeline(
            yolo_model_path='yolo11n.pt',
            ensemble_model=ensemble_model,
            class_names=class_names,
            device=device,
            yolo_confidence=0.25,
            entropy_threshold=2.5
        )
        print("✅ Complete pipeline created successfully")
        
        # Get pipeline info
        info = pipeline.get_pipeline_info()
        print(f"\nPipeline Information:")
        print(f"  Architecture: {info['architecture']}")
        print(f"  Stage 1: {info['stage1']['name']}")
        print(f"    Parameters: {info['stage1']['parameters']}")
        print(f"  Stage 2: {info['stage2']['name']}")
        print(f"    Base models: {len(info['stage2']['base_models'])}")
        print(f"    Final accuracy: {info['stage2']['final_accuracy']}")
        print(f"  Number of classes: {info['num_classes']}")
        print(f"  Device: {info['device']}")
        
        # Test with dummy image (skip detection for testing)
        dummy_image = Image.new('RGB', (640, 480), color='green')
        result = pipeline.process(dummy_image, skip_detection=True, top_k=3)
        
        print(f"\nPipeline Test Result:")
        print(f"  Success: {result['success']}")
        print(f"  Stage: {result['stage']}")
        if result['success']:
            print(f"  Top breed: {result['top_breed']}")
            print(f"  Confidence: {result['top_confidence']:.2%}")
            print(f"  Wild cat: {result['is_wild_cat']}")
            print(f"  Total time: {result['timing']['total']:.3f}s")
        
        return True
    except Exception as e:
        print(f"❌ Complete pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pipeline_with_real_image():
    """Test pipeline with a real image if available"""
    print("\n" + "="*60)
    print("Testing Pipeline with Real Image")
    print("="*60)
    
    # Look for test images
    test_images = [
        'test_cat.jpg',
        'sample.jpg',
        'uploads/test.jpg'
    ]
    
    test_image_path = None
    for img_path in test_images:
        if Path(img_path).exists():
            test_image_path = img_path
            break
    
    if test_image_path is None:
        print("⚠️  No test image found. Skipping real image test.")
        print("   Add a cat image as 'test_cat.jpg' to test with real data.")
        return False
    
    print(f"Found test image: {test_image_path}")
    
    try:
        # Load image
        image = Image.open(test_image_path)
        print(f"  Image size: {image.size}")
        print(f"  Image mode: {image.mode}")
        
        # Create pipeline
        ensemble_model = OptimalEnsembleModel(num_classes=59)
        class_names = [f"Breed_{i}" for i in range(59)]
        device = torch.device('cpu')
        
        pipeline = TwoStagePipeline(
            yolo_model_path='yolo11n.pt',
            ensemble_model=ensemble_model,
            class_names=class_names,
            device=device
        )
        
        # Process image
        print("\nProcessing image through pipeline...")
        result = pipeline.process(image, skip_detection=False, top_k=5)
        
        # Visualize result
        PipelineVisualizer.print_result(result, verbose=True)
        
        return True
    except Exception as e:
        print(f"❌ Real image test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" " * 15 + "🐱 2-STAGE PIPELINE TEST SUITE")
    print("="*70)
    
    results = []
    
    # Test Stage 1
    results.append(("Stage 1: Detection", test_stage1_detection()))
    
    # Test Stage 2
    results.append(("Stage 2: Classification", test_stage2_classification()))
    
    # Test Complete Pipeline
    results.append(("Complete Pipeline", test_complete_pipeline()))
    
    # Test with real image
    results.append(("Real Image Test", test_pipeline_with_real_image()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<50} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("\n" + "="*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)
    
    if passed == total:
        print("\n🎉 All tests passed! Pipeline is ready.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check errors above.")


if __name__ == "__main__":
    main()
