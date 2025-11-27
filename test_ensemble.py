"""
Quick Test: Ensemble Model Components
Test model creation and basic functionality
"""

import torch
from ensemble_model import (
    ResNet50Classifier, 
    EfficientNetB3Classifier, 
    ConvNeXtClassifier,
    StackingEnsemble,
    MetaLearner
)

def test_single_model(ModelClass, model_name):
    """Test a single model"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print('='*60)
    
    try:
        model = ModelClass(num_classes=59)
        print(f"✓ {model_name} created")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            features = model.get_features(dummy_input)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Feature shape: {features.shape}")
        print(f"✓ {model_name} forward pass successful")
        
        return True
    
    except Exception as e:
        print(f"✗ {model_name} failed: {str(e)}")
        return False

def test_meta_learner():
    """Test meta-learner"""
    print(f"\n{'='*60}")
    print("Testing Meta-Learner")
    print('='*60)
    
    try:
        meta = MetaLearner(num_classes=59, feature_dim=3)
        print("✓ Meta-Learner created")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        meta = meta.to(device)
        
        # Simulated concatenated predictions from 3 models
        dummy_input = torch.randn(2, 59 * 3).to(device)
        
        with torch.no_grad():
            output = meta(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print("✓ Meta-Learner forward pass successful")
        
        return True
    
    except Exception as e:
        print(f"✗ Meta-Learner failed: {str(e)}")
        return False

def test_ensemble():
    """Test full ensemble"""
    print(f"\n{'='*60}")
    print("Testing Full Ensemble")
    print('='*60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ensemble = StackingEnsemble(num_classes=59, device=device)
        ensemble = ensemble.to(device)
        print("✓ Ensemble created")
        
        # Count total parameters
        total_params = sum(p.numel() for p in ensemble.parameters())
        trainable_params = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        
        print("\n  Testing with meta-learner...")
        with torch.no_grad():
            output_meta = ensemble(dummy_input, use_meta=True)
        print(f"    Output shape: {output_meta.shape}")
        
        print("\n  Testing with averaging...")
        with torch.no_grad():
            output_avg = ensemble(dummy_input, use_meta=False)
        print(f"    Output shape: {output_avg.shape}")
        
        print("\n  Testing detailed predictions...")
        with torch.no_grad():
            details = ensemble.predict_with_details(dummy_input)
        
        for model_name, pred in details.items():
            print(f"    {model_name}: {pred.shape}")
        
        print("✓ Ensemble forward pass successful")
        
        # Test freeze/unfreeze
        print("\n  Testing freeze/unfreeze...")
        ensemble.freeze_base_models()
        frozen_trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        print(f"    Frozen trainable: {frozen_trainable:,}")
        
        ensemble.unfreeze_base_models()
        unfrozen_trainable = sum(p.numel() for p in ensemble.parameters() if p.requires_grad)
        print(f"    Unfrozen trainable: {unfrozen_trainable:,}")
        
        print("✓ Freeze/unfreeze works")
        
        return True
    
    except Exception as e:
        print(f"✗ Ensemble failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("ENSEMBLE MODEL COMPONENT TEST")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    results = []
    
    # Test individual models
    results.append(("ResNet-50", test_single_model(ResNet50Classifier, "ResNet-50")))
    results.append(("EfficientNet-B3", test_single_model(EfficientNetB3Classifier, "EfficientNet-B3")))
    results.append(("ConvNeXt-Tiny", test_single_model(ConvNeXtClassifier, "ConvNeXt-Tiny")))
    
    # Test meta-learner
    results.append(("Meta-Learner", test_meta_learner()))
    
    # Test full ensemble
    results.append(("Full Ensemble", test_ensemble()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:20s} {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("\nReady to train:")
        print("  python train_ensemble.py --epochs-base 15 --epochs-meta 10")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix errors before training")
    print("="*60)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
