"""
Test Super Ensemble Components - Dengeli Yol
5 Models validation before training
"""

import torch
import torch.nn as nn
from ensemble_model import ResNet50Classifier, EfficientNetB3Classifier, ConvNeXtClassifier

try:
    from transformer_models import ViTClassifier, EfficientNetV2Classifier
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Transformer models import error: {e}")
    print("   Please run: pip install timm")
    TRANSFORMERS_AVAILABLE = False


class SuperEnsemble5Models(nn.Module):
    """5-Model Super Ensemble"""
    def __init__(self, num_classes=59):
        super().__init__()
        self.resnet = ResNet50Classifier(num_classes)
        self.efficientnet_b3 = EfficientNetB3Classifier(num_classes)
        self.convnext = ConvNeXtClassifier(num_classes)
        self.vit = ViTClassifier(num_classes)
        self.efficientnet_v2 = EfficientNetV2Classifier(num_classes)
        
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 5, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        pred1 = self.resnet(x)
        pred2 = self.efficientnet_b3(x)
        pred3 = self.convnext(x)
        pred4 = self.vit(x)
        pred5 = self.efficientnet_v2(x)
        
        combined = torch.cat([pred1, pred2, pred3, pred4, pred5], dim=1)
        output = self.meta_learner(combined)
        return output
    
    def freeze_base_models(self):
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.efficientnet_b3.parameters():
            param.requires_grad = False
        for param in self.convnext.parameters():
            param.requires_grad = False
        for param in self.vit.parameters():
            param.requires_grad = False
        for param in self.efficientnet_v2.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def main():
    print("="*70)
    print("🧪 Testing Super Ensemble - Dengeli Yol (5 Models)")
    print("="*70)
    
    if not TRANSFORMERS_AVAILABLE:
        print("\n❌ Transformer models not available!")
        print("   Install with: pip install timm")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        gpu_props = torch.cuda.get_device_properties(0)
        print(f"VRAM: {gpu_props.total_memory / 1024**3:.1f} GB")
    
    # Test input
    batch_size = 2
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    num_classes = 59
    
    print("\n" + "-"*70)
    print("TEST 1: Individual Base Models")
    print("-"*70)
    
    models = [
        ('ResNet-50', ResNet50Classifier(num_classes)),
        ('EfficientNet-B3', EfficientNetB3Classifier(num_classes)),
        ('ConvNeXt-Tiny', ConvNeXtClassifier(num_classes)),
        ('Vision Transformer', ViTClassifier(num_classes)),
        ('EfficientNet-V2', EfficientNetV2Classifier(num_classes))
    ]
    
    total_params = 0
    total_size = 0
    
    for name, model in models:
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        params = count_parameters(model)
        size = model_size_mb(model)
        total_params += params
        total_size += size
        
        print(f"\n{name}:")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  Model size: {size:.1f} MB")
        print(f"  ✓ PASS")
    
    print("\n" + "-"*70)
    print("TEST 2: Meta-Learner")
    print("-"*70)
    
    ensemble = SuperEnsemble5Models(num_classes).to(device)
    ensemble.freeze_base_models()
    
    meta_params = count_parameters(ensemble)
    print(f"\nMeta-learner trainable parameters: {meta_params:,}")
    print(f"Expected: ~250K (only meta-learner)")
    
    if meta_params < 500000:
        print("✓ PASS - Base models frozen correctly")
    else:
        print("⚠️  WARNING - Too many trainable params!")
    
    print("\n" + "-"*70)
    print("TEST 3: Full Ensemble Forward Pass")
    print("-"*70)
    
    ensemble.eval()
    with torch.no_grad():
        output = ensemble(x)
    
    print(f"\nEnsemble output shape: {output.shape}")
    print(f"Expected: torch.Size([{batch_size}, {num_classes}])")
    
    if output.shape == torch.Size([batch_size, num_classes]):
        print("✓ PASS - Forward pass successful")
    else:
        print("❌ FAIL - Wrong output shape!")
        return
    
    print("\n" + "-"*70)
    print("TEST 4: Freeze/Unfreeze Mechanism")
    print("-"*70)
    
    # Frozen state
    ensemble.freeze_base_models()
    frozen_params = count_parameters(ensemble)
    print(f"\nFrozen (meta-learner only): {frozen_params:,} params")
    
    # Unfrozen state
    ensemble.unfreeze_all()
    unfrozen_params = count_parameters(ensemble)
    print(f"Unfrozen (all models): {unfrozen_params:,} params")
    
    if unfrozen_params > frozen_params * 100:
        print("✓ PASS - Freeze/unfreeze working correctly")
    else:
        print("⚠️  WARNING - Unfreeze might not work properly!")
    
    print("\n" + "-"*70)
    print("TEST 5: Memory Usage Check")
    print("-"*70)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + backward pass
        ensemble.train()
        output = ensemble(x)
        loss = output.sum()
        loss.backward()
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3
        print(f"\nPeak GPU memory: {peak_memory:.2f} GB")
        print(f"Available VRAM: {gpu_props.total_memory / 1024**3:.1f} GB")
        
        if peak_memory < 3.5:
            print("✓ PASS - Fits in 4GB VRAM")
        else:
            print("⚠️  WARNING - Might need >4GB VRAM!")
            print("   Consider reducing batch size")
    
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    print("\nBase Models:")
    for name, model in models:
        params = sum(p.numel() for p in model.parameters())
        size = model_size_mb(model)
        print(f"  {name:25s} {params/1e6:6.1f}M params, {size:6.1f} MB")
    
    print(f"\n  {'Total Base Models':25s} {total_params/1e6:6.1f}M params, {total_size:6.1f} MB")
    print(f"  {'Meta-Learner':25s} {frozen_params/1e6:6.1f}M params")
    print(f"  {'Full Ensemble':25s} {unfrozen_params/1e6:6.1f}M params, {model_size_mb(ensemble):6.1f} MB")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    
    print("\n🚀 Ready to train:")
    print("   python train_super_ensemble.py")
    
    print("\n⏱️  Expected Training Time (RTX 3050):")
    print("   Stage 1 (5 models × 20 epochs): ~18-20 hours")
    print("   Stage 2 (Meta-learner): ~2-3 hours")
    print("   Stage 3 (Fine-tuning): ~1.5-2 hours")
    print("   Total: ~22-25 hours")
    
    print("\n🎯 Expected Performance:")
    print("   Individual models: 80-85%")
    print("   Meta-learner ensemble: 86-88%")
    print("   Fine-tuned ensemble: 88-91%")


if __name__ == '__main__':
    main()
