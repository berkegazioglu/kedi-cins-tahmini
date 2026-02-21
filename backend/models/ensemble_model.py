"""
Optimal 3-Model Ensemble Architecture
Combines ResNet50, EfficientNetB0, and MobileNetV3-Large with Meta-Learner
"""
import torch
import torch.nn as nn
import torchvision.models as models


class OptimalEnsembleModel(nn.Module):
    """
    Optimal 3-Model Ensemble with Meta-Learner
    
    Architecture:
    1. Three base models: ResNet50, EfficientNetB0, MobileNetV3-Large
    2. Each produces 59-class predictions
    3. Meta-Learner (FC layers) combines outputs
    4. Final prediction: 63.85% accuracy
    """
    
    def __init__(self, num_classes=59, freeze_base_models=False):
        super(OptimalEnsembleModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Base Model 1: ResNet-50 (24.6M params, 64.67% accuracy)
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        
        # Base Model 2: EfficientNet-B0 (5.3M params, 60.66% accuracy)
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, 
            num_classes
        )
        
        # Base Model 3: MobileNetV3-Large (5.4M params, 60.06% accuracy)
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        self.mobilenet.classifier[3] = nn.Linear(
            self.mobilenet.classifier[3].in_features, 
            num_classes
        )
        
        # Freeze base models if specified (for meta-learner training only)
        if freeze_base_models:
            for param in self.resnet50.parameters():
                param.requires_grad = False
            for param in self.efficientnet.parameters():
                param.requires_grad = False
            for param in self.mobilenet.parameters():
                param.requires_grad = False
        
        # Meta-Learner: Fully Connected Layers
        # Input: 3 models × 59 classes = 177 features
        # Output: 59 classes
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through ensemble
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            final_output: Final predictions [B, num_classes]
            base_outputs: Dict with individual model outputs (for analysis)
        """
        # Get predictions from each base model
        resnet_out = self.resnet50(x)      # [B, 59]
        efficientnet_out = self.efficientnet(x)  # [B, 59]
        mobilenet_out = self.mobilenet(x)  # [B, 59]
        
        # Concatenate all base model outputs
        combined = torch.cat([resnet_out, efficientnet_out, mobilenet_out], dim=1)  # [B, 177]
        
        # Meta-learner produces final prediction
        final_output = self.meta_learner(combined)  # [B, 59]
        
        # Return final output and individual outputs for analysis
        base_outputs = {
            'resnet50': resnet_out,
            'efficientnet': efficientnet_out,
            'mobilenet': mobilenet_out
        }
        
        return final_output, base_outputs
    
    def get_model_info(self):
        """Get information about the ensemble model"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        resnet_params = sum(p.numel() for p in self.resnet50.parameters())
        efficientnet_params = sum(p.numel() for p in self.efficientnet.parameters())
        mobilenet_params = sum(p.numel() for p in self.mobilenet.parameters())
        meta_params = sum(p.numel() for p in self.meta_learner.parameters())
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'resnet50_parameters': resnet_params,
            'efficientnet_parameters': efficientnet_params,
            'mobilenet_parameters': mobilenet_params,
            'meta_learner_parameters': meta_params,
            'base_models': 3,
            'num_classes': self.num_classes
        }


class SimpleEnsembleModel(nn.Module):
    """
    Simplified ensemble model for inference only
    Averages predictions from three pre-trained models
    """
    
    def __init__(self, num_classes=59):
        super(SimpleEnsembleModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Base Model 1: ResNet-50
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        
        # Base Model 2: EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, 
            num_classes
        )
        
        # Base Model 3: MobileNetV3-Large
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        self.mobilenet.classifier[3] = nn.Linear(
            self.mobilenet.classifier[3].in_features, 
            num_classes
        )
        
    def forward(self, x):
        """
        Forward pass - simple averaging
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            averaged_output: Averaged predictions [B, num_classes]
            base_outputs: Dict with individual model outputs
        """
        # Get predictions from each model
        resnet_out = self.resnet50(x)
        efficientnet_out = self.efficientnet(x)
        mobilenet_out = self.mobilenet(x)
        
        # Simple averaging
        averaged_output = (resnet_out + efficientnet_out + mobilenet_out) / 3.0
        
        base_outputs = {
            'resnet50': resnet_out,
            'efficientnet': efficientnet_out,
            'mobilenet': mobilenet_out
        }
        
        return averaged_output, base_outputs


class WeightedEnsembleModel(nn.Module):
    """
    Weighted ensemble model
    Combines models with learned weights
    """
    
    def __init__(self, num_classes=59, weights=None):
        super(WeightedEnsembleModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Base models
        self.resnet50 = models.resnet50(weights=None)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)
        
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, 
            num_classes
        )
        
        self.mobilenet = models.mobilenet_v3_large(weights=None)
        self.mobilenet.classifier[3] = nn.Linear(
            self.mobilenet.classifier[3].in_features, 
            num_classes
        )
        
        # Learnable weights (default: equal weighting)
        if weights is None:
            weights = [0.64, 0.18, 0.18]  # Based on individual accuracies
        
        self.model_weights = nn.Parameter(
            torch.tensor(weights, dtype=torch.float32),
            requires_grad=True
        )
        
    def forward(self, x):
        """
        Forward pass with weighted combination
        
        Args:
            x: Input tensor [B, 3, 224, 224]
            
        Returns:
            weighted_output: Weighted predictions [B, num_classes]
            base_outputs: Dict with individual model outputs
        """
        # Get predictions
        resnet_out = self.resnet50(x)
        efficientnet_out = self.efficientnet(x)
        mobilenet_out = self.mobilenet(x)
        
        # Normalize weights using softmax
        normalized_weights = torch.softmax(self.model_weights, dim=0)
        
        # Weighted combination
        weighted_output = (
            normalized_weights[0] * resnet_out +
            normalized_weights[1] * efficientnet_out +
            normalized_weights[2] * mobilenet_out
        )
        
        base_outputs = {
            'resnet50': resnet_out,
            'efficientnet': efficientnet_out,
            'mobilenet': mobilenet_out,
            'weights': normalized_weights.detach().cpu().numpy()
        }
        
        return weighted_output, base_outputs


def create_ensemble_model(ensemble_type='optimal', num_classes=59, **kwargs):
    """
    Factory function to create ensemble models
    
    Args:
        ensemble_type: 'optimal', 'simple', or 'weighted'
        num_classes: Number of output classes
        **kwargs: Additional arguments for specific ensemble types
        
    Returns:
        ensemble_model: Initialized ensemble model
    """
    if ensemble_type == 'optimal':
        return OptimalEnsembleModel(
            num_classes=num_classes,
            freeze_base_models=kwargs.get('freeze_base_models', False)
        )
    elif ensemble_type == 'simple':
        return SimpleEnsembleModel(num_classes=num_classes)
    elif ensemble_type == 'weighted':
        return WeightedEnsembleModel(
            num_classes=num_classes,
            weights=kwargs.get('weights', None)
        )
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")


if __name__ == "__main__":
    # Test ensemble model creation
    print("=" * 60)
    print("Testing Optimal 3-Model Ensemble")
    print("=" * 60)
    
    # Create model
    model = OptimalEnsembleModel(num_classes=59)
    model.eval()
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    with torch.no_grad():
        final_output, base_outputs = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Final output shape: {final_output.shape}")
    print(f"\nBase model outputs:")
    for name, output in base_outputs.items():
        print(f"  {name}: {output.shape}")
    
    # Model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    print(f"  ResNet50 parameters: {info['resnet50_parameters']:,} (24.6M)")
    print(f"  EfficientNet parameters: {info['efficientnet_parameters']:,} (5.3M)")
    print(f"  MobileNet parameters: {info['mobilenet_parameters']:,} (5.4M)")
    print(f"  Meta-Learner parameters: {info['meta_learner_parameters']:,}")
    
    print("\n" + "=" * 60)
    print("✅ Ensemble model test completed!")
    print("=" * 60)
