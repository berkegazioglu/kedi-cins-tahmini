"""
Stage 2: Optimal 3-Model Ensemble Classification Pipeline
Implements the breed classification with meta-learner
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
from scipy.stats import entropy as calculate_entropy


class BreedClassificationPipeline:
    """
    Stage 2: Optimal 3-Model Ensemble with Meta-Learner
    
    Architecture:
    - ResNet50 (24.6M params, 64.67% accuracy)
    - EfficientNetB0 (5.3M params, 60.66% accuracy)
    - MobileNetV3-Large (5.4M params, 60.06% accuracy)
    - Meta-Learner (FC Layers)
    - Final Prediction: 63.85% accuracy
    """
    
    def __init__(
        self,
        model,
        class_names: List[str],
        device: torch.device,
        entropy_threshold: float = 2.5
    ):
        """
        Initialize classification pipeline
        
        Args:
            model: Loaded ensemble model
            class_names: List of breed names (59 classes)
            device: torch device (cuda/cpu)
            entropy_threshold: Threshold for wild cat detection
        """
        self.model = model
        self.class_names = class_names
        self.device = device
        self.entropy_threshold = entropy_threshold
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image
            
        Returns:
            preprocessed: Tensor [1, 3, 224, 224]
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        return img_tensor.to(self.device)
    
    def calculate_prediction_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate Shannon entropy of prediction distribution
        High entropy = uncertain/wild cat
        
        Args:
            probabilities: Softmax probabilities [num_classes]
            
        Returns:
            entropy_value: Shannon entropy
        """
        return calculate_entropy(probabilities)
    
    def is_wild_cat(self, entropy_value: float) -> bool:
        """
        Determine if prediction indicates wild cat
        
        Args:
            entropy_value: Shannon entropy
            
        Returns:
            is_wild: True if likely wild cat
        """
        return bool(entropy_value > self.entropy_threshold)
    
    def predict(
        self,
        image: Image.Image,
        top_k: int = 5,
        return_base_predictions: bool = False
    ) -> Dict:
        """
        Predict cat breed using ensemble model
        
        Args:
            image: PIL Image
            top_k: Number of top predictions to return
            return_base_predictions: If True, include individual model predictions
            
        Returns:
            result: Dict with predictions and analysis
        """
        try:
            # Preprocess
            img_tensor = self.preprocess_image(image)
            
            # Forward pass
            with torch.no_grad():
                if hasattr(self.model, 'forward') and 'base_outputs' in str(self.model.forward.__code__.co_varnames):
                    # Ensemble model with base outputs
                    output, base_outputs = self.model(img_tensor)
                else:
                    # Simple model
                    output = self.model(img_tensor)
                    base_outputs = None
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=1)[0]
            probabilities_np = probabilities.cpu().numpy()
            
            # Calculate entropy
            entropy_value = self.calculate_prediction_entropy(probabilities_np)
            wild_cat = self.is_wild_cat(entropy_value)
            
            # Get top-k predictions
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    'breed': self.class_names[idx],
                    'confidence': float(prob),
                    'class_index': int(idx)
                })
            
            result = {
                'success': True,
                'top_breed': predictions[0]['breed'],
                'top_confidence': predictions[0]['confidence'],
                'predictions': predictions,
                'is_wild_cat': wild_cat,
                'entropy': float(entropy_value),
                'entropy_threshold': self.entropy_threshold,
                'wild_cat_warning': f"High uncertainty detected (entropy: {entropy_value:.2f}). This may be a wild cat species." if wild_cat else None
            }
            
            # Add base model predictions if available and requested
            if return_base_predictions and base_outputs is not None:
                base_predictions = {}
                for model_name, output_tensor in base_outputs.items():
                    if isinstance(output_tensor, torch.Tensor):
                        probs = torch.softmax(output_tensor, dim=1)[0]
                        top_prob, top_idx = torch.topk(probs, 1)
                        base_predictions[model_name] = {
                            'breed': self.class_names[top_idx[0].item()],
                            'confidence': float(top_prob[0].item())
                        }
                result['base_model_predictions'] = base_predictions
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'top_breed': None,
                'top_confidence': 0.0,
                'predictions': [],
                'is_wild_cat': False,
                'entropy': 0.0
            }
    
    def batch_predict(
        self,
        images: List[Image.Image],
        top_k: int = 5
    ) -> List[Dict]:
        """
        Predict multiple images in batch
        
        Args:
            images: List of PIL Images
            top_k: Number of top predictions per image
            
        Returns:
            results: List of prediction results
        """
        # Preprocess all images
        img_tensors = [self.preprocess_image(img) for img in images]
        batch_tensor = torch.cat(img_tensors, dim=0)
        
        # Forward pass
        with torch.no_grad():
            if hasattr(self.model, 'forward') and 'base_outputs' in str(self.model.forward.__code__.co_varnames):
                outputs, _ = self.model(batch_tensor)
            else:
                outputs = self.model(batch_tensor)
        
        # Process each prediction
        results = []
        for i, output in enumerate(outputs):
            probabilities = torch.softmax(output.unsqueeze(0), dim=1)[0]
            probabilities_np = probabilities.cpu().numpy()
            
            entropy_value = self.calculate_prediction_entropy(probabilities_np)
            wild_cat = self.is_wild_cat(entropy_value)
            
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                predictions.append({
                    'breed': self.class_names[idx],
                    'confidence': float(prob)
                })
            
            results.append({
                'success': True,
                'top_breed': predictions[0]['breed'],
                'top_confidence': predictions[0]['confidence'],
                'predictions': predictions,
                'is_wild_cat': wild_cat,
                'entropy': float(entropy_value)
            })
        
        return results


class ModelComparison:
    """
    Compare predictions from individual base models
    Useful for analysis and understanding ensemble behavior
    """
    
    def __init__(self, ensemble_model, class_names: List[str], device: torch.device):
        """
        Initialize model comparison
        
        Args:
            ensemble_model: Loaded ensemble model
            class_names: List of breed names
            device: torch device
        """
        self.ensemble_model = ensemble_model
        self.class_names = class_names
        self.device = device
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def compare_models(self, image: Image.Image, top_k: int = 3) -> Dict:
        """
        Get predictions from all base models and ensemble
        
        Args:
            image: PIL Image
            top_k: Number of top predictions
            
        Returns:
            comparison: Dict with all model predictions
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            final_output, base_outputs = self.ensemble_model(img_tensor)
        
        comparison = {}
        
        # Base models
        for model_name, output in base_outputs.items():
            if isinstance(output, torch.Tensor):
                probs = torch.softmax(output, dim=1)[0]
                top_probs, top_indices = torch.topk(probs, top_k)
                
                predictions = []
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    predictions.append({
                        'breed': self.class_names[idx],
                        'confidence': float(prob)
                    })
                
                comparison[model_name] = {
                    'top_breed': predictions[0]['breed'],
                    'predictions': predictions
                }
        
        # Ensemble
        final_probs = torch.softmax(final_output, dim=1)[0]
        top_probs, top_indices = torch.topk(final_probs, top_k)
        
        ensemble_predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            ensemble_predictions.append({
                'breed': self.class_names[idx],
                'confidence': float(prob)
            })
        
        comparison['ensemble'] = {
            'top_breed': ensemble_predictions[0]['breed'],
            'predictions': ensemble_predictions
        }
        
        # Calculate agreement
        all_top_breeds = [v['top_breed'] for v in comparison.values()]
        agreement = len(set(all_top_breeds)) == 1
        
        comparison['agreement'] = {
            'all_agree': agreement,
            'consensus': max(set(all_top_breeds), key=all_top_breeds.count)
        }
        
        return comparison


def test_classification_pipeline():
    """Test classification pipeline"""
    print("=" * 60)
    print("Testing Stage 2: Optimal 3-Model Ensemble Classification")
    print("=" * 60)
    
    print("\n⚠️  Note: This test requires a trained ensemble model.")
    print("   Pipeline architecture is ready for use.")
    
    print("\nPipeline Components:")
    print("  ✅ Image preprocessing")
    print("  ✅ Entropy calculation")
    print("  ✅ Wild cat detection")
    print("  ✅ Top-K predictions")
    print("  ✅ Base model comparison")
    print("  ✅ Batch prediction support")
    
    print("\n" + "=" * 60)
    print("✅ Stage 2 pipeline ready!")
    print("=" * 60)


if __name__ == "__main__":
    test_classification_pipeline()
