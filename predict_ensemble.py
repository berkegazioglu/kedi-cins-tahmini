"""
Predict with Ensemble Model
Command-line interface for predictions
"""

import torch
from PIL import Image
import argparse
import json
from torchvision import transforms
import numpy as np
from ensemble_model import StackingEnsemble

def load_image(image_path):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0), image

def predict(model, image_tensor, classes, device, top_k=5):
    """Make prediction"""
    image_tensor = image_tensor.to(device)
    model.eval()
    
    with torch.no_grad():
        # Get all predictions
        predictions = model.predict_with_details(image_tensor)
        
        results = {}
        for model_name, pred in predictions.items():
            probs = pred[0].cpu().numpy()
            top_idx = np.argsort(probs)[-top_k:][::-1]
            
            results[model_name] = {
                'classes': [classes[i] for i in top_idx],
                'confidences': [float(probs[i]) * 100 for i in top_idx],
                'top_class': classes[top_idx[0]],
                'top_confidence': float(probs[top_idx[0]]) * 100
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Ensemble Model Prediction')
    parser.add_argument('--image', required=True, help='Path to image')
    parser.add_argument('--model', default='runs/ensemble/ensemble_finetuned_best.pth', 
                       help='Path to ensemble model')
    parser.add_argument('--summary', default='runs/ensemble/training_summary.json',
                       help='Path to training summary')
    parser.add_argument('--top-k', type=int, default=5, help='Top K predictions')
    parser.add_argument('--show-all', action='store_true', help='Show all model predictions')
    args = parser.parse_args()
    
    # Load classes
    with open(args.summary, 'r') as f:
        summary = json.load(f)
    classes = summary['classes']
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print(f"Loading ensemble model from {args.model}...")
    model = StackingEnsemble(num_classes=len(classes), device=device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    model.to(device)
    print("✓ Model loaded")
    
    # Load image
    print(f"Loading image: {args.image}")
    image_tensor, original_image = load_image(args.image)
    print(f"✓ Image loaded: {original_image.size}")
    
    # Predict
    print("\nMaking predictions...")
    results = predict(model, image_tensor, classes, device, args.top_k)
    
    # Display results
    print("\n" + "="*70)
    print("🏆 ENSEMBLE PREDICTION (META-LEARNER)")
    print("="*70)
    ensemble_result = results['Ensemble']
    print(f"Top Prediction: {ensemble_result['top_class']}")
    print(f"Confidence: {ensemble_result['top_confidence']:.2f}%")
    print(f"\nTop {args.top_k} Predictions:")
    for i, (cls, conf) in enumerate(zip(ensemble_result['classes'], 
                                        ensemble_result['confidences']), 1):
        bar = '█' * int(conf/2)
        print(f"{i}. {cls:30s} {conf:6.2f}% {bar}")
    
    if args.show_all:
        # Show individual model predictions
        for model_name in ['ResNet50', 'EfficientNetB3', 'ConvNeXt']:
            print("\n" + "-"*70)
            print(f"📊 {model_name} Predictions")
            print("-"*70)
            result = results[model_name]
            print(f"Top: {result['top_class']} ({result['top_confidence']:.2f}%)")
            print(f"\nTop {args.top_k}:")
            for i, (cls, conf) in enumerate(zip(result['classes'], 
                                                result['confidences']), 1):
                bar = '█' * int(conf/2)
                print(f"{i}. {cls:30s} {conf:6.2f}% {bar}")
        
        # Comparison
        print("\n" + "="*70)
        print("📈 MODEL COMPARISON")
        print("="*70)
        print(f"{'Model':<20} {'Top Prediction':<30} {'Confidence':>10}")
        print("-"*70)
        for model_name in ['ResNet50', 'EfficientNetB3', 'ConvNeXt', 'Ensemble']:
            result = results[model_name]
            print(f"{model_name:<20} {result['top_class']:<30} {result['top_confidence']:>9.2f}%")
        
        # Agreement check
        predictions_set = {results[m]['top_class'] for m in ['ResNet50', 'EfficientNetB3', 'ConvNeXt']}
        if len(predictions_set) == 1:
            print("\n✓ All base models agree!")
        else:
            print(f"\n⚠ Models disagree: {len(predictions_set)} different predictions")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
