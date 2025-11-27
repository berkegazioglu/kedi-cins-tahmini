"""
Comprehensive Evaluation Script for Ensemble Models
Generates detailed metrics, confusion matrices, and analysis for academic reporting
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_recall_fscore_support,
    top_k_accuracy_score
)
import json
import os
from tqdm import tqdm
import argparse
from datetime import datetime
import pandas as pd

from ensemble_model import StackingEnsemble, ResNet50Classifier, EfficientNetB3Classifier, ConvNeXtClassifier


class MetricsCalculator:
    """Calculate comprehensive metrics for models"""
    
    def __init__(self, num_classes, class_names):
        self.num_classes = num_classes
        self.class_names = class_names
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        self.correct = 0
        self.total = 0
        self.loss_sum = 0.0
        self.batch_count = 0
    
    def update(self, outputs, labels, loss=None):
        """Update metrics with batch results"""
        probs = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())
        
        self.correct += predicted.eq(labels).sum().item()
        self.total += labels.size(0)
        
        if loss is not None:
            self.loss_sum += loss.item()
            self.batch_count += 1
    
    def compute(self):
        """Compute all metrics"""
        all_preds = np.array(self.all_preds)
        all_labels = np.array(self.all_labels)
        all_probs = np.array(self.all_probs)
        
        # Basic metrics
        accuracy = 100. * self.correct / self.total
        avg_loss = self.loss_sum / self.batch_count if self.batch_count > 0 else 0
        
        # Top-k accuracy
        top3_acc = top_k_accuracy_score(all_labels, all_probs, k=3) * 100
        top5_acc = top_k_accuracy_score(all_labels, all_probs, k=5) * 100
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        
        # Overall metrics
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'top3_accuracy': top3_acc,
            'top5_accuracy': top5_acc,
            'loss': avg_loss,
            'precision_macro': precision_macro * 100,
            'recall_macro': recall_macro * 100,
            'f1_macro': f1_macro * 100,
            'precision_weighted': precision_weighted * 100,
            'recall_weighted': recall_weighted * 100,
            'f1_weighted': f1_weighted * 100,
            'per_class': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': support
            },
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }


def evaluate_model(model, dataloader, criterion, device, num_classes, class_names, model_name):
    """Evaluate a single model"""
    model.eval()
    metrics_calc = MetricsCalculator(num_classes, class_names)
    
    print(f"\nEvaluating {model_name}...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f'{model_name}'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            metrics_calc.update(outputs, labels, loss)
    
    return metrics_calc.compute()


def evaluate_ensemble_detailed(ensemble, dataloader, criterion, device, num_classes, class_names):
    """Evaluate ensemble and all base models"""
    ensemble.eval()
    
    # Calculators for each model
    calc_resnet = MetricsCalculator(num_classes, class_names)
    calc_efficient = MetricsCalculator(num_classes, class_names)
    calc_convnext = MetricsCalculator(num_classes, class_names)
    calc_ensemble = MetricsCalculator(num_classes, class_names)
    calc_voting = MetricsCalculator(num_classes, class_names)
    
    print("\nEvaluating Ensemble (all models)...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Ensemble Eval'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Get individual predictions
            pred_resnet = ensemble.resnet50(inputs)
            pred_efficient = ensemble.efficientnet(inputs)
            pred_convnext = ensemble.convnext(inputs)
            
            # Ensemble prediction (meta-learner)
            pred_ensemble = ensemble(inputs, use_meta=True)
            
            # Simple voting (averaging)
            pred_voting = (pred_resnet + pred_efficient + pred_convnext) / 3.0
            
            # Calculate losses
            loss_resnet = criterion(pred_resnet, labels)
            loss_efficient = criterion(pred_efficient, labels)
            loss_convnext = criterion(pred_convnext, labels)
            loss_ensemble = criterion(pred_ensemble, labels)
            loss_voting = criterion(pred_voting, labels)
            
            # Update metrics
            calc_resnet.update(pred_resnet, labels, loss_resnet)
            calc_efficient.update(pred_efficient, labels, loss_efficient)
            calc_convnext.update(pred_convnext, labels, loss_convnext)
            calc_ensemble.update(pred_ensemble, labels, loss_ensemble)
            calc_voting.update(pred_voting, labels, loss_voting)
    
    return {
        'ResNet50': calc_resnet.compute(),
        'EfficientNetB3': calc_efficient.compute(),
        'ConvNeXt': calc_convnext.compute(),
        'Ensemble_Meta': calc_ensemble.compute(),
        'Ensemble_Voting': calc_voting.compute()
    }


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix', figsize=(20, 18)):
    """Plot and save confusion matrix"""
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title(title, fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_per_class_metrics(metrics, class_names, save_path, title='Per-Class Metrics'):
    """Plot per-class precision, recall, F1"""
    per_class = metrics['per_class']
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    x = np.arange(len(class_names))
    
    # Precision
    axes[0].bar(x, per_class['precision'] * 100, color='steelblue', alpha=0.7)
    axes[0].set_title(f'{title} - Precision', fontsize=14)
    axes[0].set_ylabel('Precision (%)', fontsize=12)
    axes[0].set_ylim([0, 100])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[1].bar(x, per_class['recall'] * 100, color='forestgreen', alpha=0.7)
    axes[1].set_title(f'{title} - Recall', fontsize=14)
    axes[1].set_ylabel('Recall (%)', fontsize=12)
    axes[1].set_ylim([0, 100])
    axes[1].grid(axis='y', alpha=0.3)
    
    # F1-Score
    axes[2].bar(x, per_class['f1'] * 100, color='coral', alpha=0.7)
    axes[2].set_title(f'{title} - F1-Score', fontsize=14)
    axes[2].set_ylabel('F1-Score (%)', fontsize=12)
    axes[2].set_xlabel('Class', fontsize=12)
    axes[2].set_ylim([0, 100])
    axes[2].grid(axis='y', alpha=0.3)
    
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=90, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Per-class metrics saved: {save_path}")


def plot_model_comparison(all_metrics, save_path):
    """Compare all models side by side"""
    models = list(all_metrics.keys())
    metrics_names = ['accuracy', 'top3_accuracy', 'top5_accuracy', 
                    'precision_macro', 'recall_macro', 'f1_macro']
    metrics_labels = ['Top-1 Acc', 'Top-3 Acc', 'Top-5 Acc', 
                     'Precision', 'Recall', 'F1-Score']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    x = np.arange(len(models))
    width = 0.6
    
    for idx, (metric, label) in enumerate(zip(metrics_names, metrics_labels)):
        values = [all_metrics[model][metric] for model in models]
        
        bars = axes[idx].bar(x, values, width, alpha=0.7)
        axes[idx].set_title(label, fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('Score (%)', fontsize=12)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        axes[idx].set_ylim([0, 100])
        axes[idx].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}%',
                          ha='center', va='bottom', fontsize=10)
        
        # Color best performer
        best_idx = np.argmax(values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('orange')
        bars[best_idx].set_linewidth(2)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Model comparison saved: {save_path}")


def generate_latex_table(all_metrics, save_path):
    """Generate LaTeX table for academic paper"""
    models = list(all_metrics.keys())
    
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comprehensive Performance Metrics of All Models}
\label{tab:model_performance}
\begin{tabular}{lcccccc}
\hline
\textbf{Model} & \textbf{Top-1} & \textbf{Top-3} & \textbf{Top-5} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\hline
"""
    
    for model in models:
        metrics = all_metrics[model]
        latex += f"{model} & "
        latex += f"{metrics['accuracy']:.2f}\% & "
        latex += f"{metrics['top3_accuracy']:.2f}\% & "
        latex += f"{metrics['top5_accuracy']:.2f}\% & "
        latex += f"{metrics['precision_macro']:.2f}\% & "
        latex += f"{metrics['recall_macro']:.2f}\% & "
        latex += f"{metrics['f1_macro']:.2f}\% \\\\\n"
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    
    with open(save_path, 'w') as f:
        f.write(latex)
    
    print(f"✓ LaTeX table saved: {save_path}")


def save_detailed_report(all_metrics, class_names, save_dir):
    """Save comprehensive text report"""
    report_path = os.path.join(save_dir, 'detailed_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE EVALUATION REPORT\n")
        f.write("="*80 + "\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Classes: {len(class_names)}\n")
        f.write(f"Number of Models Evaluated: {len(all_metrics)}\n")
        f.write("="*80 + "\n\n")
        
        for model_name, metrics in all_metrics.items():
            f.write("\n" + "="*80 + "\n")
            f.write(f"MODEL: {model_name}\n")
            f.write("="*80 + "\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-"*80 + "\n")
            f.write(f"  Top-1 Accuracy:      {metrics['accuracy']:.2f}%\n")
            f.write(f"  Top-3 Accuracy:      {metrics['top3_accuracy']:.2f}%\n")
            f.write(f"  Top-5 Accuracy:      {metrics['top5_accuracy']:.2f}%\n")
            f.write(f"  Loss:                {metrics['loss']:.4f}\n")
            f.write(f"  Precision (Macro):   {metrics['precision_macro']:.2f}%\n")
            f.write(f"  Recall (Macro):      {metrics['recall_macro']:.2f}%\n")
            f.write(f"  F1-Score (Macro):    {metrics['f1_macro']:.2f}%\n")
            f.write(f"  Precision (Weighted):{metrics['precision_weighted']:.2f}%\n")
            f.write(f"  Recall (Weighted):   {metrics['recall_weighted']:.2f}%\n")
            f.write(f"  F1-Score (Weighted): {metrics['f1_weighted']:.2f}%\n")
            
            f.write("\nPer-Class Performance:\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-"*80 + "\n")
            
            per_class = metrics['per_class']
            for i, class_name in enumerate(class_names):
                f.write(f"{class_name:<30} "
                       f"{per_class['precision'][i]*100:>10.2f}%  "
                       f"{per_class['recall'][i]*100:>10.2f}%  "
                       f"{per_class['f1'][i]*100:>10.2f}%  "
                       f"{int(per_class['support'][i]):>8}\n")
            
            f.write("\n")
    
    print(f"✓ Detailed report saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Ensemble Evaluation')
    parser.add_argument('--data-dir', default='images_split', help='Data directory')
    parser.add_argument('--ensemble-model', default='runs/ensemble/ensemble_finetuned_best.pth',
                       help='Path to ensemble model')
    parser.add_argument('--summary', default='runs/ensemble/training_summary.json',
                       help='Training summary file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num workers')
    parser.add_argument('--output-dir', default='runs/ensemble/evaluation', 
                       help='Output directory')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load classes
    with open(args.summary, 'r') as f:
        summary = json.load(f)
    class_names = summary['classes']
    num_classes = len(class_names)
    
    print(f"Classes: {num_classes}")
    
    # Data loader
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    print(f"Validation samples: {len(val_dataset)}")
    
    # Load ensemble
    print(f"\nLoading ensemble from {args.ensemble_model}...")
    ensemble = StackingEnsemble(num_classes=num_classes, device=device)
    ensemble.load_state_dict(torch.load(args.ensemble_model, map_location=device))
    ensemble.eval()
    ensemble.to(device)
    print("✓ Ensemble loaded")
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    print("\n" + "="*80)
    print("STARTING COMPREHENSIVE EVALUATION")
    print("="*80)
    
    all_metrics = evaluate_ensemble_detailed(
        ensemble, val_loader, criterion, device, num_classes, class_names
    )
    
    # Save metrics as JSON
    metrics_json_path = os.path.join(args.output_dir, 'all_metrics.json')
    metrics_serializable = {}
    for model_name, metrics in all_metrics.items():
        metrics_serializable[model_name] = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in metrics.items()
            if k not in ['confusion_matrix', 'predictions', 'labels', 'probabilities', 'per_class']
        }
        metrics_serializable[model_name]['per_class'] = {
            k: v.tolist() for k, v in metrics['per_class'].items()
        }
    
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=4)
    print(f"\n✓ Metrics JSON saved: {metrics_json_path}")
    
    # Generate plots
    print("\nGenerating visualizations...")
    for model_name, metrics in all_metrics.items():
        # Confusion matrix
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_{model_name}.png')
        plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path, 
                            title=f'Confusion Matrix - {model_name}')
        
        # Per-class metrics
        pcm_path = os.path.join(args.output_dir, f'per_class_metrics_{model_name}.png')
        plot_per_class_metrics(metrics, class_names, pcm_path, title=model_name)
    
    # Model comparison
    comparison_path = os.path.join(args.output_dir, 'model_comparison.png')
    plot_model_comparison(all_metrics, comparison_path)
    
    # LaTeX table
    latex_path = os.path.join(args.output_dir, 'metrics_table.tex')
    generate_latex_table(all_metrics, latex_path)
    
    # Detailed report
    save_detailed_report(all_metrics, class_names, args.output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'F1-Score':<10}")
    print("-"*80)
    for model_name, metrics in all_metrics.items():
        print(f"{model_name:<25} "
              f"{metrics['accuracy']:>8.2f}% "
              f"{metrics['top3_accuracy']:>8.2f}% "
              f"{metrics['top5_accuracy']:>8.2f}% "
              f"{metrics['f1_macro']:>8.2f}%")
    print("="*80)
    
    print(f"\n✓ All results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
