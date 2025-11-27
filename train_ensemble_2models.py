"""
Train Ensemble with Only 2 Models (ResNet50 + EfficientNetB3)
Faster training, still good performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
import argparse

from ensemble_model import StackingEnsemble, ResNet50Classifier, EfficientNetB3Classifier


def create_data_loaders(data_dir, batch_size=16, num_workers=4):
    """Create train and validation data loaders"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.classes


class TwoModelEnsemble(nn.Module):
    """Ensemble with only ResNet50 and EfficientNetB3"""
    def __init__(self, num_classes, device='cuda'):
        super().__init__()
        self.resnet = ResNet50Classifier(num_classes)
        self.efficient = EfficientNetB3Classifier(num_classes)
        
        # Meta-learner: 118 features (59+59) -> 59 classes
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        self.device = device
    
    def forward(self, x):
        # Get predictions from both models
        resnet_out = self.resnet(x)
        efficient_out = self.efficient(x)
        
        # Concatenate
        combined = torch.cat([resnet_out, efficient_out], dim=1)
        
        # Meta-learner
        return self.meta_learner(combined)
    
    def freeze_base_models(self):
        """Freeze ResNet and EfficientNet"""
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.efficient.parameters():
            param.requires_grad = False
    
    def unfreeze_base_models(self):
        """Unfreeze all models"""
        for param in self.parameters():
            param.requires_grad = True
    
    def load_base_models(self, resnet_path, efficient_path):
        """Load pretrained base models"""
        print(f"Loading ResNet50 from: {resnet_path}")
        self.resnet.load_state_dict(torch.load(resnet_path))
        
        print(f"Loading EfficientNetB3 from: {efficient_path}")
        self.efficient.load_state_dict(torch.load(efficient_path))


def train_meta_learner(ensemble, train_loader, val_loader, epochs, device, save_dir):
    """Train meta-learner with frozen base models"""
    print("\n" + "="*60)
    print("Training Meta-Learner (2 Models)")
    print("="*60)
    
    ensemble.freeze_base_models()
    ensemble = ensemble.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble.meta_learner.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_top3': [], 'train_top5': [],
        'val_loss': [], 'val_acc': [], 'val_top3': [], 'val_top5': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        # Training
        ensemble.train()
        train_loss, train_correct, train_top3, train_top5, train_total = 0, 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Meta Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ensemble(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Top-1
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Top-3
            _, top3_pred = outputs.topk(3, 1, True, True)
            train_top3 += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()
            
            # Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            train_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        # Validation
        ensemble.eval()
        val_loss, val_correct, val_top3, val_top5, val_total = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                _, top3_pred = outputs.topk(3, 1, True, True)
                val_top3 += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()
                
                _, top5_pred = outputs.topk(5, 1, True, True)
                val_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        # Metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_top3_acc = 100. * train_top3 / train_total
        train_top5_acc = 100. * train_top5 / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_top3_acc = 100. * val_top3 / val_total
        val_top5_acc = 100. * val_top5 / val_total
        lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top3'].append(train_top3_acc)
        history['train_top5'].append(train_top5_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3'].append(val_top3_acc)
        history['val_top5'].append(val_top5_acc)
        history['learning_rate'].append(lr)
        
        print(f'\nMeta Epoch {epoch+1}/{epochs}:')
        print(f'Train: Acc={train_acc:.2f}%, Top-3={train_top3_acc:.2f}%, Top-5={train_top5_acc:.2f}%')
        print(f'Val:   Acc={val_acc:.2f}%, Top-3={val_top3_acc:.2f}%, Top-5={val_top5_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ensemble.state_dict(), os.path.join(save_dir, 'ensemble_2models_best.pth'))
            print(f'✓ Best ensemble saved: {val_acc:.2f}%')
        
        scheduler.step()
    
    # Save history
    with open(os.path.join(save_dir, 'meta_learner_2models_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return ensemble, history, best_acc


def fine_tune_ensemble(ensemble, train_loader, val_loader, epochs, device, save_dir):
    """Fine-tune entire ensemble"""
    print("\n" + "="*60)
    print("Fine-tuning 2-Model Ensemble")
    print("="*60)
    
    ensemble.unfreeze_base_models()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(ensemble.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_top3': [], 'train_top5': [],
        'val_loss': [], 'val_acc': [], 'val_top3': [], 'val_top5': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        # Training
        ensemble.train()
        train_loss, train_correct, train_top3, train_top5, train_total = 0, 0, 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Fine-tune {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ensemble(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            _, top3_pred = outputs.topk(3, 1, True, True)
            train_top3 += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()
            
            _, top5_pred = outputs.topk(5, 1, True, True)
            train_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation
        ensemble.eval()
        val_loss, val_correct, val_top3, val_top5, val_total = 0, 0, 0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                _, top3_pred = outputs.topk(3, 1, True, True)
                val_top3 += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()
                
                _, top5_pred = outputs.topk(5, 1, True, True)
                val_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        # Metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_top3_acc = 100. * train_top3 / train_total
        train_top5_acc = 100. * train_top5 / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_top3_acc = 100. * val_top3 / val_total
        val_top5_acc = 100. * val_top5 / val_total
        lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top3'].append(train_top3_acc)
        history['train_top5'].append(train_top5_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3'].append(val_top3_acc)
        history['val_top5'].append(val_top5_acc)
        history['learning_rate'].append(lr)
        
        print(f'\nFine-tune {epoch+1}/{epochs}:')
        print(f'Train: Acc={train_acc:.2f}%, Top-3={train_top3_acc:.2f}%, Top-5={train_top5_acc:.2f}%')
        print(f'Val:   Acc={val_acc:.2f}%, Top-3={val_top3_acc:.2f}%, Top-5={val_top5_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(ensemble.state_dict(), os.path.join(save_dir, 'ensemble_2models_final.pth'))
            print(f'✓ Final ensemble saved: {val_acc:.2f}%')
        
        scheduler.step()
    
    # Save history
    with open(os.path.join(save_dir, 'ensemble_2models_final_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    return ensemble, history, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train 2-Model Ensemble (ResNet50 + EfficientNetB3)')
    parser.add_argument('--data-dir', default='images_split', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs-meta', type=int, default=2, help='Epochs for meta-learner')
    parser.add_argument('--epochs-fine', type=int, default=1, help='Epochs for fine-tuning')
    parser.add_argument('--num-workers', type=int, default=4, help='Num workers')
    parser.add_argument('--save-dir', default='runs/ensemble', help='Save directory')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data loaders
    train_loader, val_loader, classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    
    # Create ensemble
    ensemble = TwoModelEnsemble(num_classes, device=device)
    
    # Load existing base models
    resnet_path = os.path.join(args.save_dir, 'ResNet50_best.pth')
    efficient_path = os.path.join(args.save_dir, 'EfficientNetB3_best.pth')
    
    if not os.path.exists(resnet_path) or not os.path.exists(efficient_path):
        print("ERROR: Base models not found!")
        print(f"Required: {resnet_path}")
        print(f"Required: {efficient_path}")
        return
    
    ensemble.load_base_models(resnet_path, efficient_path)
    print("✓ Base models loaded successfully!")
    
    # Stage 1: Train meta-learner
    ensemble, hist_meta, acc_meta = train_meta_learner(
        ensemble, train_loader, val_loader, args.epochs_meta, device, args.save_dir
    )
    print(f"\nMeta-Learner Best Accuracy: {acc_meta:.2f}%")
    
    # Stage 2: Fine-tune ensemble
    ensemble, hist_fine, acc_fine = fine_tune_ensemble(
        ensemble, train_loader, val_loader, args.epochs_fine, device, args.save_dir
    )
    print(f"\nFinal 2-Model Ensemble Accuracy: {acc_fine:.2f}%")
    
    # Save summary
    summary = {
        'num_classes': num_classes,
        'classes': classes,
        'models': ['ResNet50', 'EfficientNetB3'],
        'meta_learner_accuracy': acc_meta,
        'final_ensemble_accuracy': acc_fine,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(args.save_dir, 'training_summary_2models.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("2-MODEL ENSEMBLE TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Accuracy: {acc_fine:.2f}%")
    print(f"Models: ResNet50 + EfficientNetB3")


if __name__ == '__main__':
    main()
