"""
Train Stacking Ensemble Model
3-Stage Training:
1. Train base models individually
2. Train meta-learner with frozen base models
3. Fine-tune entire ensemble
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
import matplotlib.pyplot as plt
import pandas as pd

from ensemble_model import StackingEnsemble, ResNet50Classifier, EfficientNetB3Classifier, ConvNeXtClassifier
from augmentation_utils import mixup_data, cutmix_data, mixup_criterion, EarlyStopping, GradientClipping


def create_data_loaders(data_dir, batch_size=16, num_workers=4):
    """Create train and validation data loaders with ADVANCED augmentation"""
    
    # 🔥 AGGRESSIVE Data Augmentation - Ezberlemeni önler!
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        
        # Geometrik Transformasyonlar
        transforms.RandomHorizontalFlip(p=0.5),  # Sağ-sol çevirme
        transforms.RandomRotation(25),  # -25 ile +25 derece rotasyon
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  # %10 kaydırma
            scale=(0.9, 1.1),  # %90-110 zoom
            shear=10  # Eğme
        ),
        
        # Renk ve Parlaklık Manipülasyonları
        transforms.ColorJitter(
            brightness=0.3,  # Parlaklık ±30%
            contrast=0.3,  # Kontrast ±30%
            saturation=0.3,  # Doygunluk ±30%
            hue=0.1  # Renk tonu ±10%
        ),
        
        # Rastgele Gri Ton (bazen renkler yanıltıcı olabilir)
        transforms.RandomGrayscale(p=0.1),
        
        # Gaussian Blur (odak dışı fotoğraflar için)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        
        # Rastgele Perspektif Değişimi
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        
        # Rastgele Silme (Random Erasing)
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=0.3,  # %30 ihtimalle
            scale=(0.02, 0.15),  # Silinecek alan %2-15
            ratio=(0.3, 3.3),  # Dikdörtgen oranı
            value='random'  # Rastgele piksel değerleri
        )
    ])
    
    # Validation transform - Augmentation YOK (gerçek performans için)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, train_dataset.classes


def train_single_model(model, train_loader, val_loader, epochs, device, model_name, save_dir):
    """Train a single base model"""
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # 🔥 Anti-Overfitting Arsenal
    use_augmentation = True
    early_stopping = EarlyStopping(patience=7, min_delta=0.001, mode='max')
    gradient_clipper = GradientClipping()
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'train_top3': [], 'train_top5': [],
        'val_loss': [], 'val_acc': [], 'val_top3': [], 'val_top5': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_top3, train_top5 = 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 🔥 ANTI-OVERFITTING: Rastgele Mixup veya CutMix uygula
            if use_augmentation and np.random.rand() > 0.3:  # %70 ihtimalle augmentation
                if np.random.rand() > 0.5:
                    # Mixup
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, alpha=0.4)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    # CutMix
                    inputs, targets_a, targets_b, lam = cutmix_data(inputs, labels, alpha=1.0)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                # Normal training
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 🔥 Gradient Clipping - Exploding gradients önleme
            gradient_clipper.clip_gradients(model, max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Top-3 and Top-5 accuracy
            _, top3_pred = outputs.topk(3, 1, True, True)
            _, top5_pred = outputs.topk(5, 1, True, True)
            
            for i in range(labels.size(0)):
                if labels[i] in top3_pred[i]:
                    train_top3 += 1
                if labels[i] in top5_pred[i]:
                    train_top5 += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}', 
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_top3, val_top5 = 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # Top-1
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Top-3 and Top-5
                _, top3_pred = outputs.topk(3, 1, True, True)
                _, top5_pred = outputs.topk(5, 1, True, True)
                
                for i in range(labels.size(0)):
                    if labels[i] in top3_pred[i]:
                        val_top3 += 1
                    if labels[i] in top5_pred[i]:
                        val_top5 += 1
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        train_top3_acc = 100. * train_top3 / train_total
        train_top5_acc = 100. * train_top5 / train_total
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_top3_acc = 100. * val_top3 / val_total
        val_top5_acc = 100. * val_top5 / val_total
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_top3'].append(train_top3_acc)
        history['train_top5'].append(train_top5_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_top3'].append(val_top3_acc)
        history['val_top5'].append(val_top5_acc)
        history['learning_rate'].append(current_lr)
        
        print(f'\n{model_name} - Epoch {epoch+1}/{epochs}:')
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Top-3={train_top3_acc:.2f}%, Top-5={train_top5_acc:.2f}%')
        print(f'  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%, Top-3={val_top3_acc:.2f}%, Top-5={val_top5_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, f'{model_name}_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f'  ✓ Best model saved: {val_acc:.2f}%')
        
        # 🔥 Early Stopping Check
        if early_stopping(val_acc):
            print(f'  ⚠️  Early stopping triggered! No improvement for {early_stopping.patience} epochs.')
            print(f'  💾 Best Val Acc: {best_acc:.2f}% (stopping to prevent overfitting)')
            break
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f'\n✓ Training history saved: {history_path}')
    
    return model, history, best_acc


def train_meta_learner(ensemble, train_loader, val_loader, epochs, device, save_dir):
    """Train meta-learner with frozen base models"""
    print(f"\n{'='*60}")
    print("Training Meta-Learner (Base models frozen)")
    print(f"{'='*60}")
    
    ensemble.freeze_base_models()
    ensemble = ensemble.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ensemble.meta_learner.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        ensemble.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Meta Epoch {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ensemble(inputs, use_meta=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        # Validation
        ensemble.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs, use_meta=True)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        print(f'Meta Epoch {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, 'ensemble_best.pth')
            torch.save(ensemble.state_dict(), save_path)
            print(f'✓ Best ensemble saved: {val_acc:.2f}%')
    
    return ensemble, history, best_acc


def fine_tune_ensemble(ensemble, train_loader, val_loader, epochs, device, save_dir):
    """Fine-tune entire ensemble"""
    print(f"\n{'='*60}")
    print("Fine-tuning Entire Ensemble")
    print(f"{'='*60}")
    
    ensemble.unfreeze_base_models()
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(ensemble.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Training
        ensemble.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Fine-tune {epoch+1}/{epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = ensemble(inputs, use_meta=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100.*train_correct/train_total:.2f}%'})
        
        scheduler.step()
        
        # Validation
        ensemble.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs, use_meta=True)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f'Fine-tune {epoch+1}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
        
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(save_dir, 'ensemble_finetuned_best.pth')
            torch.save(ensemble.state_dict(), save_path)
            print(f'✓ Best fine-tuned ensemble saved: {val_acc:.2f}%')
    
    return ensemble, history, best_acc


def main():
    parser = argparse.ArgumentParser(description='Train Stacking Ensemble')
    parser.add_argument('--data-dir', default='images_split', help='Data directory')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs-base', type=int, default=15, help='Epochs for base models')
    parser.add_argument('--epochs-meta', type=int, default=10, help='Epochs for meta-learner')
    parser.add_argument('--epochs-fine', type=int, default=5, help='Epochs for fine-tuning')
    parser.add_argument('--num-workers', type=int, default=4, help='Num workers')
    parser.add_argument('--skip-base', action='store_true', help='Skip base model training')
    parser.add_argument('--save-dir', default='runs/ensemble', help='Save directory')
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Data loaders
    train_loader, val_loader, classes = create_data_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )
    num_classes = len(classes)
    print(f"Classes: {num_classes}")
    
    # Stage 1: Train base models
    if not args.skip_base:
        print("\n" + "="*60)
        print("STAGE 1: Training Base Models")
        print("="*60)
        
        # ResNet50
        resnet = ResNet50Classifier(num_classes)
        resnet, hist_res, acc_res = train_single_model(
            resnet, train_loader, val_loader, args.epochs_base, 
            device, 'ResNet50', args.save_dir
        )
        
        # EfficientNetB3
        efficient = EfficientNetB3Classifier(num_classes)
        efficient, hist_eff, acc_eff = train_single_model(
            efficient, train_loader, val_loader, args.epochs_base,
            device, 'EfficientNetB3', args.save_dir
        )
        
        # ConvNeXt
        convnext = ConvNeXtClassifier(num_classes)
        convnext, hist_conv, acc_conv = train_single_model(
            convnext, train_loader, val_loader, args.epochs_base,
            device, 'ConvNeXt', args.save_dir
        )
        
        print(f"\nBase Models Best Accuracies:")
        print(f"  ResNet50: {acc_res:.2f}%")
        print(f"  EfficientNetB3: {acc_eff:.2f}%")
        print(f"  ConvNeXt: {acc_conv:.2f}%")
    
    # Stage 2: Train meta-learner
    print("\n" + "="*60)
    print("STAGE 2: Training Meta-Learner")
    print("="*60)
    
    ensemble = StackingEnsemble(num_classes, device=device)
    
    # Load base models
    ensemble.load_base_models(
        os.path.join(args.save_dir, 'ResNet50_best.pth'),
        os.path.join(args.save_dir, 'EfficientNetB3_best.pth'),
        os.path.join(args.save_dir, 'ConvNeXt_best.pth')
    )
    
    ensemble, hist_meta, acc_meta = train_meta_learner(
        ensemble, train_loader, val_loader, args.epochs_meta, device, args.save_dir
    )
    
    print(f"\nMeta-Learner Best Accuracy: {acc_meta:.2f}%")
    
    # Stage 3: Fine-tune ensemble
    print("\n" + "="*60)
    print("STAGE 3: Fine-tuning Ensemble")
    print("="*60)
    
    ensemble, hist_fine, acc_fine = fine_tune_ensemble(
        ensemble, train_loader, val_loader, args.epochs_fine, device, args.save_dir
    )
    
    print(f"\nFinal Ensemble Accuracy: {acc_fine:.2f}%")
    
    # Save training summary
    summary = {
        'num_classes': num_classes,
        'classes': classes,
        'base_models': {
            'ResNet50': acc_res if not args.skip_base else 'loaded',
            'EfficientNetB3': acc_eff if not args.skip_base else 'loaded',
            'ConvNeXt': acc_conv if not args.skip_base else 'loaded'
        },
        'meta_learner_accuracy': acc_meta,
        'final_ensemble_accuracy': acc_fine,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(args.save_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Models saved in: {args.save_dir}")
    print(f"Final Ensemble Accuracy: {acc_fine:.2f}%")


if __name__ == '__main__':
    main()
