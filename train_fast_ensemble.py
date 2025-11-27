"""
Fast Ensemble Training - Optimized for 4GB VRAM
Models: ResNet50 + MobileNetV3 + EfficientNet-B0 + ResNet34
Beklenen Accuracy: 75-80%
Eğitim Süresi: ~6-8 saat
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import json
import time
from pathlib import Path
from tqdm import tqdm
import sys

from augmentation_utils import (
    mixup_data, cutmix_data, mixup_criterion,
    LabelSmoothingCrossEntropy, EarlyStopping, GradientClipping
)


class MobileNetV3Classifier(nn.Module):
    """MobileNetV3-Large Model (~5.4M params)"""
    def __init__(self, num_classes=59, dropout=0.5):
        super().__init__()
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.classifier[0].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class EfficientNetB0Classifier(nn.Module):
    """EfficientNet-B0 Model (~5.3M params)"""
    def __init__(self, num_classes=59, dropout=0.5):
        super().__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.classifier[1].in_features
        
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ResNet34Classifier(nn.Module):
    """ResNet-34 Model (~21.3M params)"""
    def __init__(self, num_classes=59, dropout=0.5):
        super().__init__()
        self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ResNet50Classifier(nn.Module):
    """ResNet-50 Model (~24.6M params)"""
    def __init__(self, num_classes=59, dropout=0.5):
        super().__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.model.fc.in_features
        
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout * 0.5),
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class FastEnsemble(nn.Module):
    """
    Fast 4-Model Ensemble
    Models: ResNet50 + MobileNetV3 + EfficientNet-B0 + ResNet34
    """
    def __init__(self, num_classes=59, dropout=0.5):
        super().__init__()
        self.num_classes = num_classes
        
        self.resnet50 = ResNet50Classifier(num_classes, dropout)
        self.mobilenet = MobileNetV3Classifier(num_classes, dropout)
        self.efficientnet = EfficientNetB0Classifier(num_classes, dropout)
        self.resnet34 = ResNet34Classifier(num_classes, dropout)
        
        # Meta-learner: 4 models * 59 classes = 236 features
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x, use_meta=True):
        pred1 = self.resnet50(x)
        pred2 = self.mobilenet(x)
        pred3 = self.efficientnet(x)
        pred4 = self.resnet34(x)
        
        if use_meta:
            stacked = torch.cat([pred1, pred2, pred3, pred4], dim=1)
            output = self.meta_learner(stacked)
        else:
            output = (pred1 + pred2 + pred3 + pred4) / 4.0
        
        return output
    
    def freeze_base_models(self):
        """Freeze base models for meta-learner training"""
        for model in [self.resnet50, self.mobilenet, self.efficientnet, self.resnet34]:
            for param in model.parameters():
                param.requires_grad = False
    
    def unfreeze_base_models(self):
        """Unfreeze base models for fine-tuning"""
        for model in [self.resnet50, self.mobilenet, self.efficientnet, self.resnet34]:
            for param in model.parameters():
                param.requires_grad = True


def get_transforms(mode='train'):
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_single_model(model, model_name, train_loader, val_loader, device, num_epochs=15, save_dir='runs/fast_ensemble'):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"🚀 Training {model_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=5)
    
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Random augmentation
            if torch.rand(1).item() > 0.5:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
                outputs = model(inputs)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = Path(save_dir) / f'{model_name}_best.pth'
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved! Accuracy: {val_acc:.2f}%")
        
        scheduler.step()
        
        # Early stopping
        if early_stopping(val_loss):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    return model, history, best_acc


def train_ensemble_stage(ensemble, train_loader, val_loader, device, num_epochs, stage_name):
    """Train ensemble (meta-learner or fine-tuning)"""
    print(f"\n{'='*70}")
    print(f"STAGE: {stage_name}")
    print(f"{'='*70}")
    
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(ensemble.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # Training
        ensemble.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = ensemble(inputs, use_meta=True)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        ensemble.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = ensemble(inputs, use_meta=True)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  ✓ Best accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    return ensemble, history, best_acc


def main():
    print("🚀 Fast Ensemble Training - 4 Efficient Models")
    print("Models: ResNet50 + MobileNetV3 + EfficientNet-B0 + ResNet34")
    print("="*70)
    
    # Configuration
    DATA_DIR = 'images_split'
    SAVE_DIR = 'runs/fast_ensemble'
    BATCH_SIZE = 32  # Higher batch size for smaller models
    NUM_WORKERS = 4
    
    EPOCHS_BASE = 15   # Good balance for base models
    EPOCHS_META = 10
    EPOCHS_FINE = 5
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Data loaders
    print("\n📁 Loading datasets...")
    train_dataset = datasets.ImageFolder(f'{DATA_DIR}/train', transform=get_transforms('train'))
    val_dataset = datasets.ImageFolder(f'{DATA_DIR}/val', transform=get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    print(f"Classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create save directory
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    all_histories = {}
    
    print("\n" + "="*70)
    print("STAGE 1: Training Base Models")
    print("="*70)
    
    # Check if ResNet50 exists
    resnet50_path = Path('runs/resnet50_v2/weights/best.pth')
    if resnet50_path.exists():
        print(f"\n{'='*60}")
        print("📦 Loading Existing ResNet50 Model")
        print(f"{'='*60}")
        print(f"Found existing model at {resnet50_path}")
        
        # Load old ResNet50
        resnet_old = models.resnet50(weights=None)
        resnet_old.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(2048, 59))
        
        checkpoint = torch.load(resnet50_path, weights_only=False)
        resnet_old.load_state_dict(checkpoint['model_state_dict'])
        
        # Validate
        resnet_old = resnet_old.to(device)
        resnet_old.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = resnet_old(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        acc_resnet50 = 100. * correct / total
        print(f"✓ ResNet50 Validation Accuracy: {acc_resnet50:.2f}%")
        
        # Use existing model
        resnet50 = ResNet50Classifier(num_classes)
        resnet50.model = resnet_old
        all_histories['resnet50'] = {'note': 'Pre-trained model loaded', 'val_acc': acc_resnet50}
        
        # Save in ensemble format
        torch.save(resnet50.state_dict(), f'{SAVE_DIR}/ResNet50_best.pth')
        print(f"✓ Saved in fast_ensemble format")
    else:
        resnet50 = ResNet50Classifier(num_classes)
        resnet50, hist_resnet50, acc_resnet50 = train_single_model(
            resnet50, "ResNet50", train_loader, val_loader, device,
            num_epochs=EPOCHS_BASE, save_dir=SAVE_DIR
        )
        all_histories['resnet50'] = hist_resnet50
    
    # MobileNetV3
    mobilenet = MobileNetV3Classifier(num_classes)
    mobilenet, hist_mobile, acc_mobile = train_single_model(
        mobilenet, "MobileNetV3", train_loader, val_loader, device,
        num_epochs=EPOCHS_BASE, save_dir=SAVE_DIR
    )
    all_histories['mobilenet'] = hist_mobile
    
    # EfficientNet-B0
    efficientnet = EfficientNetB0Classifier(num_classes)
    efficientnet, hist_effnet, acc_effnet = train_single_model(
        efficientnet, "EfficientNetB0", train_loader, val_loader, device,
        num_epochs=EPOCHS_BASE, save_dir=SAVE_DIR
    )
    all_histories['efficientnet_b0'] = hist_effnet
    
    # ResNet34
    resnet34 = ResNet34Classifier(num_classes)
    resnet34, hist_resnet34, acc_resnet34 = train_single_model(
        resnet34, "ResNet34", train_loader, val_loader, device,
        num_epochs=EPOCHS_BASE, save_dir=SAVE_DIR
    )
    all_histories['resnet34'] = hist_resnet34
    
    stage1_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("STAGE 1 COMPLETE!")
    print(f"Time: {stage1_time/3600:.2f} hours")
    print(f"  ResNet50: {acc_resnet50:.2f}%")
    print(f"  MobileNetV3: {acc_mobile:.2f}%")
    print(f"  EfficientNet-B0: {acc_effnet:.2f}%")
    print(f"  ResNet34: {acc_resnet34:.2f}%")
    print("="*70)
    
    # Stage 2: Meta-learner
    print("\n" + "="*70)
    print("STAGE 2: Training Meta-Learner")
    print("="*70)
    
    ensemble = FastEnsemble(num_classes)
    ensemble.resnet50 = resnet50
    ensemble.mobilenet = mobilenet
    ensemble.efficientnet = efficientnet
    ensemble.resnet34 = resnet34
    
    ensemble = ensemble.to(device)
    ensemble.freeze_base_models()
    
    ensemble, hist_meta, acc_meta = train_ensemble_stage(
        ensemble, train_loader, val_loader, device,
        num_epochs=EPOCHS_META, stage_name="Meta-Learner Training"
    )
    all_histories['meta_learner'] = hist_meta
    
    stage2_time = time.time() - start_time - stage1_time
    
    print("\n" + "="*70)
    print("STAGE 2 COMPLETE!")
    print(f"Time: {stage2_time/3600:.2f} hours")
    print(f"  Meta-Learner Accuracy: {acc_meta:.2f}%")
    print("="*70)
    
    # Stage 3: Fine-tuning
    print("\n" + "="*70)
    print("STAGE 3: Fine-Tuning Entire Ensemble")
    print("="*70)
    
    ensemble.unfreeze_base_models()
    
    ensemble, hist_fine, acc_fine = train_ensemble_stage(
        ensemble, train_loader, val_loader, device,
        num_epochs=EPOCHS_FINE, stage_name="Fine-Tuning"
    )
    all_histories['fine_tuning'] = hist_fine
    
    stage3_time = time.time() - start_time - stage1_time - stage2_time
    total_time = time.time() - start_time
    
    # Save final ensemble
    torch.save(ensemble.state_dict(), f'{SAVE_DIR}/fast_ensemble_final.pth')
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"Total Time: {total_time/3600:.2f} hours")
    print(f"\nStage 1 (Base Models): {stage1_time/3600:.2f} hours")
    print(f"Stage 2 (Meta-Learner): {stage2_time/3600:.2f} hours")
    print(f"Stage 3 (Fine-Tuning): {stage3_time/3600:.2f} hours")
    print(f"\n🏆 Final Ensemble Accuracy: {acc_fine:.2f}%")
    print("="*70)
    
    # Save training history
    history_path = Path(SAVE_DIR) / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(all_histories, f, indent=4)
    print(f"\n✓ Training history saved: {history_path}")
    
    # Save class names
    class_names_path = Path(SAVE_DIR) / 'class_names.json'
    with open(class_names_path, 'w') as f:
        json.dump(train_dataset.classes, f, indent=4)
    print(f"✓ Class names saved: {class_names_path}")
    
    print("\n✅ Ready for evaluation!")


if __name__ == '__main__':
    main()
