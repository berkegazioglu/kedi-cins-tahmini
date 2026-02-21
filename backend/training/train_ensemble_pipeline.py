"""
Training Pipeline for Optimal 3-Model Ensemble
Train meta-learner on top of pre-trained base models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import json
from typing import Dict, Tuple
from tqdm import tqdm
import time

from backend.models.ensemble_model import OptimalEnsembleModel


class EnsembleTrainer:
    """
    Train the optimal 3-model ensemble with meta-learner
    
    Training Strategies:
    1. End-to-end: Train all models together
    2. Freeze base models: Only train meta-learner
    3. Fine-tune: Train meta-learner first, then fine-tune base models
    """
    
    def __init__(
        self,
        model: OptimalEnsembleModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 0.001,
        freeze_base_models: bool = False
    ):
        """
        Initialize trainer
        
        Args:
            model: Ensemble model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: torch device
            learning_rate: Learning rate
            freeze_base_models: If True, only train meta-learner
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.freeze_base_models = freeze_base_models
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        
        if freeze_base_models:
            # Only optimize meta-learner parameters
            self.optimizer = optim.Adam(
                self.model.meta_learner.parameters(),
                lr=learning_rate
            )
        else:
            # Optimize all parameters
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate
            )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        self.best_val_acc = 0.0
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            avg_loss: Average training loss
            accuracy: Training accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs, _ = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """
        Validate model
        
        Returns:
            avg_loss: Average validation loss
            accuracy: Validation accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        avg_loss = running_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int,
        save_dir: str = 'models/ensemble',
        save_best: bool = True
    ) -> Dict:
        """
        Train the ensemble model
        
        Args:
            num_epochs: Number of training epochs
            save_dir: Directory to save models
            save_best: Save best model based on validation accuracy
            
        Returns:
            history: Training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print("=" * 60)
        print("Training Optimal 3-Model Ensemble")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Base models frozen: {self.freeze_base_models}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                best_model_path = save_path / 'optimal_ensemble_best.pth'
                self.save_model(best_model_path)
                print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save final model
        final_model_path = save_path / 'optimal_ensemble_final.pth'
        self.save_model(final_model_path)
        
        # Save history
        history_path = save_path / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Models saved to: {save_path}")
        print("=" * 60)
        
        return self.history
    
    def save_model(self, path: Path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'class_names': self.train_loader.dataset.classes if hasattr(self.train_loader.dataset, 'classes') else None
        }, path)
    
    def load_model(self, path: Path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.history = checkpoint.get('history', self.history)


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Root directory with train/ and val/ subdirectories
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader
    """
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # No augmentation for validation
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f'{data_dir}/train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=f'{data_dir}/val',
        transform=val_transform
    )
    
    # Create data loaders
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
    
    return train_loader, val_loader


def main():
    """Example training script"""
    # Configuration
    data_dir = 'images_split'
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader = create_data_loaders(data_dir, batch_size)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Classes: {len(train_loader.dataset.classes)}")
    
    # Create model
    print("\nCreating ensemble model...")
    model = OptimalEnsembleModel(num_classes=59, freeze_base_models=False)
    
    # Print model info
    info = model.get_model_info()
    print(f"\nModel Information:")
    print(f"  Total parameters: {info['total_parameters']:,}")
    print(f"  Trainable parameters: {info['trainable_parameters']:,}")
    
    # Create trainer
    trainer = EnsembleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        freeze_base_models=False
    )
    
    # Train
    history = trainer.train(num_epochs=num_epochs)
    
    print("\n✅ Training completed!")


if __name__ == "__main__":
    main()
