"""
ResNet50 Eğitiminde Kullanılan Augmentation Örneklerini Gösterir
- Random resizing & cropping
- Color jitter (renk değişimleri)
- Random horizontal flip
- Random rotation
- Mixup (iki görüntü karıştırma)
- CutMix (görüntü kesip yapıştırma)
"""

import torch
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Augmentation transform'ları
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomRotation(degrees=15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Denormalization için
def denormalize(tensor):
    """Normalize edilmiş tensörü geri çevir"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def mixup_images(img1, img2, alpha=0.4):
    """İki görüntüyü mixup ile karıştır"""
    lam = np.random.beta(alpha, alpha)
    mixed = lam * img1 + (1 - lam) * img2
    return mixed, lam

def cutmix_images(img1, img2, alpha=1.0):
    """CutMix: Bir görüntüden dikdörtgen kesip diğerine yapıştır"""
    lam = np.random.beta(alpha, alpha)
    
    _, H, W = img1.shape
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    result = img1.clone()
    result[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
    
    return result, lam

def show_augmentation_examples():
    """Augmentation örneklerini görselleştir ve kaydet"""
    
    # Dataset yükle
    data_dir = 'images_split/train'
    if not os.path.exists(data_dir):
        print(f"❌ Veri dizini bulunamadı: {data_dir}")
        return
    
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    
    # Rastgele 4 görüntü seç
    indices = np.random.choice(len(dataset), 4, replace=False)
    
    # 3x4 grid oluştur
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('ResNet50 Eğitimi - Data Augmentation Örnekleri', fontsize=16, fontweight='bold')
    
    # Her görüntü için augmentation örnekleri
    for col, idx in enumerate(indices):
        img_tensor, label = dataset[idx]
        class_name = dataset.classes[label]
        
        # 1. Satır: Orijinal augmented görüntü
        img_denorm = denormalize(img_tensor)
        img_np = img_denorm.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)
        
        axes[0, col].imshow(img_np)
        axes[0, col].set_title(f'{class_name}\n(Random Crop + Flip + Color)', fontsize=9)
        axes[0, col].axis('off')
        
        # 2. Satır: Mixup örneği
        idx2 = np.random.choice(len(dataset))
        img2_tensor, label2 = dataset[idx2]
        
        mixed_img, lam = mixup_images(img_tensor, img2_tensor)
        mixed_denorm = denormalize(mixed_img)
        mixed_np = mixed_denorm.permute(1, 2, 0).numpy()
        mixed_np = np.clip(mixed_np, 0, 1)
        
        axes[1, col].imshow(mixed_np)
        axes[1, col].set_title(f'Mixup (λ={lam:.2f})\n{class_name} + {dataset.classes[label2]}', fontsize=9)
        axes[1, col].axis('off')
        
        # 3. Satır: CutMix örneği
        idx3 = np.random.choice(len(dataset))
        img3_tensor, label3 = dataset[idx3]
        
        cutmixed_img, lam = cutmix_images(img_tensor, img3_tensor)
        cutmixed_denorm = denormalize(cutmixed_img)
        cutmixed_np = cutmixed_denorm.permute(1, 2, 0).numpy()
        cutmixed_np = np.clip(cutmixed_np, 0, 1)
        
        axes[2, col].imshow(cutmixed_np)
        axes[2, col].set_title(f'CutMix (λ={lam:.2f})\n{class_name} + {dataset.classes[label3]}', fontsize=9)
        axes[2, col].axis('off')
    
    plt.tight_layout()
    
    # Kaydet
    output_path = 'augmentation_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Augmentation örnekleri kaydedildi: {output_path}")
    
    plt.show()

def show_color_jitter_examples():
    """Sadece Color Jitter örneklerini göster"""
    
    data_dir = 'images_split/train'
    if not os.path.exists(data_dir):
        print(f"❌ Veri dizini bulunamadı: {data_dir}")
        return
    
    # Orijinal görüntü için transform (augmentation yok)
    original_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Farklı Color Jitter seviyeleri
    color_transforms = [
        ('Orijinal', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])),
        ('Brightness +30%', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor()
        ])),
        ('Contrast +30%', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(contrast=0.3),
            transforms.ToTensor()
        ])),
        ('Saturation +30%', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(saturation=0.3),
            transforms.ToTensor()
        ])),
        ('Hue ±0.1', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(hue=0.1),
            transforms.ToTensor()
        ])),
        ('Hepsi Birlikte', transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor()
        ]))
    ]
    
    # Rastgele bir görüntü seç
    dataset = datasets.ImageFolder(data_dir)
    idx = np.random.choice(len(dataset))
    img_path, label = dataset.samples[idx]
    original_img = Image.open(img_path).convert('RGB')
    class_name = dataset.classes[label]
    
    # 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Color Jitter Örnekleri - {class_name}', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for i, (title, transform) in enumerate(color_transforms):
        img_tensor = transform(original_img)
        img_np = img_tensor.permute(1, 2, 0).numpy()
        
        axes[i].imshow(img_np)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    output_path = 'color_jitter_examples.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Color jitter örnekleri kaydedildi: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    print("🎨 ResNet50 Data Augmentation Örnekleri\n")
    print("=" * 60)
    
    print("\n1️⃣  Genel augmentation örnekleri oluşturuluyor...")
    show_augmentation_examples()
    
    print("\n2️⃣  Color jitter örnekleri oluşturuluyor...")
    show_color_jitter_examples()
    
    print("\n" + "=" * 60)
    print("✅ Tamamlandı!")
    print("\nOluşturulan dosyalar:")
    print("  - augmentation_examples.png (Mixup, CutMix, genel augmentation)")
    print("  - color_jitter_examples.png (Renk manipülasyonları)")
