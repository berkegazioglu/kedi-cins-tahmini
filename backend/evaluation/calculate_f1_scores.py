"""
ResNet50 modeli için detaylı performans metrikleri hesaplama
- Precision, Recall, F1-Score (her sınıf için)
- Confusion Matrix
- Macro ve Weighted F1 skorları
"""

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import os

def load_model(model_path, device='cuda'):
    """ResNet-50 modelini yükle"""
    print(f'🔄 Model yükleniyor: {model_path}')
    
    if not os.path.exists(model_path):
        print(f'❌ Model dosyası bulunamadı: {model_path}')
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    num_classes = len(checkpoint['class_names'])
    class_names = checkpoint['class_names']
    
    model = models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    
    # Model checkpoint'inde fc yapısını kontrol et
    if 'fc.1.weight' in checkpoint['model_state_dict']:
        # Dropout + Linear yapısı
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )
    else:
        # Sadece Linear
        model.fc = nn.Linear(num_ftrs, num_classes)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f'✅ Model yüklendi!')
    print(f'📊 Sınıf sayısı: {num_classes}')
    
    return model, class_names

def create_dataloader(data_dir, batch_size=32):
    """Validation dataloader oluştur"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return dataloader, dataset.classes

def evaluate_model(model, dataloader, device='cuda'):
    """Model değerlendirmesi - tüm tahminleri topla"""
    print('\n🔍 Model değerlendiriliyor...')
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Değerlendirme'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def calculate_metrics(y_true, y_pred, class_names):
    """Detaylı metrikler hesapla (manuel implementasyon)"""
    print('\n📊 Metrikler hesaplanıyor...\n')
    
    num_classes = len(class_names)
    
    # Her sınıf için TP, FP, FN hesapla
    class_metrics = []
    
    for i in range(num_classes):
        # True Positives, False Positives, False Negatives
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        tn = np.sum((y_true != i) & (y_pred != i))
        
        # Support (gerçek sınıf örnek sayısı)
        support = np.sum(y_true == i)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics.append({
            'class': class_names[i],
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(support)
        })
    
    # F1 skoruna göre sırala (en iyiden en kötüye)
    class_metrics_sorted = sorted(class_metrics, key=lambda x: x['f1_score'], reverse=True)
    
    # Genel metrikler
    accuracy = np.mean(y_true == y_pred)
    
    # Macro averages (tüm sınıfların ortalaması)
    macro_precision = np.mean([m['precision'] for m in class_metrics])
    macro_recall = np.mean([m['recall'] for m in class_metrics])
    macro_f1 = np.mean([m['f1_score'] for m in class_metrics])
    
    # Weighted averages (support ile ağırlıklı)
    total_support = sum([m['support'] for m in class_metrics])
    weighted_precision = sum([m['precision'] * m['support'] for m in class_metrics]) / total_support
    weighted_recall = sum([m['recall'] * m['support'] for m in class_metrics]) / total_support
    weighted_f1 = sum([m['f1_score'] * m['support'] for m in class_metrics]) / total_support
    
    # Sonuçları yazdır
    print('='*80)
    print('📈 RESNET50 MODEL PERFORMANS METRİKLERİ')
    print('='*80)
    print(f'\n🎯 GENEL PERFORMANS:')
    print(f'  Accuracy:           {accuracy*100:.2f}%')
    print(f'  Macro F1-Score:     {macro_f1*100:.2f}%')
    print(f'  Weighted F1-Score:  {weighted_f1*100:.2f}%')
    print(f'  Macro Precision:    {macro_precision*100:.2f}%')
    print(f'  Macro Recall:       {macro_recall*100:.2f}%')
    
    print(f'\n🏆 EN İYİ 10 SINIF (F1-Score):')
    print('-'*80)
    print(f"{'Sıra':<6} {'Sınıf':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print('-'*80)
    
    for i, metric in enumerate(class_metrics_sorted[:10], 1):
        print(f'{i:<6} {metric["class"]:<30} {metric["precision"]*100:>10.2f}% {metric["recall"]*100:>10.2f}% {metric["f1_score"]*100:>10.2f}% {metric["support"]:>10}')
    
    print(f'\n📉 EN KÖTÜ 10 SINIF (F1-Score):')
    print('-'*80)
    print(f"{'Sıra':<6} {'Sınıf':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print('-'*80)
    
    for i, metric in enumerate(class_metrics_sorted[-10:], 1):
        print(f'{i:<6} {metric["class"]:<30} {metric["precision"]*100:>10.2f}% {metric["recall"]*100:>10.2f}% {metric["f1_score"]*100:>10.2f}% {metric["support"]:>10}')
    
    # JSON olarak kaydet
    results = {
        'overall': {
            'accuracy': float(accuracy),
            'macro_precision': float(macro_precision),
            'macro_recall': float(macro_recall),
            'macro_f1': float(macro_f1),
            'weighted_precision': float(weighted_precision),
            'weighted_recall': float(weighted_recall),
            'weighted_f1': float(weighted_f1)
        },
        'per_class': class_metrics_sorted
    }
    
    with open('resnet50_detailed_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f'\n💾 Detaylı sonuçlar kaydedildi: resnet50_detailed_metrics.json')
    print('='*80)
    
    return results

def main():
    # Device seçimi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🖥️  Device: {device}')
    
    # Model yolu
    model_path = 'runs/resnet50_v2/weights/best.pth'
    
    # Model yükle
    model, class_names = load_model(model_path, device)
    
    if model is None:
        print('❌ Model yüklenemedi, çıkılıyor...')
        return
    
    # Validation dataloader
    val_dir = 'images_split/val'
    if not os.path.exists(val_dir):
        print(f'❌ Validation dizini bulunamadı: {val_dir}')
        return
    
    val_loader, dataset_classes = create_dataloader(val_dir, batch_size=64)
    print(f'📦 Validation set: {len(val_loader.dataset)} görüntü')
    
    # Değerlendirme
    y_pred, y_true = evaluate_model(model, val_loader, device)
    
    # Metrikler
    results = calculate_metrics(y_true, y_pred, class_names)
    
    print('\n✅ Tamamlandı!')

if __name__ == '__main__':
    main()
