"""
EfficientNet-B0 Transfer Learning ile Araba Gövde Tipi Sınıflandırma.

Veri yapısı:
    data/train/<SINIF>/*.jpg
    data/val/<SINIF>/*.jpg

Çıktılar:
    models/best_model.pth          — En iyi model ağırlıkları
    outputs/training_history.json  — Epoch bazlı loss/accuracy

Kullanım:
    python scripts/train.py
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


# ── Konfigürasyon ─────────────────────────────────────────────────────────────
NUM_CLASSES = 8
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS      = 50
LR          = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE    = 7   # Early stopping sabır değeri

DATA_DIR    = Path('data')
MODEL_DIR   = Path('models')
OUTPUT_DIR  = Path('outputs')

# Türkçe görüntüleme etiketleri
CLASS_LABELS = {
    'F1':            'Açık Tekerlekli (F1)',
    'HATCHBACK':     'Hatchback',
    'MICRO':         'Micro',
    'PICKUP':        'Pick-Up',
    'SEDAN':         'Sedan',
    'STATION_WAGON': 'Station Wagon',
    'SUV':           'SUV',
    'VAN':           'Van',
}


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes: int) -> nn.Module:
    """EfficientNet-B0 yükle, son katmanı sınıf sayısına göre değiştir."""
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, desc='  Train', leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct = 0.0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='  Val  ', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
    return running_loss / len(loader.dataset), correct / len(loader.dataset)


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Veri Setleri ──────────────────────────────────────────────────────────
    train_tf, val_tf = build_transforms()

    train_dataset = datasets.ImageFolder(DATA_DIR / 'train', transform=train_tf)
    val_dataset   = datasets.ImageFolder(DATA_DIR / 'val',   transform=val_tf)

    # Windows'ta num_workers>0 sorun çıkarabilir; güvenli değer: 0
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    print(f'Train: {len(train_dataset)} görsel | Val: {len(val_dataset)} görsel')
    print(f'Sınıf eşlemesi: {train_dataset.class_to_idx}')

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5, verbose=True
    )

    # ── Eğitim Döngüsü ────────────────────────────────────────────────────────
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch {epoch}/{EPOCHS}')

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        print(f'  Train  Loss={train_loss:.4f}  Acc={train_acc*100:.2f}%')
        print(f'  Val    Loss={val_loss:.4f}  Acc={val_acc*100:.2f}%')

        scheduler.step(val_loss)

        # Model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'class_to_idx': train_dataset.class_to_idx,
            }, MODEL_DIR / 'best_model.pth')
            print(f'  ✓ Model kaydedildi (val_loss={val_loss:.4f})')
        else:
            epochs_no_improve += 1
            print(f'  Gelişme yok ({epochs_no_improve}/{PATIENCE})')
            if epochs_no_improve >= PATIENCE:
                print(f'\nEarly stopping! {PATIENCE} epoch boyunca iyileşme olmadı.')
                break

    # Eğitim geçmişini kaydet
    history_path = OUTPUT_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    model_size = os.path.getsize(MODEL_DIR / 'best_model.pth') / 1024 / 1024
    print(f'\nEğitim tamamlandı!')
    print(f'Model boyutu : {model_size:.1f} MB (sınır: 95 MB)')
    print(f'Geçmiş       : {history_path}')


if __name__ == '__main__':
    main()
