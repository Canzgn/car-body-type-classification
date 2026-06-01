

import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


NUM_CLASSES = 8
IMG_SIZE    = 224
BATCH_SIZE  = 32

DATA_DIR   = Path('data')
MODEL_DIR  = Path('models')
OUTPUT_DIR = Path('outputs')

CLASS_NAMES = ['F1', 'HATCHBACK', 'MICRO', 'PICKUP', 'SEDAN', 'STATION_WAGON', 'SUV', 'VAN']
CLASS_LABELS = {
    'F1':            'Açık Tekerlekli',
    'HATCHBACK':     'Hatchback',
    'MICRO':         'Micro',
    'PICKUP':        'Pick-Up',
    'SEDAN':         'Sedan',
    'STATION_WAGON': 'Station Wagon',
    'SUV':           'SUV',
    'VAN':           'Van',
}
DISPLAY_NAMES = [CLASS_LABELS[c] for c in CLASS_NAMES]


def load_model(model_path: Path, device: torch.device) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Model yüklendi — Epoch: {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']*100:.2f}%")
    return model


def get_predictions(model, loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Değerlendirme'):
            outputs = model(images.to(device))
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_training_curves(history: dict, save_path: Path):
    epochs = range(1, len(history['train_loss']) + 1)
    save_dir = save_path.parent

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(epochs, history['train_loss'], color='#e74c3c', linewidth=2, label='Training Loss')
    ax.plot(epochs, history['val_loss'],   color='#3498db', linewidth=2, label='Validation Loss')
    ax.set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(epochs, [a * 100 for a in history['train_acc']], color='#e74c3c', linewidth=2, label='Training Accuracy')
    ax.plot(epochs, [a * 100 for a in history['val_acc']],   color='#3498db', linewidth=2, label='Validation Accuracy')
    ax.set_title('Training & Validation Accuracy', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'training_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Grafikler kaydedildi: training_loss.png, training_accuracy.png')


def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(5, 4.2))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=DISPLAY_NAMES,
        yticklabels=DISPLAY_NAMES,
        linewidths=0.5,
        vmin=0,
        vmax=1,
        annot_kws={"size": 7},
    )
    plt.title('Normalized Confusion Matrix', fontsize=10, fontweight='bold', pad=10)
    plt.ylabel('Gerçek Sınıf', fontsize=9)
    plt.xlabel('Tahmin Edilen Sınıf', fontsize=9)
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Confusion matrix kaydedildi: {save_path}')


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_path = MODEL_DIR / 'best_model.pth'
    if not model_path.exists():
        print(f'HATA: {model_path} bulunamadı. Önce train.py çalıştırın.')
        return

    model = load_model(model_path, device)

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_dir = DATA_DIR / 'test' if (DATA_DIR / 'test').exists() else DATA_DIR / 'val'
    eval_label = 'TEST' if eval_dir.name == 'test' else 'VAL'
    print(f'Değerlendirme seti: {eval_dir} ({eval_label})')

    eval_dataset = datasets.ImageFolder(eval_dir, transform=val_transform)
    eval_loader  = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f'Sınıf eşlemesi: {eval_dataset.class_to_idx}')

    y_true, y_pred = get_predictions(model, eval_loader, device)

    report = classification_report(y_true, y_pred, target_names=DISPLAY_NAMES, digits=4)
    print('\n=== Sınıflandırma Raporu ===')
    print(report)

    report_path = OUTPUT_DIR / 'metrics_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'Rapor kaydedildi: {report_path}')

    plot_confusion_matrix(y_true, y_pred, OUTPUT_DIR / 'confusion_matrix.png')

    history_path = OUTPUT_DIR / 'training_history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, OUTPUT_DIR / 'training_curves.png')
    else:
        print(f'[ATLA] training_history.json bulunamadı, eğitim grafikleri üretilmedi.')

    print(f'\nTüm çıktılar: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
