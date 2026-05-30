
import os
import shutil
import random
import argparse
from pathlib import Path

CLASSES = ['SUV', 'VAN', 'SEDAN', 'HATCHBACK', 'PICKUP', 'STATION_WAGON', 'MICRO', 'F1']
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

KAGGLE_NAME_MAP = {
    'suv': 'SUV',
    'van': 'VAN',
    'sedan': 'SEDAN',
    'hatchback': 'HATCHBACK',
    'pick-up': 'PICKUP',
    'pickup': 'PICKUP',
    'pick_up': 'PICKUP',
    'station wagon': 'STATION_WAGON',
    'station_wagon': 'STATION_WAGON',
    'stationwagon': 'STATION_WAGON',
    'wagon': 'STATION_WAGON',
    'micro': 'MICRO',
    'microcar': 'MICRO',
    'city car': 'MICRO',
    'f1': 'F1',
    'formula 1': 'F1',
    'open wheel': 'F1',
}


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    collected: dict[str, list[Path]] = {cls: [] for cls in CLASSES}

    for item in source_dir.rglob('*'):
        if item.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        folder = item.parent.name.lower().strip()
        mapped = KAGGLE_NAME_MAP.get(folder)
        if mapped:
            collected[mapped].append(item)

    return collected


def split_dataset(source_dir: Path, val_ratio: float = 0.15, test_ratio: float = 0.15, seed: int = 42):
    random.seed(seed)
    train_dir = source_dir.parent / 'train'
    val_dir = source_dir.parent / 'val'
    test_dir = source_dir.parent / 'test'

    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    collected = collect_images(source_dir)

    print("\n=== Veri Seti Bölme Başlıyor ===\n")
    total_train, total_val, total_test = 0, 0, 0

    for cls in CLASSES:
        images = collected[cls]
        if not images:
            print(f"  [ATLANDI]  {cls:20s} — görsel bulunamadı")
            continue

        random.shuffle(images)
        n_test = max(1, int(len(images) * test_ratio))
        n_val  = max(1, int(len(images) * val_ratio))
        test_imgs  = images[:n_test]
        val_imgs   = images[n_test:n_test + n_val]
        train_imgs = images[n_test + n_val:]

        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir   / cls).mkdir(parents=True, exist_ok=True)
        (test_dir  / cls).mkdir(parents=True, exist_ok=True)

        for img in train_imgs:
            shutil.copy2(img, train_dir / cls / img.name)
        for img in val_imgs:
            shutil.copy2(img, val_dir / cls / img.name)
        for img in test_imgs:
            shutil.copy2(img, test_dir / cls / img.name)

        total_train += len(train_imgs)
        total_val   += len(val_imgs)
        total_test  += len(test_imgs)
        print(f"  [OK]  {cls:20s}  train={len(train_imgs):4d}  val={len(val_imgs):4d}  test={len(test_imgs):4d}  toplam={len(images):4d}")

    print(f"\n=== Özet ===")
    print(f"  Train toplam : {total_train}")
    print(f"  Val toplam   : {total_val}")
    print(f"  Test toplam  : {total_test}")
    print(f"  Genel toplam : {total_train + total_val + total_test}")
    print(f"\n  Train → {train_dir}")
    print(f"  Val   → {val_dir}")
    print(f"  Test  → {test_dir}")

    missing = [cls for cls in CLASSES if not collected[cls]]
    if missing:
        print(f"\n  [UYARI] Eksik sınıflar: {', '.join(missing)}")
        print("  Bu sınıflar için ek görsel toplamanız gerekiyor.\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Veri setini train/val/test olarak böler.')
    parser.add_argument('--source',     type=str,   default='data/raw', help='Ham veri klasörü')
    parser.add_argument('--val_ratio',  type=float, default=0.15,       help='Validasyon oranı (0–1)')
    parser.add_argument('--test_ratio', type=float, default=0.15,       help='Test oranı (0–1)')
    parser.add_argument('--seed',       type=int,   default=42,         help='Rastgelelik için seed')
    args = parser.parse_args()

    split_dataset(Path(args.source), args.val_ratio, args.test_ratio, args.seed)
