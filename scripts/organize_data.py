

import os
import shutil
import random
from pathlib import Path


CARS_BODY_TYPE = Path(r"C:\Users\Barış\OneDrive\Masaüstü\Cars_Body_Type")
STANFORD       = Path(r"C:\Users\Barış\OneDrive\Masaüstü\stanford_cars_type")
F1_SOURCE      = Path(r"C:\Users\Barış\OneDrive\Masaüstü\Formula One Cars")
RAW_DIR        = Path("data/raw")

MAX_PER_CLASS  = 800

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def copy_images(src_dir: Path, dest_dir: Path, label: str, existing_count: int, limit: int) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = [f for f in src_dir.rglob("*") if f.suffix.lower() in IMG_EXTS and f.is_file()]
    random.shuffle(files)

    counter = existing_count
    copied  = 0
    for f in files:
        if counter >= limit:
            break
        dest = dest_dir / f"{counter:06d}{f.suffix.lower()}"
        shutil.copy2(f, dest)
        counter += 1
        copied  += 1

    print(f"  [{label}] {copied} görsel kopyalandı → {dest_dir} (toplam {counter})")
    return counter


def count_existing(dest_dir: Path) -> int:
    if not dest_dir.exists():
        return 0
    return len([f for f in dest_dir.iterdir() if f.suffix.lower() in IMG_EXTS])


def main():
    random.seed(42)

    print("=" * 60)
    print("Veri organizasyonu başlıyor...")
    print(f"Hedef: {RAW_DIR.resolve()}")
    print(f"Sınıf başı maksimum: {MAX_PER_CLASS}")
    print("=" * 60)

    print("\n[SUV]")
    dest = RAW_DIR / "SUV"
    n = count_existing(dest)
    print(f"  Mevcut: {n}")
    if n < MAX_PER_CLASS:
        for split in ["train", "valid", "test"]:
            src = CARS_BODY_TYPE / split / "SUV"
            if src.exists() and n < MAX_PER_CLASS:
                n = copy_images(src, dest, f"Cars_Body_Type/{split}", n, MAX_PER_CLASS)
        src = STANFORD / "SUV"
        if src.exists() and n < MAX_PER_CLASS:
            n = copy_images(src, dest, "Stanford/SUV", n, MAX_PER_CLASS)
    print(f"  → TOPLAM SUV: {count_existing(dest)}")

    print("\n[SEDAN]")
    dest = RAW_DIR / "SEDAN"
    n = count_existing(dest)
    print(f"  Mevcut: {n}")
    if n < MAX_PER_CLASS:
        for split in ["train", "valid", "test"]:
            src = CARS_BODY_TYPE / split / "Sedan"
            if src.exists() and n < MAX_PER_CLASS:
                n = copy_images(src, dest, f"Cars_Body_Type/{split}", n, MAX_PER_CLASS)
        src = STANFORD / "Sedan"
        if src.exists() and n < MAX_PER_CLASS:
            n = copy_images(src, dest, "Stanford/Sedan", n, MAX_PER_CLASS)
    print(f"  → TOPLAM SEDAN: {count_existing(dest)}")

    print("\n[HATCHBACK]")
    dest = RAW_DIR / "HATCHBACK"
    n = count_existing(dest)
    print(f"  Mevcut: {n}")
    if n < MAX_PER_CLASS:
        for split in ["train", "valid", "test"]:
            src = CARS_BODY_TYPE / split / "Hatchback"
            if src.exists() and n < MAX_PER_CLASS:
                n = copy_images(src, dest, f"Cars_Body_Type/{split}", n, MAX_PER_CLASS)
        src = STANFORD / "Hatchback"
        if src.exists() and n < MAX_PER_CLASS:
            n = copy_images(src, dest, "Stanford/Hatchback", n, MAX_PER_CLASS)
    print(f"  → TOPLAM HATCHBACK: {count_existing(dest)}")

    print("\n[PICKUP]")
    dest = RAW_DIR / "PICKUP"
    n = count_existing(dest)
    print(f"  Mevcut: {n}")
    if n < MAX_PER_CLASS:
        for split in ["train", "valid", "test"]:
            for folder_name in ["Pick-Up", "Pickup", "pick-up"]:
                src = CARS_BODY_TYPE / split / folder_name
                if src.exists() and n < MAX_PER_CLASS:
                    n = copy_images(src, dest, f"Cars_Body_Type/{split}", n, MAX_PER_CLASS)
                    break
    print(f"  → TOPLAM PICKUP: {count_existing(dest)}")

    print("\n[VAN]")
    dest = RAW_DIR / "VAN"
    n = count_existing(dest)
    print(f"  Mevcut: {n}")
    if n < MAX_PER_CLASS:
        for split in ["train", "valid", "test"]:
            src = CARS_BODY_TYPE / split / "VAN"
            if src.exists() and n < MAX_PER_CLASS:
                n = copy_images(src, dest, f"Cars_Body_Type/{split}", n, MAX_PER_CLASS)
        src = STANFORD / "Van"
        if src.exists() and n < MAX_PER_CLASS:
            n = copy_images(src, dest, "Stanford/Van", n, MAX_PER_CLASS)
    print(f"  → TOPLAM VAN: {count_existing(dest)}")

    print("\n[STATION_WAGON]")
    dest = RAW_DIR / "STATION_WAGON"
    n = count_existing(dest)
    print(f"  Mevcut (crawled): {n}")
    src = STANFORD / "Wagon"
    if src.exists() and n < MAX_PER_CLASS:
        n = copy_images(src, dest, "Stanford/Wagon", n, MAX_PER_CLASS)
    print(f"  → TOPLAM STATION_WAGON: {count_existing(dest)}")

    print("\n[F1]")
    dest = RAW_DIR / "F1"
    n = count_existing(dest)
    print(f"  Mevcut: {n}")
    if n < MAX_PER_CLASS and F1_SOURCE.exists():
        for team_dir in sorted(F1_SOURCE.iterdir()):
            if team_dir.is_dir() and n < MAX_PER_CLASS:
                n = copy_images(team_dir, dest, team_dir.name, n, MAX_PER_CLASS)
    print(f"  → TOPLAM F1: {count_existing(dest)}")

    micro_count = count_existing(RAW_DIR / "MICRO")
    print(f"\n[MICRO] Mevcut: {micro_count} (değiştirilmedi)")

    print("\n" + "=" * 60)
    print("ÖZET:")
    total = 0
    for cls in ["F1", "HATCHBACK", "MICRO", "PICKUP", "SEDAN", "STATION_WAGON", "SUV", "VAN"]:
        c = count_existing(RAW_DIR / cls)
        total += c
        status = "✓" if c >= 500 else "⚠ AZ"
        print(f"  {cls:15s}: {c:4d} görsel  {status}")
    print(f"  {'TOPLAM':15s}: {total:4d}")
    print("=" * 60)
    print("\nSONRAKI ADIM: python scripts/split_dataset.py")


if __name__ == "__main__":
    main()
