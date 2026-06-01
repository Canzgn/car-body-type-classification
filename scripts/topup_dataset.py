

import os
import shutil
import argparse
import hashlib
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import requests
from io import BytesIO
from ddgs import DDGS


RAW_DIR   = Path("data/raw")
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET    = 800
THRESHOLD = 0.25

DESKTOP = Path(r"C:\Users\Barış\OneDrive\Masaüstü")

CLASS_PROMPTS = {
    "F1": [
        "a formula one racing car on track",
        "an open wheel racing car exterior",
        "a formula 1 car side view",
    ],
    "HATCHBACK": [
        "a hatchback car exterior side view",
        "a 3 door hatchback automobile exterior",
        "a 5 door hatchback car photo",
    ],
    "MICRO": [
        "a microcar city car exterior side view",
        "a small miniature city car exterior",
        "a kei car small automobile exterior",
    ],
    "PICKUP": [
        "a pickup truck exterior side view",
        "a pickup truck with cargo bed",
        "a truck pickup automobile exterior",
    ],
    "SEDAN": [
        "a sedan car exterior side view",
        "a 4 door sedan automobile exterior",
        "a sedan car photo",
    ],
    "STATION_WAGON": [
        "a station wagon estate car exterior",
        "an estate car side view exterior",
        "a touring wagon car exterior",
    ],
    "SUV": [
        "an SUV sport utility vehicle exterior",
        "a sport utility vehicle side view",
        "an SUV car exterior photo",
    ],
    "VAN": [
        "a van vehicle exterior side view",
        "a minivan cargo van exterior",
        "a van automobile exterior photo",
    ],
}

DESKTOP_SOURCES = {
    "F1": [
        DESKTOP / "Formula One Cars",
    ],
    "HATCHBACK": [
        DESKTOP / "Cars_Body_Type" / "train" / "Hatchback",
        DESKTOP / "Cars_Body_Type" / "valid" / "Hatchback",
        DESKTOP / "Cars_Body_Type" / "test"  / "Hatchback",
        DESKTOP / "stanford_cars_type" / "Hatchback",
    ],
    "PICKUP": [
        DESKTOP / "Cars_Body_Type" / "train" / "Pick-Up",
        DESKTOP / "Cars_Body_Type" / "valid" / "Pick-Up",
        DESKTOP / "Cars_Body_Type" / "test"  / "Pick-Up",
    ],
    "SEDAN": [
        DESKTOP / "Cars_Body_Type" / "train" / "Sedan",
        DESKTOP / "Cars_Body_Type" / "valid" / "Sedan",
        DESKTOP / "Cars_Body_Type" / "test"  / "Sedan",
        DESKTOP / "stanford_cars_type" / "Sedan",
    ],
    "STATION_WAGON": [
        DESKTOP / "stanford_cars_type" / "Wagon",
    ],
    "SUV": [
        DESKTOP / "Cars_Body_Type" / "train" / "SUV",
        DESKTOP / "Cars_Body_Type" / "valid" / "SUV",
        DESKTOP / "Cars_Body_Type" / "test"  / "SUV",
        DESKTOP / "stanford_cars_type" / "SUV",
    ],
    "VAN": [
        DESKTOP / "Cars_Body_Type" / "train" / "VAN",
        DESKTOP / "Cars_Body_Type" / "valid" / "VAN",
        DESKTOP / "Cars_Body_Type" / "test"  / "VAN",
        DESKTOP / "stanford_cars_type" / "Van",
    ],
}

DDG_KEYWORDS = {}

MICRO_KEYWORDS = [
    "Seat Mii city car exterior side view photo",
    "Skoda Citigo city car exterior side photo",
    "Kia Picanto city car exterior side view",
    "Hyundai i10 city car exterior side view",
    "Renault Twingo city car exterior side",
    "Fiat 500 city car exterior side view",
    "Smart Fortwo city car exterior side view",
    "Toyota Aygo city car exterior side photo",
    "Volkswagen Up city car exterior side",
    "Peugeot 108 city car exterior side view",
    "Citroen C1 city car exterior side view",
    "Suzuki Alto kei car exterior side view",
    "Daihatsu Mira kei car exterior photo",
    "Honda N-One kei car exterior side view",
]

STATION_WAGON_KEYWORDS = [
    "station wagon estate car exterior side view photo",
    "estate car wagon exterior side view photo",
    "Volvo V60 station wagon exterior side view",
    "Skoda Octavia Combi estate car exterior",
    "Volkswagen Passat estate wagon exterior side",
    "BMW 3 series touring wagon exterior side view",
    "Mercedes C class estate wagon exterior photo",
    "Audi A4 Avant wagon exterior side view",
    "Subaru Outback station wagon exterior side",
    "Ford Mondeo estate wagon exterior side view",
    "Toyota Corolla wagon exterior side view photo",
    "Opel Astra Sports Tourer estate exterior",
    "Peugeot 308 SW estate exterior side view",
    "Renault Megane estate wagon exterior photo",
]

DDG_KEYWORDS["MICRO"] = MICRO_KEYWORDS
DDG_KEYWORDS["STATION_WAGON"] = STATION_WAGON_KEYWORDS

SW_PROMPTS = [
    "a station wagon estate car exterior",
    "an estate car side view exterior",
    "a touring wagon car exterior",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
    )
}
MIN_W, MIN_H = 200, 150



def load_clip():
    print("CLIP modeli yükleniyor (ViT-B/32)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    print(f"  Cihaz: {device}\n")
    return model, preprocess, tokenizer, device


def encode_prompts(prompts, tokenizer, model, device):
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats


def score_images_batch(files, text_features, preprocess, model, device, batch_size=64):
    scores = []
    for i in tqdm(range(0, len(files), batch_size), desc="    Analiz", ncols=72):
        batch_files = files[i:i + batch_size]
        tensors, valid_idx = [], []
        for j, f in enumerate(batch_files):
            try:
                img = Image.open(f).convert("RGB")
                tensors.append(preprocess(img))
                valid_idx.append(j)
            except Exception:
                pass
        if not tensors:
            scores.extend([0.0] * len(batch_files))
            continue
        batch_tensor = torch.stack(tensors).to(device)
        with torch.no_grad():
            img_features = model.encode_image(batch_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            sims = (img_features @ text_features.T)
            batch_scores = sims.max(dim=1).values.cpu().tolist()
        result = [0.0] * len(batch_files)
        for k, idx in enumerate(valid_idx):
            result[idx] = batch_scores[k]
        scores.extend(result)
    return scores



def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_existing_hashes(raw_class_dir: Path) -> set:
    """Mevcut raw klasöründeki tüm görsellerin MD5 hash'lerini döndür."""
    hashes = set()
    for f in raw_class_dir.iterdir():
        if f.suffix.lower() in IMG_EXTS:
            try:
                hashes.add(file_hash(f))
            except Exception:
                pass
    return hashes



def next_filename(raw_class_dir: Path) -> Path:
    """data/raw/<CLASS>/ içinde boş bir 6 haneli dosya adı döndür."""
    existing = {f.stem for f in raw_class_dir.iterdir() if f.suffix.lower() in IMG_EXTS}
    i = 1
    while f"{i:06d}" in existing:
        i += 1
    return raw_class_dir / f"{i:06d}.jpg"


def copy_from_desktop(class_name: str, raw_class_dir: Path,
                       needed: int, existing_hashes: set, dry_run: bool) -> int:
    """Desktop kaynaklarından ihtiyaç kadar görsel kopyalar. Eklenen sayıyı döndürür."""
    sources = DESKTOP_SOURCES.get(class_name, [])
    if not sources:
        return 0

    candidates = []
    for src_dir in sources:
        if src_dir.exists():
            for f in src_dir.rglob("*"):
                if f.suffix.lower() in IMG_EXTS:
                    candidates.append(f)

    if not candidates:
        print(f"    Desktop'ta kaynak bulunamadı: {[str(s) for s in sources]}")
        return 0

    import random
    random.shuffle(candidates)

    added = 0
    for src in candidates:
        if added >= needed:
            break
        try:
            h = file_hash(src)
            if h in existing_hashes:
                continue  # zaten var
            img = Image.open(src)
            w, h_px = img.size
            if w < MIN_W or h_px < MIN_H:
                continue
            if dry_run:
                added += 1
                existing_hashes.add(h)
                continue
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            dest = next_filename(raw_class_dir)
            img.save(dest, "JPEG", quality=90)
            existing_hashes.add(h)
            added += 1
        except Exception:
            continue

    return added


def download_ddg(class_name: str, raw_class_dir: Path,
                  needed: int, existing_hashes: set, dry_run: bool) -> int:
    keywords = DDG_KEYWORDS.get(class_name, MICRO_KEYWORDS)
    if dry_run:
        print(f"    [DRY-RUN] {needed} görsel indirilecek ({class_name})")
        return needed  # kaba tahmin
    added = 0
    for kw in keywords:
        if added >= needed:
            break
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    kw, region="us-en", safesearch="off",
                    type_image="photo", size="Medium",
                    max_results=max(needed * 2, 30),
                ))
            time.sleep(2)
        except Exception as e:
            print(f"    DDG hata: {e}")
            continue

        for item in results:
            if added >= needed:
                break
            url = item.get("image", "")
            if not url:
                continue
            try:
                r = requests.get(url, headers=HEADERS, timeout=10)
                if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
                    continue
                content = r.content
                h = hashlib.md5(content).hexdigest()
                if h in existing_hashes:
                    continue
                img = Image.open(BytesIO(content))
                w, h_px = img.size
                if w < MIN_W or h_px < MIN_H:
                    continue
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                dest = next_filename(raw_class_dir)
                img.save(dest, "JPEG", quality=90)
                existing_hashes.add(h)
                added += 1
            except Exception:
                continue

    return added



def process_class(class_name: str, model, preprocess, tokenizer, device,
                   threshold: float, dry_run: bool):
    raw_class_dir = RAW_DIR / class_name
    raw_class_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in raw_class_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    current = len(files)
    print(f"\n[{class_name}]  mevcut: {current}")

    prompts = CLASS_PROMPTS.get(class_name, SW_PROMPTS)
    text_features = encode_prompts(prompts, tokenizer, model, device)

    print(f"  CLIP filtresi uygulanıyor ({threshold})...")
    scores = score_images_batch(files, text_features, preprocess, model, device)
    low = [(s, f) for s, f in zip(scores, files) if s < threshold]
    low.sort()
    print(f"  Eşik altı: {len(low)} görsel")

    if dry_run:
        print(f"  [DRY-RUN] {len(low)} silinecek. En düşük 5:")
        for s, f in low[:5]:
            print(f"    {s:.3f} → {f.name}")
    else:
        for _, f in low:
            try:
                f.unlink()
            except Exception:
                pass
        if low:
            print(f"  ✗ {len(low)} görsel silindi")

    after_filter = current - len(low)
    needed = TARGET - after_filter
    print(f"  Filtre sonrası: {after_filter}  →  {needed} görsel gerekiyor")

    if needed <= 0:
        print(f"  ✓ {class_name} zaten {TARGET}+ görsele sahip")
        return

    existing_hashes = build_existing_hashes(raw_class_dir)

    if class_name == "MICRO":
        print(f"  DuckDuckGo'dan {needed} görsel indiriliyor...")
        added = download_ddg(class_name, raw_class_dir, needed, existing_hashes, dry_run)
    else:
        print(f"  Desktop'tan {needed} görsel kopyalanıyor...")
        added = copy_from_desktop(class_name, raw_class_dir, needed, existing_hashes, dry_run)
        if added < needed:
            # Desktop yetmedi, geri kalanı DuckDuckGo'dan indir
            still_needed = needed - added
            print(f"  Desktop yetmedi ({added}/{needed}), DDG'den {still_needed} daha...")
            added += download_ddg(class_name, raw_class_dir, still_needed, existing_hashes, dry_run)

    print(f"  ✓ {added} yeni görsel eklendi (hedef: {needed})")

    if not dry_run and added > 0:
        all_files = sorted([f for f in raw_class_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        new_files = all_files[after_filter:]  # filtreden sonra eklenenlerin sırası
        if not new_files:
            return
        print(f"  Yeni {len(new_files)} görsel CLIP ile doğrulanıyor...")
        new_scores = score_images_batch(new_files, text_features, preprocess, model, device)
        bad_new = [(s, f) for s, f in zip(new_scores, new_files) if s < threshold]
        bad_new.sort()
        if bad_new:
            print(f"  Yeni görsellerde {len(bad_new)} kötü görsel bulundu, siliniyor...")
            for _, f in bad_new:
                try:
                    f.unlink()
                except Exception:
                    pass
            print(f"  ✗ {len(bad_new)} yeni görsel kalite testini geçemedi (silindi)")
        else:
            print(f"  ✓ Tüm yeni görseller kalite testini geçti")

    final = len([f for f in raw_class_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    print(f"  → Sonuç: {final} görsel")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Silme/kopyalama yapma, sadece raporla")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    model, preprocess, tokenizer, device = load_clip()

    classes = ["STATION_WAGON", "F1", "HATCHBACK", "MICRO", "PICKUP", "SEDAN", "SUV", "VAN"]

    print(f"Eşik: {args.threshold} | Dry-run: {args.dry_run}")
    print("=" * 60)

    for cls in classes:
        process_class(cls, model, preprocess, tokenizer, device,
                      args.threshold, args.dry_run)

    print("\n" + "=" * 60)
    print("ÖZET:")
    for cls in sorted(RAW_DIR.iterdir(), key=lambda x: x.name):
        if cls.is_dir():
            count = len([f for f in cls.iterdir() if f.suffix.lower() in IMG_EXTS])
            status = "✓" if count >= TARGET else "✗"
            print(f"  {status} {cls.name:<16}: {count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
