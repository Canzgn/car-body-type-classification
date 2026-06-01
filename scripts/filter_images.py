
import os
import argparse
import torch
import open_clip
from PIL import Image
from pathlib import Path
import time
import requests
from io import BytesIO
from ddgs import DDGS

RAW_DIR   = Path("data/raw")
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
THRESHOLD = 0.22   
TARGET    = 800    

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

REFILL_KEYWORDS = {
    "F1": [
        "Red Bull Racing RB19 formula one car exterior",
        "Ferrari SF-23 formula one car exterior",
        "Mercedes W14 formula one car side view",
        "McLaren MCL60 formula one car exterior",
    ],
    "HATCHBACK": [
        "Volkswagen Golf Mk7 hatchback exterior side",
        "Ford Focus Mk3 hatchback exterior side",
        "Toyota Corolla hatchback exterior side view",
        "Honda Civic hatchback exterior side view",
    ],
    "MICRO": [
        "Seat Mii city car exterior side view",
        "Skoda Citigo city car exterior side",
        "Kia Picanto city car exterior photo",
        "Hyundai i10 city car exterior photo",
    ],
    "PICKUP": [
        "Toyota Hilux pickup truck exterior side",
        "Ford Ranger pickup truck exterior side",
        "Chevrolet Colorado pickup truck exterior",
        "Nissan Navara pickup truck exterior side",
    ],
    "SEDAN": [
        "Toyota Camry sedan exterior side view",
        "Honda Accord sedan exterior side view",
        "BMW 3 Series sedan exterior side view",
        "Mercedes C-Class sedan exterior side",
    ],
    "STATION_WAGON": [
        "BMW 3 Series Touring wagon exterior",
        "Audi A4 Avant wagon exterior photo",
        "Volvo V60 estate car exterior side",
        "Skoda Octavia Combi estate exterior",
    ],
    "SUV": [
        "Toyota RAV4 SUV exterior side view",
        "Honda CR-V SUV exterior side view",
        "BMW X3 SUV exterior side view photo",
        "Ford Escape SUV exterior side view",
    ],
    "VAN": [
        "Volkswagen Transporter T6 van exterior",
        "Ford Transit van exterior side view",
        "Mercedes Sprinter van exterior side",
        "Renault Trafic van exterior side view",
    ],
}

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
    print(f"  Cihaz: {device}")
    return model, preprocess, tokenizer, device


def score_images_batch(files: list[Path], text_features, preprocess, model,
                        device, batch_size: int = 64) -> list[float]:
    """CLIP ile görsel listesini batch olarak skorlar (GPU hızlı)."""
    from tqdm import tqdm
    scores = []
    for i in tqdm(range(0, len(files), batch_size), desc="  Analiz", ncols=70):
        batch_files = files[i:i + batch_size]
        tensors = []
        valid_idx = []
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
            sims = (img_features @ text_features.T)  # [B, num_prompts]
            batch_scores = sims.max(dim=1).values.cpu().tolist()
        result = [0.0] * len(batch_files)
        for k, idx in enumerate(valid_idx):
            result[idx] = batch_scores[k]
        scores.extend(result)
    return scores


def encode_prompts(prompts: list[str], tokenizer, model, device):
    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats



def download_url(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
            return False
        img = Image.open(BytesIO(r.content))
        w, h = img.size
        if w < MIN_W or h < MIN_H:
            return False
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest, "JPEG", quality=90)
        return True
    except Exception:
        return False


def refill_class(class_name: str, save_dir: Path, needed: int):
    if needed <= 0:
        return
    print(f"  ↺ {needed} yeni görsel indiriliyor...")
    existing = len([f for f in save_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    counter = existing
    keywords = REFILL_KEYWORDS.get(class_name, [])

    for kw in keywords:
        if counter - existing >= needed:
            break
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    kw, region="us-en", safesearch="off",
                    type_image="photo", size="Medium",
                    max_results=needed,
                ))
            time.sleep(2)
        except Exception as e:
            print(f"    DDG hatası: {e}")
            continue

        for item in results:
            if counter - existing >= needed:
                break
            url = item.get("image", "")
            if not url:
                continue
            dest = save_dir / f"{counter:06d}.jpg"
            if download_url(url, dest):
                counter += 1

    added = counter - existing
    print(f"  ✓ {added} yeni görsel eklendi")



def filter_class(class_name: str, model, preprocess, tokenizer, device,
                 threshold: float, dry_run: bool):
    save_dir = RAW_DIR / class_name
    if not save_dir.exists():
        print(f"  [ATLANDI] {save_dir} bulunamadı")
        return

    prompts = CLASS_PROMPTS[class_name]
    text_features = encode_prompts(prompts, tokenizer, model, device)

    files = sorted([f for f in save_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
    total = len(files)
    print(f"  {total} görsel analiz ediliyor...")

    scores = score_images_batch(files, text_features, preprocess, model, device)

    low_score = [(s, f) for s, f in zip(scores, files) if s < threshold]
    low_score.sort()
    print(f"  Eşik altı ({threshold:.2f}): {len(low_score)} görsel")

    if dry_run:
        print("  [DRY-RUN] Silme işlemi yapılmadı. En düşük 5 skor:")
        for score, f in low_score[:5]:
            print(f"    {score:.3f} → {f.name}")
        return

    deleted = 0
    for score, f in low_score:
        try:
            f.unlink()
            deleted += 1
        except Exception:
            pass
    print(f"  ✗ {deleted} görsel silindi")

    if deleted > 0:
        refill_class(class_name, save_dir, deleted)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Sadece raporla, silme")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help=f"CLIP benzerlik eşiği (varsayılan: {THRESHOLD})")
    parser.add_argument("--classes", nargs="+", default=list(CLASS_PROMPTS.keys()),
                        help="Sadece belirtilen sınıfları işle")
    args = parser.parse_args()

    model, preprocess, tokenizer, device = load_clip()

    print(f"\nEşik: {args.threshold} | Dry-run: {args.dry_run}")
    print("=" * 60)

    for cls in args.classes:
        if cls not in CLASS_PROMPTS:
            print(f"[{cls}] Bilinmeyen sınıf, atlanıyor")
            continue
        print(f"\n[{cls}]")
        filter_class(cls, model, preprocess, tokenizer, device,
                     args.threshold, args.dry_run)

    print("\n" + "=" * 60)
    print("ÖZET:")
    for cls in args.classes:
        d = RAW_DIR / cls
        c = len([f for f in d.iterdir() if f.suffix.lower() in IMG_EXTS]) if d.exists() else 0
        print(f"  {cls:15s}: {c} görsel")
    print("=" * 60)


if __name__ == "__main__":
    main()
