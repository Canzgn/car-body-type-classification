
import os
import time
import requests
from io import BytesIO
from PIL import Image
from ddgs import DDGS

BASE_DIR = "data/raw"


MICRO_KEYWORDS = [
    "Seat Mii city car exterior side view",
    "Skoda Citigo city car exterior",
    "Kia Picanto 2013 city car exterior",
    "Hyundai i10 2012 city car exterior",
    "Peugeot 107 city car exterior side",
    "Toyota Aygo 2010 city car exterior",
    "Mitsubishi Mirage city car exterior",
    "Mazda Carol kei car exterior",
    "Suzuki Cervo kei car exterior",
    "Subaru R1 kei car exterior",
    "Nissan Pixo city car exterior",
    "Honda Life kei car exterior",
    "Daihatsu Move kei car exterior",
    "Suzuki Wagon R kei car exterior",
]

MICRO_TARGET = 800


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


MIN_WIDTH  = 200
MIN_HEIGHT = 150

def download_url(url: str, dest_path: str) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return False
        content_type = resp.headers.get("Content-Type", "")
        if "image" not in content_type or "svg" in content_type:
            return False
        img = Image.open(BytesIO(resp.content))
        w, h = img.size
        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return False
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        img.save(dest_path, "JPEG", quality=90)
        return True
    except Exception:
        pass
    return False


def crawl_class(class_name: str, keywords: list[str], num_per_keyword: int = 80, target: int = None):
    save_dir = os.path.join(BASE_DIR, class_name)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    counter = existing
    print(f"[{class_name}] Mevcut: {existing} görsel, hedef: {target or 'sınırsız'}")

    for keyword in keywords:
        if target and counter >= target:
            print(f"[{class_name}] Hedefe ulaşıldı ({counter}/{target}), durduruluyor.")
            break

        kalan = (target - counter) if target else num_per_keyword
        istek = min(num_per_keyword, kalan)

        print(f"\n[{class_name}] Arıyor: '{keyword}' ({istek} adet)")
        try:
            with DDGS() as ddgs:
                results = list(ddgs.images(
                    keyword,
                    region="us-en",
                    safesearch="off",
                    type_image="photo",   
                    size="Medium",       
                    max_results=istek,
                ))
            time.sleep(2)  
        except Exception as e:
            print(f"  DDG hatası: {e}")
            time.sleep(5)
            continue

        downloaded_this = 0
        for item in results:
            if target and counter >= target:
                break
            url = item.get("image", "")
            if not url:
                continue
            dest = os.path.join(save_dir, f"{counter:06d}.jpg")
            if download_url(url, dest):
                counter += 1
                downloaded_this += 1

        print(f"  {downloaded_this} görsel indirildi")

    total = len([f for f in os.listdir(save_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"\n[{class_name}] Toplam {total} görsel → {save_dir}")



if __name__ == "__main__":
    print("=" * 60)
    print("MICRO → 800 görsele tamamlanıyor...")
    print("Hedef klasör:", os.path.abspath(BASE_DIR))
    print("=" * 60)

    crawl_class("MICRO", MICRO_KEYWORDS, num_per_keyword=50, target=MICRO_TARGET)

    print("\n" + "=" * 60)
    micro_count = len([f for f in os.listdir(os.path.join(BASE_DIR, 'MICRO')) if f.lower().endswith(('.jpg','.jpeg','.png'))]) if os.path.exists(os.path.join(BASE_DIR,'MICRO')) else 0
    print(f"  MICRO: {micro_count} görsel")
    print("=" * 60)
    print("\nSONRAKİ ADIM: Görselleri gözden geçir, kalitesizleri sil,")
    print("sonra split_dataset.py'yi çalıştır.")
