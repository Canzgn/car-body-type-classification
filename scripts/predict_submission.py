# =============================================================================
# ARABA GÖVDE TİPİ SINIFLANDIRMA — TAHMİN SCRİPTİ
# Kocaeli Üniversitesi, Yazılım Laboratuvarı II, Proje III
#
# Bu dosya PredictionScript.txt olarak kaydedilerek best_model.pth ile
# birlikte teslim klasörüne eklenmelidir.
#
# Colab'da çalıştırma adımları:
#   1. Zip dosyasını /content/ altına yükle ve aç
#   2. best_model.pth dosyasının yolunu MODEL_PATH değişkeninde güncelle
#   3. Aşağıdaki kodu çalıştır:
#        from predict_submission import Predict
#        Predict("/content/testdata")
# =============================================================================

import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install",
                       "torch", "torchvision", "Pillow", "-q"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# ── Sınıf numarası eşleme ─────────────────────────────────────────────────────
# Test.txt'teki sınıf sıralama:
# 1=SUV, 2=VAN, 3=STATION WAGON, 4=MİCRO, 5=AÇIK TEKERLEKLİ, 6=SEDAN,
# 7=HATCHBACK, 8=PICK UP
#
# Modelimizin alfabetik sıralaması (ImageFolder):
# 0=F1, 1=HATCHBACK, 2=MICRO, 3=PICKUP, 4=SEDAN, 5=STATION_WAGON, 6=SUV, 7=VAN

MODEL_IDX_TO_CLASS_NUM = {
    0: 5,  # F1          → AÇIK TEKERLEKLİ
    1: 7,  # HATCHBACK   → HATCHBACK
    2: 4,  # MICRO       → MİCRO
    3: 8,  # PICKUP      → PICK UP
    4: 6,  # SEDAN       → SEDAN
    5: 3,  # STATION_WAGON → STATION WAGON
    6: 1,  # SUV         → SUV
    7: 2,  # VAN         → VAN
}

IMG_SIZE   = 224
NUM_CLASSES = 8

# Colab'da modelin konumu — zip içindeki yola göre güncelle
MODEL_PATH = "/content/best_model.pth"

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None


def _get_model():
    """Modeli bir kez yükle, sonraki çağrılarda önbellekten döndür."""
    global _model
    if _model is None:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, NUM_CLASSES
        )
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(device)
        _model = model
        print(f"Model yüklendi — device: {device}")
    return _model


def Predict(file_path):
    """
    file_path : testdata/ klasörünün dosya yolu
    Klasör yapısı:
        testdata/
          1/  ← SUV
          2/  ← VAN
          3/  ← STATION WAGON
          4/  ← MİCRO
          5/  ← AÇIK TEKERLEKLİ
          6/  ← SEDAN
          7/  ← HATCHBACK
          8/  ← PICK UP

    Çıktı: /content/Preds.txt
    Format: {dosya_adı} | Pred: {sınıf_numarası}
    """
    model = _get_model()

    image_names = []
    preds = []

    SUPPORTED_EXTS = (".jpg", ".jpeg", ".png")

    for class_folder in sorted(os.listdir(file_path)):
        class_dir = os.path.join(file_path, class_folder)
        if not os.path.isdir(class_dir):
            continue

        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(SUPPORTED_EXTS):
                continue

            img_path = os.path.join(class_dir, fname)
            try:
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    model_idx = int(output.argmax(1).item())

                pred_class_num = MODEL_IDX_TO_CLASS_NUM[model_idx]
                image_names.append(fname)
                preds.append(pred_class_num)

            except Exception as e:
                print(f"[HATA] {fname}: {e}")

    # Sonuçları Preds.txt'e yaz
    filename = f"/content/Preds.txt"
    f = open(filename, "w", encoding="utf-8")

    for i in range(len(preds)):
        line = (
            f"{image_names[i]} | Pred: {preds[i]}"
        )
        print(line)
        f.write(line + "\n")

    f.close()
    print(f"\nToplam {len(preds)} görsel işlendi.")
    print(f"Sonuçlar: {filename}")
