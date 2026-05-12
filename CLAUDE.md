# CLAUDE.md — AI Bağlam Dosyası

Bu dosya AI asistanların projeyi token harcamadan hızlıca anlaması içindir.
**Proje güncellendiğinde bu dosyayı da güncelleyin.**

---

## Proje Tanımı

Kocaeli Üniversitesi — Yazılım Laboratuvarı II, Proje III
**8 sınıflı araba gövde tipi görüntü sınıflandırması**
EfficientNet-B0 (PyTorch) + FastAPI web arayüzü
**Teslim tarihi: 30.05.2026**

---

## Sınıflar (Klasör Adları)

`SUV` · `VAN` · `SEDAN` · `HATCHBACK` · `PICKUP` · `STATION_WAGON` · `MICRO` · `F1`

Görüntülemede Türkçe label kullanılır (bkz. CLASS_LABELS dict'i train.py/web/main.py içinde).

---

## Teknoloji Stack

| Katman | Araç |
|--------|------|
| Model | EfficientNet-B0 (torchvision) — transfer learning |
| Eğitim | PyTorch 2.x |
| Backend | FastAPI + uvicorn |
| Frontend | HTML + Vanilla JS + Chart.js (CDN) |
| Metrikler | scikit-learn |
| Görselleştirme | matplotlib + seaborn |
| Rapor | LaTeX IEEE format |

---

## Dosya Yapısı

```
car-body-classifier/
├── CLAUDE.md              ← Bu dosya
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/               ← Ham görseller buraya (git'e alınmaz)
│   ├── train/             ← 8 sınıf klasörü (split_dataset.py üretir)
│   ├── val/               ← 8 sınıf klasörü (split_dataset.py üretir)
│   └── SOURCES.txt        ← Veri kaynakları (rapor + sunum için zorunlu)
├── models/
│   └── best_model.pth     ← Eğitilmiş model (git'e alınmaz, < 95 MB)
├── outputs/               ← train.py ve evaluate.py çıktıları (git'e alınmaz)
│   ├── training_history.json
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   └── metrics_report.txt
├── scripts/
│   ├── split_dataset.py   ← Veriyi train/val'a böler
│   ├── train.py           ← EfficientNet-B0 eğitim scripti
│   └── evaluate.py        ← Metrikler + grafikler
├── web/
│   ├── main.py            ← FastAPI uygulaması
│   └── static/
│       ├── index.html     ← Arayüz
│       ├── app.js         ← Frontend JS
│       └── style.css      ← Stiller
└── report/                ← LaTeX dosyaları (Overleaf'te yazılır, buraya kopyalanır)
```

---

## Branch Yapısı

```
main          ← Sunum-ready, stabil
└── dev       ← Entegrasyon (tüm PR'lar buraya gelir)
    ├── feature/dataset        ← Veri toplama ve organizasyon
    ├── feature/preprocessing  ← split_dataset.py
    ├── feature/model          ← train.py
    ├── feature/evaluation     ← evaluate.py + grafikler
    ├── feature/web-api        ← FastAPI backend + HTML/JS
    └── feature/report         ← LaTeX rapor
```

---

## Proje Durumu (BU TABLOYU GÜNCELLE!)

| Faz | Branch | Durum | Not |
|-----|--------|-------|-----|
| Repo kurulumu | main | ✅ Tamamlandı | |
| Dataset indirme | feature/dataset | ⏳ Manuel yapılıyor | Kaggle'dan elle indirilecek |
| Dataset organize | feature/preprocessing | ⏳ Bekliyor | split_dataset.py hazır |
| Model eğitimi | feature/model | ⏳ Bekliyor | train.py hazır |
| Değerlendirme | feature/evaluation | ⏳ Bekliyor | evaluate.py hazır |
| Web arayüzü | feature/web-api | ⏳ Bekliyor | Kod hazır, model bekleniyor |
| Rapor | feature/report | ❌ Başlanmadı | |

---

## Komutlar

```bash
# Bağımlılıkları kur
pip install -r requirements.txt

# 1. Ham görselleri data/raw/<SINIF>/ altına koy, sonra:
python scripts/split_dataset.py --source data/raw --val_ratio 0.2

# 2. Modeli eğit (GPU önerilir, CPU'da ~saatler sürer)
python scripts/train.py

# 3. Metrikleri hesapla ve grafikleri üret
python scripts/evaluate.py

# 4. Web arayüzünü başlat
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

---

## Kritik Kurallar (Değiştirme!)

- Model boyutu **< 95 MB** — EfficientNet-B0 ~20 MB, sorun yok
- Test verisi eğitimde **KESİNLİKLE kullanılmayacak**
- **F1-Score** en önemli metrik → Accuracy → Precision → Recall
- Veri kaynakları `data/SOURCES.txt`'e yazılmalı (sunum + rapor zorunluluğu)
- Sınıf dengesine dikkat: her sınıfta yaklaşık eşit sayıda görsel

---

## Veri Seti Notları

**Dataset 1 — Cars Body Type Cropped (Kaggle, CC0 lisansı):**
- Mevcut sınıflar: SUV, VAN, SEDAN, HATCHBACK, PICK-UP (~1000/sınıf)
- Eksik: STATION_WAGON, MICRO, F1

**Eksik sınıflar için kaynaklar:**
- STATION_WAGON → Stanford Car Body Type Data (Kaggle) + Google Images
- MICRO → Google Images: "smart fortwo", "fiat 500", "city car"
- F1 → Google Images: "formula 1 car side view", "open wheel racing car"

**Hedef:** Her sınıfta 500-1000 görsel, farklı açı/ışık/arka plan

---

## Model Mimarisi Notu

**Neden EfficientNet-B0?**
- ~20 MB model boyutu (95 MB sınırının çok altında)
- ImageNet'te önceden eğitilmiş → az veriyle yüksek doğruluk
- CPU'da bile hızlı inference (~50-100ms)
- Son katman: `classifier[1]` → `nn.Linear(1280, 8)`

---

## Önemli Dosya Konumları

| Dosya | Açıklama |
|-------|----------|
| `models/best_model.pth` | Eğitilmiş model ağırlıkları |
| `outputs/training_history.json` | Her epoch'un loss/acc değerleri |
| `outputs/training_curves.png` | Loss + Accuracy grafikleri |
| `outputs/confusion_matrix.png` | 8x8 normalized confusion matrix |
| `outputs/metrics_report.txt` | Per-class F1/Precision/Recall raporu |
| `data/SOURCES.txt` | Veri kaynakları listesi |
