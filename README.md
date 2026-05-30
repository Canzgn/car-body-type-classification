# 🚗 Araba Gövde Tipi Sınıflandırıcı

> Kocaeli Üniversitesi — Yazılım Laboratuvarı II, Proje III  
> EfficientNet-B0 (PyTorch) + FastAPI ile 8 sınıflı araba gövde tipi sınıflandırması

---

## Sınıflar

| # | İngilizce | Türkçe |
|---|-----------|--------|
| 1 | `SUV` | SUV |
| 2 | `VAN` | Van |
| 3 | `SEDAN` | Sedan |
| 4 | `HATCHBACK` | Hatchback |
| 5 | `PICKUP` | Pick-Up |
| 6 | `STATION_WAGON` | Station Wagon |
| 7 | `MICRO` | Micro |
| 8 | `F1` | Açık Tekerlekli (F1) |

---

## Teknoloji

| Katman | Araç |
|--------|------|
| Model | EfficientNet-B0 (torchvision) — transfer learning |
| Eğitim | PyTorch 2.x |
| Backend | FastAPI + uvicorn |
| Frontend | HTML + Vanilla JS + Chart.js |
| Metrikler | scikit-learn |
| Görselleştirme | matplotlib + seaborn |

---

## Kurulum

```bash
# Bağımlılıkları kur
pip install -r requirements.txt
```

---

## Kullanım

### 1 — Veri Setini Böl

Kaggle'dan indirdiğin görselleri `data/raw/<SINIF>/` altına koy, ardından:

```bash
python scripts/split_dataset.py --source data/raw --val_ratio 0.2
```

`data/train/` ve `data/val/` otomatik olarak oluşturulur.

### 2 — Modeli Eğit

```bash
python scripts/train.py
```

Çıktı: `models/best_model.pth` (~20 MB, sınır 95 MB)

### 3 — Değerlendirme

```bash
python scripts/evaluate.py
```

Çıktılar `outputs/` klasörüne kaydedilir:
- `training_curves.png` — Loss & Accuracy grafikleri
- `confusion_matrix.png` — 8×8 Normalized Confusion Matrix
- `metrics_report.txt` — Per-class F1 / Precision / Recall

### 4 — Web Arayüzünü Başlat

```bash
uvicorn web.main:app --reload --host 0.0.0.0 --port 8000
```

- Arayüz → http://localhost:8000  
- Swagger UI → http://localhost:8000/docs

---

## Proje Yapısı

```
car-body-type-classification/
├── data/
│   ├── raw/                ← Ham görseller (git'e alınmaz)
│   ├── train/              ← 8 sınıf klasörü
│   ├── val/                ← 8 sınıf klasörü
│   └── SOURCES.txt         ← Veri kaynakları
├── models/
│   └── best_model.pth      ← Eğitilmiş model (git'e alınmaz)
├── outputs/                ← Grafikler ve metrikler (git'e alınmaz)
├── scripts/
│   ├── split_dataset.py
│   ├── train.py
│   └── evaluate.py
├── web/
│   ├── main.py             ← FastAPI backend
│   └── static/
│       ├── index.html
│       ├── app.js
│       └── style.css
└── report/                 ← LaTeX rapor dosyaları
```

---

## API

| Method | Endpoint | Açıklama |
|--------|----------|----------|
| `GET` | `/` | Web arayüzü |
| `GET` | `/health` | Model durumu |
| `POST` | `/predict` | Görüntü yükle, tahmin al |
| `GET` | `/docs` | Swagger UI |

### Örnek `/predict` yanıtı

```json
{
  "prediction": "Sedan",
  "class_key": "SEDAN",
  "confidence": 0.9421,
  "confidence_percent": "94.2%",
  "all_predictions": [
    { "class_key": "SEDAN", "class_label": "Sedan", "probability": 0.9421 },
    { "class_key": "HATCHBACK", "class_label": "Hatchback", "probability": 0.0312 },
    ...
  ]
}
```

---

## Değerlendirme Metrikleri

Öncelik sırası: **F1-Score** → Accuracy → Precision → Recall

Her metrik hem per-class hem de macro/weighted average olarak hesaplanır.

---

## Veri Kaynakları

- [Cars Body Type Cropped — Kaggle (CC0)](https://www.kaggle.com/datasets/ademboukhris/cars-body-type-cropped)
- [Stanford Car Body Type Data — Kaggle](https://www.kaggle.com/datasets/mayurmahurkar/stanford-car-body-type-data)

Ek kaynaklar: `data/SOURCES.txt`
