"""
FastAPI — Araba Gövde Tipi Sınıflandırıcı Backend

Çalıştır:
    uvicorn web.main:app --reload --host 0.0.0.0 --port 8000

Endpoint'ler:
    GET  /          → Arayüz (index.html)
    GET  /health    → Model durumu
    POST /predict   → Görüntü yükle, tahmin al
    GET  /docs      → Swagger UI (otomatik)
"""

import io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Konfigürasyon ─────────────────────────────────────────────────────────────
MODEL_PATH  = Path('models/best_model.pth')
STATIC_DIR  = Path('web/static')
IMG_SIZE    = 224
NUM_CLASSES = 8

# ImageFolder alfabetik sıralama ile aynı sıra olmalı
CLASS_NAMES = ['F1', 'HATCHBACK', 'MICRO', 'PICKUP', 'SEDAN', 'STATION_WAGON', 'SUV', 'VAN']
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model Yükle ───────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        return None
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    print(f'Model yüklendi — device: {device}')
    return model

model = load_model()

# ── Görüntü Dönüşümü ──────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title='Araba Gövde Tipi Sınıflandırıcı',
    description='EfficientNet-B0 ile 8 farklı araba gövde tipi sınıflandırması — Kocaeli Üniversitesi',
    version='1.0.0',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')


@app.get('/', response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / 'index.html'
    return HTMLResponse(content=html_path.read_text(encoding='utf-8'))


@app.get('/health')
async def health():
    return {
        'status': 'ok' if model is not None else 'model_not_loaded',
        'device': str(device),
        'model_loaded': model is not None,
        'model_path': str(MODEL_PATH),
    }


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail='Model henüz yüklenmedi. Önce scripts/train.py çalıştırın.',
        )

    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail='Sadece görüntü dosyaları kabul edilir.')

    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert('RGB')
    except Exception:
        raise HTTPException(status_code=400, detail='Görüntü dosyası okunamadı.')

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

    probs = probabilities.cpu().numpy().tolist()
    top_idx = int(probabilities.argmax().item())
    top_class = CLASS_NAMES[top_idx]
    top_label = CLASS_LABELS[top_class]
    top_confidence = float(probs[top_idx])

    all_predictions = [
        {
            'class_key':   CLASS_NAMES[i],
            'class_label': CLASS_LABELS[CLASS_NAMES[i]],
            'probability': round(probs[i], 4),
        }
        for i in range(NUM_CLASSES)
    ]
    all_predictions.sort(key=lambda x: x['probability'], reverse=True)

    return JSONResponse({
        'prediction':        top_label,
        'class_key':         top_class,
        'confidence':        round(top_confidence, 4),
        'confidence_percent': f'{top_confidence * 100:.1f}%',
        'all_predictions':   all_predictions,
    })
