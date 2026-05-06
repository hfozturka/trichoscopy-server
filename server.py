"""
ClinicDesk Trichoscopy Server
OpenCV tabanlı saç analizi - internetsiz çalışır
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
import math
from typing import Optional

app = FastAPI(title="ClinicDesk Trichoscopy", version="1.0.0")

# CORS - ClinicDesk CRM'den erişim için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def load_image(file_bytes: bytes) -> np.ndarray:
    """Bytes'tan OpenCV görüntüsü yükle"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def detect_square_marker(img: np.ndarray):
    """
    Mor/mavi marker kalemi ile çizilmiş kareyi tespit et
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([120, 40, 40])
    upper = np.array([160, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    squares = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000 or area > (h * w * 0.8):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        x, y, bw, bh = cv2.boundingRect(cnt)
        ratio = float(bw) / bh if bh > 0 else 0
        if 0.6 < ratio < 1.6 and len(approx) >= 4:
            squares.append({
                'contour': approx,
                'area': area,
                'x': x, 'y': y,
                'w': bw, 'h': bh,
                'cx': x + bw//2,
                'cy': y + bh//2
            })
    if not squares:
        return None
    return max(squares, key=lambda s: s['area'])

def calibrate_from_square(square, real_size_mm: float = 10.0) -> float:
    """
    1cm² (10mm x 10mm) kareden piksel/µm kalibrasyonu
    real_size_mm: gerçek boyut mm cinsinden (default 10mm = 1cm)
    """
    pixel_size = (square['w'] + square['h']) / 2  # ortalama piksel boyutu
    real_size_um = real_size_mm * 1000  # mm → µm
    microns_per_pixel = real_size_um / pixel_size
    return microns_per_pixel

def segment_hair(img: np.ndarray, roi=None):
    """
    Saç tellerini segmente et
    roi: (x, y, w, h) - ilgilenilen alan (1cm² kare)
    """
    if roi:
        x, y, w, h = roi
        # Biraz padding ekle
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img.shape[1], x + w + pad)
        y2 = min(img.shape[0], y + h + pad)
        img_roi = img[y1:y2, x1:x2]
    else:
        img_roi = img

    gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Gaussian blur - gürültü azalt
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # 2. Adaptif threshold - saç telleri koyu, arka plan açık
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4
    )

    # 3. Morfolojik işlemler - gürültü temizle
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Küçük gürültüleri sil
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
    # Saç tellerini birleştir
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_medium)

    return thresh, img_roi

def count_and_measure_hairs(thresh: np.ndarray, microns_per_pixel: float):
    """
    Saç tellerini say ve kalınlıklarını ölç
    """
    # Konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hair_data = []
    img_h, img_w = thresh.shape

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Çok küçük veya çok büyük konturları ele
        min_area = 20
        max_area = img_h * img_w * 0.1

        if area < min_area or area > max_area:
            continue

        # Bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Uzunluk/genişlik oranı - saç teli uzun ve ince olmalı
        if bw == 0 or bh == 0:
            continue

        aspect_ratio = max(bw, bh) / min(bw, bh)

        # Saç teli: uzun ve ince (oran > 2)
        if aspect_ratio < 1.5:
            continue

        # Kalınlık = kısa kenar (piksel)
        thickness_px = min(bw, bh)
        thickness_um = thickness_px * microns_per_pixel

        # Makul saç kalınlığı filtresi (10-200 µm)
        if thickness_um < 10 or thickness_um > 200:
            continue

        # Uzunluk
        length_px = max(bw, bh)
        length_um = length_px * microns_per_pixel

        # Merkez koordinatlar
        cx = x + bw // 2
        cy = y + bh // 2

        hair_data.append({
            'thickness_px': float(thickness_px),
            'thickness_um': round(float(thickness_um), 1),
            'length_um': round(float(length_um), 1),
            'area_px': float(area),
            'cx': cx,
            'cy': cy,
        })

    return hair_data

def calculate_statistics(hair_data: list, microns_per_pixel: float, area_cm2: float = 1.0):
    """
    Saç istatistiklerini hesapla
    """
    if not hair_data:
        return {
            'hair_count': 0,
            'hair_per_cm2': 0,
            'avg_thickness_um': 0,
            'min_thickness_um': 0,
            'max_thickness_um': 0,
            'thickness_category': 'Tespit edilemedi',
            'thin_percent': 0,
            'medium_percent': 0,
            'thick_percent': 0,
        }

    thicknesses = [h['thickness_um'] for h in hair_data]
    hair_count = len(hair_data)
    hair_per_cm2 = round(hair_count / area_cm2)
    avg_thickness = round(np.mean(thicknesses), 1)
    min_thickness = round(min(thicknesses), 1)
    max_thickness = round(max(thicknesses), 1)
    std_thickness = round(float(np.std(thicknesses)), 1)

    # Kalınlık kategorileri
    thin = sum(1 for t in thicknesses if t < 60)
    medium = sum(1 for t in thicknesses if 60 <= t < 80)
    thick = sum(1 for t in thicknesses if t >= 80)

    thin_pct = round(thin / hair_count * 100)
    medium_pct = round(medium / hair_count * 100)
    thick_pct = round(thick / hair_count * 100)

    # Genel kategori
    if avg_thickness < 60:
        category = "İnce"
    elif avg_thickness < 80:
        category = "Orta"
    else:
        category = "Kalın"

    # Miniaturizasyon skoru (ince saç oranı)
    miniaturization = thin_pct

    # Yoğunluk değerlendirmesi
    if hair_per_cm2 >= 200:
        density = "Çok Yüksek"
        density_score = min(100, 70 + (hair_per_cm2 - 200) // 5)
    elif hair_per_cm2 >= 150:
        density = "Yüksek"
        density_score = 60 + (hair_per_cm2 - 150) // 5
    elif hair_per_cm2 >= 100:
        density = "Orta"
        density_score = 40 + (hair_per_cm2 - 100) // 5
    else:
        density = "Düşük"
        density_score = max(0, hair_per_cm2 // 3)

    return {
        'hair_count': hair_count,
        'hair_per_cm2': hair_per_cm2,
        'avg_thickness_um': avg_thickness,
        'min_thickness_um': min_thickness,
        'max_thickness_um': max_thickness,
        'std_thickness_um': std_thickness,
        'thickness_category': category,
        'thin_percent': thin_pct,
        'medium_percent': medium_pct,
        'thick_percent': thick_pct,
        'miniaturization_percent': miniaturization,
        'density': density,
        'density_score': int(density_score),
    }

def create_visualization(img: np.ndarray, hair_data: list, square=None) -> str:
    """
    Tespit edilen saçları görselleştir, base64 döndür
    """
    vis = img.copy()

    # 1cm² damgayı çiz
    if square:
        cv2.rectangle(vis,
            (square['x'], square['y']),
            (square['x'] + square['w'], square['y'] + square['h']),
            (0, 255, 0), 2)
        cv2.putText(vis, "1cm²",
            (square['x'], square['y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Saç tellerini işaretle
    for i, hair in enumerate(hair_data):
        cx, cy = int(hair['cx']), int(hair['cy'])
        t = hair['thickness_um']

        # Renk: kalınlığa göre
        if t < 60:
            color = (0, 165, 255)   # turuncu = ince
        elif t < 80:
            color = (0, 255, 0)     # yeşil = orta
        else:
            color = (255, 0, 0)     # mavi = kalın

        cv2.circle(vis, (cx, cy), 4, color, -1)

    # Görüntüyü base64'e çevir
    _, buffer = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ClinicDesk Trichoscopy Server çalışıyor", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    bolge: str = Form(default="Genel"),
    real_size_mm: float = Form(default=10.0),  # damga boyutu mm
):
    """
    Ana analiz endpoint
    - file: görüntü dosyası
    - bolge: saçlı deri bölgesi (Ön/Arka/Sol/Sağ)
    - real_size_mm: damga gerçek boyutu (default 10mm = 1cm)
    """
    try:
        # Görüntü yükle
        contents = await file.read()
        img = load_image(contents)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Görüntü yüklenemedi"})

        h, w = img.shape[:2]

        # 1. Damga tespiti ve kalibrasyon
        square = detect_square_marker(img)

        if square:
            microns_per_pixel = calibrate_from_square(square, real_size_mm)
            calibration_method = "damga"
            area_px = square['w'] * square['h']
            area_cm2 = 1.0  # damga = 1cm²
            roi = (square['x'], square['y'], square['w'], square['h'])
        else:
            # Damga bulunamadı - varsayılan kalibrasyon kullan
            microns_per_pixel = 0.45  # önceki kalibrasyon değeri
            calibration_method = "varsayılan"
            area_cm2 = (w * microns_per_pixel / 10000) * (h * microns_per_pixel / 10000)
            roi = None

        # 2. Saç segmentasyonu
        thresh, img_roi = segment_hair(img, roi)

        # 3. Saç sayımı ve ölçüm
        hair_data = count_and_measure_hairs(thresh, microns_per_pixel)

        # 4. İstatistikler
        stats = calculate_statistics(hair_data, microns_per_pixel, area_cm2)

        # 5. Görselleştirme
        vis_base64 = create_visualization(img_roi, hair_data, square if square else None)

        return {
            "bolge": bolge,
            "calibration": {
                "method": calibration_method,
                "microns_per_pixel": round(microns_per_pixel, 4),
                "square_detected": square is not None,
                "real_size_mm": real_size_mm,
            },
            "image_size": {"width": w, "height": h},
            "results": stats,
            "hair_details": hair_data[:50],  # ilk 50 saç detayı
            "visualization": vis_base64,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/calibrate")
async def calibrate_only(
    file: UploadFile = File(...),
    real_size_mm: float = Form(default=10.0),
):
    """
    Sadece kalibrasyon - damga tespiti ve µm/piksel hesabı
    """
    try:
        contents = await file.read()
        img = load_image(contents)

        square = detect_square_marker(img)

        if not square:
            return JSONResponse(status_code=404, content={
                "error": "Damga tespit edilemedi",
                "tip": "Görüntüde 1cm² kare damga olduğundan emin olun"
            })

        microns_per_pixel = calibrate_from_square(square, real_size_mm)

        return {
            "square_detected": True,
            "square": {
                "x": square['x'],
                "y": square['y'],
                "width": square['w'],
                "height": square['h'],
                "pixel_size": (square['w'] + square['h']) / 2,
            },
            "calibration": {
                "microns_per_pixel": round(microns_per_pixel, 4),
                "real_size_mm": real_size_mm,
                "real_size_um": real_size_mm * 1000,
            }
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/thickness")
async def measure_thickness_only(
    file: UploadFile = File(...),
    microns_per_pixel: float = Form(default=0.45),
):
    """
    Sadece kalınlık ölçümü - tek saç teli için
    Mikrometre kalibrasyon saçı fotoğrafı için kullan
    """
    try:
        contents = await file.read()
        img = load_image(contents)

        thresh, _ = segment_hair(img)
        hair_data = count_and_measure_hairs(thresh, microns_per_pixel)

        if not hair_data:
            return JSONResponse(status_code=404, content={
                "error": "Saç teli tespit edilemedi"
            })

        # En büyük/belirgin saç telini bul
        main_hair = max(hair_data, key=lambda h: h['area_px'])

        return {
            "thickness_um": main_hair['thickness_um'],
            "thickness_px": main_hair['thickness_px'],
            "category": "İnce" if main_hair['thickness_um'] < 60 else "Orta" if main_hair['thickness_um'] < 80 else "Kalın",
            "all_hairs": len(hair_data),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("🔬 ClinicDesk Trichoscopy Server başlatılıyor...")
    print("📍 Adres: http://localhost:8000")
    print("📖 API Docs: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
