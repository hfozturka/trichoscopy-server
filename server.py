"""
ClinicDesk Trichoscopy Server
OpenCV tabanlı saç analizi - internetsiz çalışır
"""

from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from PIL import Image
import io
import base64
import math
import os
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

# API anahtarı kontrolü - TRICHOSCOPY_API_KEY ayarlıysa /analyze, /calibrate, /thickness
# uçlarına gelen her istekte doğru X-Api-Key header'ı zorunlu olur (ayarlı değilse kontrol atlanır,
# böylece yerel geliştirme eskisi gibi çalışmaya devam eder).
API_KEY = os.environ.get("TRICHOSCOPY_API_KEY")

def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ─── Yardımcı Fonksiyonlar ────────────────────────────────────────────────────

def load_image(file_bytes: bytes) -> np.ndarray:
    """Bytes'tan OpenCV görüntüsü yükle"""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def detect_square_marker_beyaz_kapak(img: np.ndarray):
    """
    Klinikte fiilen kullanilan alet: beyaz/gumus metal kapakli, ortasinda kare
    bir olcum penceresi olan bir loop (buyutec). Kapagin govdesi parlak ve
    dusuk doygunlukta (beyaz/gumus); kare pencere ise icinden sacli deri
    gorundugu icin kapagin govdesine gore belirgin derecede koyu/doygun kalir.
    Kapagi "parlak govde + ortasinda delik (pencere)" seklinde tespit edip
    o deligin sinirlayici kutusunu dondururuz.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((val > 150) & (sat < 70)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    hierarchy = hierarchy[0]
    bright_bool = mask > 0

    # NOT: Daha once burada "ebeveyn" (kapagin govdesi) konturunun GENEL
    # bounding-box'inin da kare/yuvarlak ve dolgun olmasini istiyorduk. Bu,
    # gercek fotograflarda kirilgan cikti: JPEG kalitesi/sikistirma biraz
    # degisince (ör. ayni foto farkli kalitede kaydedilince), kapagin parlak
    # maskesi kadrajin baska bir yerindeki alakasiz parlak bir bolgeyle
    # (ör. sac teli, arka plan) birlesip govdenin GENEL bounding-box'ini
    # genisletiyor, "extent" (dolgunluk) degerini dusurup gercek kapagi
    # reddediyordu - ayni fotografin sadece sikistirma kalitesi farkli iki
    # kopyasinda biri "damga tespit edildi" biri "varsayilan"a duserek
    # tamamen farkli (ve yanlis) sonuclar uretiyordu. Bunun yerine, adayin
    # HEMEN cevresindeki dar bir halkanin (kapagin govdesine ait olmasi
    # beklenen bolge) ne kadarinin parlak oldugunu kontrol ediyoruz - bu,
    # kadrajin uzak bir yerindeki alakasiz parlak bolgelerden etkilenmez.
    min_area = max(500.0, 0.008 * h * w)
    candidates = []
    for i, cnt in enumerate(contours):
        parent = hierarchy[i][3]
        if parent == -1:
            continue  # sadece "delik" (parlak govdenin icindeki bosluk) konturlariyla ilgileniyoruz
        area = cv2.contourArea(cnt)
        # Mutlak alan esigine ek olarak, kadraja oranla da makul buyuklukte
        # olmali (ör. masa/zeminde kalan kucuk, alakasiz parlak bosluklari eler)
        if area < min_area or area > (h * w * 0.3):
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        ratio = float(bw) / bh if bh > 0 else 0
        if not (0.6 < ratio < 1.6):
            continue
        extent = area / float(bw * bh) if bw * bh else 0
        if extent < 0.65:  # dolgun/kare olmali, ince/duzensiz seritler elenir
            continue

        # Yerel halka kontrolu: adayin hemen disindaki bandin buyuk kismi
        # parlak mi (yani gercekten parlak bir govde tarafindan cevrili mi)?
        ring_margin = max(3, int(round(0.25 * min(bw, bh))))
        ox1 = max(0, x - ring_margin)
        oy1 = max(0, y - ring_margin)
        ox2 = min(w, x + bw + ring_margin)
        oy2 = min(h, y + bh + ring_margin)
        outer = bright_bool[oy1:oy2, ox1:ox2]
        inner = bright_bool[y:y + bh, x:x + bw]
        ring_px = outer.size - inner.size
        ring_frac = (outer.sum() - inner.sum()) / ring_px if ring_px > 0 else 0
        if ring_frac < 0.4:
            continue

        candidates.append({'area': area, 'x': x, 'y': y, 'w': bw, 'h': bh, 'cx': x + bw // 2, 'cy': y + bh // 2})

    if not candidates:
        return None
    return max(candidates, key=lambda c: c['area'])


def detect_square_marker_cizim(img: np.ndarray):
    """
    Eski yontem: mor/mavi marker kalemiyle elle cizilmis kareyi tespit et.
    Klinikte fiilen kullanilan beyaz kapakli loop icin degil, olasi alternatif
    bir kalibrasyon yontemi (elle cizilmis kare) icin yedek olarak tutuluyor.
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


def detect_square_marker(img: np.ndarray):
    """
    Once klinikte fiilen kullanilan beyaz kapakli loop'u tespit etmeyi dener;
    bulamazsa (ör. baska bir kalibrasyon yontemi kullanildiysa) eski mor/mavi
    marker-cizimi tespitine geri duser.
    """
    return detect_square_marker_beyaz_kapak(img) or detect_square_marker_cizim(img)

def calibrate_from_square(square, real_size_mm: float = 20.0) -> float:
    """
    Kare damgadan piksel/µm kalibrasyonu
    real_size_mm: gerçek boyut mm cinsinden (default 20mm = 2cm, klinikte kullanılan damga boyutu)
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

        # Makul saç kalınlığı filtresi (10-200 µm baz alınır, orijinal varsayım
        # ~0.45 µm/piksel gibi çok yüksek çözünürlüklü görüntüler için ayarlanmıştı).
        # Damga (2x2cm) ile kalibrasyonda çözünürlük çok daha kaba olabiliyor
        # (ör. ~50+ µm/piksel) - bu durumda bulanıklık/JPEG sıkıştırması yüzünden
        # tek bir kıl piksel bazında birkaç piksel kalınlığında görünüyor ve sabit
        # 200µm tavanını her zaman aşıyor, sonuç olarak TÜM saçlar elenip "0 saç"
        # çıkıyordu. Üst sınırı çözünürlüğe göre esnetiyoruz ki kaba çözünürlükte
        # de makul kalınlıktaki kıllar (birkaç piksel) elenmesin; gürültü/gölge gibi
        # çok büyük bloklar yine de area/aspect_ratio filtrelerinden elenir.
        max_thickness_um = max(200.0, microns_per_pixel * 6)
        if thickness_um < 10 or thickness_um > max_thickness_um:
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
    Tespit edilen saçları HairMetrix tarzı halka (ring) işaretleriyle göster,
    alt kısımda kalınlık dağılım barı ve renk anahtarı ekle, base64 döndür.
    """
    vis = img.copy()
    img_h, img_w = vis.shape[:2]

    # 1cm² damgayı çiz
    if square:
        cv2.rectangle(vis,
            (square['x'], square['y']),
            (square['x'] + square['w'], square['y'] + square['h']),
            (0, 255, 0), 2)
        cv2.putText(vis, "1cm²",
            (square['x'], square['y'] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # HairMetrix tarzı 4 kademeli kalınlık renklendirmesi (BGR)
    CAT_COLORS = [
        (60, 60, 220),    # <30µm   - kırmızı: minyatürize/vellus
        (0, 215, 255),    # 30-60µm - sarı: ince terminal
        (144, 238, 144),  # 60-90µm - açık yeşil: orta terminal
        (34, 120, 34),    # >90µm   - koyu yeşil: kalın terminal
    ]

    def category_color(t):
        if t < 30:
            return CAT_COLORS[0]
        elif t < 60:
            return CAT_COLORS[1]
        elif t < 90:
            return CAT_COLORS[2]
        return CAT_COLORS[3]

    # Saç tellerini halka (ring) olarak işaretle
    for hair in hair_data:
        cx, cy = int(hair['cx']), int(hair['cy'])
        color = category_color(hair['thickness_um'])
        cv2.circle(vis, (cx, cy), 5, color, 2)

    # ─── Alt şerit: kalınlık dağılım barı + renk anahtarı ──────────────────
    bar_h = 56
    strip = np.full((bar_h, img_w, 3), (24, 24, 24), dtype=np.uint8)

    thicknesses = [hh['thickness_um'] for hh in hair_data]
    t_min = min(thicknesses) if thicknesses else 0
    t_max = max(thicknesses) if thicknesses else 0

    bar_x0, bar_x1 = 50, max(60, img_w - 260)
    bar_y0, bar_y1 = 16, 32
    if bar_x1 > bar_x0:
        seg_w = (bar_x1 - bar_x0) / 4
        for i, c in enumerate(CAT_COLORS):
            x_start = int(bar_x0 + i * seg_w)
            x_end = int(bar_x0 + (i + 1) * seg_w) if i < 3 else bar_x1
            cv2.rectangle(strip, (x_start, bar_y0), (x_end, bar_y1), c, -1)
        cv2.putText(strip, f"{t_min:.0f}", (max(2, bar_x0 - 32), 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(strip, f"{t_max:.0f}", (bar_x1 + 6, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(strip, "sac kalinligi (um)", (bar_x0, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)

    legend = [
        (CAT_COLORS[0], "<30 vellus"),
        (CAT_COLORS[1], "30-60"),
        (CAT_COLORS[2], "60-90"),
        (CAT_COLORS[3], ">90um"),
    ]
    lx = max(bar_x1 + 60, img_w - 215)
    for i, (c, label) in enumerate(legend):
        bx = lx + i * 54
        if bx + 14 > img_w:
            break
        cv2.rectangle(strip, (bx, 12), (bx + 14, 26), c, -1)
        cv2.putText(strip, label, (bx - 6, 44),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA)

    vis_with_bar = np.vstack([vis, strip])

    # Görüntüyü base64'e çevir
    _, buffer = cv2.imencode('.jpg', vis_with_bar, [cv2.IMWRITE_JPEG_QUALITY, 88])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

# ─── API Endpoints ────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ClinicDesk Trichoscopy Server çalışıyor", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(
    file: UploadFile = File(...),
    bolge: str = Form(default="Genel"),
    real_size_mm: float = Form(default=20.0),  # damga boyutu mm
):
    """
    Ana analiz endpoint
    - file: görüntü dosyası
    - bolge: saçlı deri bölgesi (Ön/Arka/Sol/Sağ)
    - real_size_mm: damga gerçek boyutu (default 20mm = 2cm, klinikte kullanılan damga boyutu)
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
            area_cm2 = (real_size_mm / 10.0) ** 2  # damga gerçek fiziksel boyutuna göre alan (cm²) - ONEMLI: sabit 1cm² varsayimi yogunlugu yanlis hesapliyordu
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


@app.post("/calibrate", dependencies=[Depends(verify_api_key)])
async def calibrate_only(
    file: UploadFile = File(...),
    real_size_mm: float = Form(default=20.0),
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
                "tip": "Görüntüde 2x2cm kare damga olduğundan emin olun"
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


@app.post("/thickness", dependencies=[Depends(verify_api_key)])
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
