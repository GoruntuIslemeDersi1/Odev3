import os
import cv2
import numpy as np
from ultralytics import YOLO

# ================== AYARLAR ==================
IMG_PATH = "images/image.png"       # Görselin konumu
OUT_DIR = "outputs"                 # Sonuç klasörü
CONF_THRES = 0.4                    # YOLO güven eşiği
INTEREST_CLASSES = {"person", "cat", "dog", "car"}  # İlgilendiğimiz sınıflar

os.makedirs(OUT_DIR, exist_ok=True)

# ================== YARDIMCI ==================
def auto_canny(image, sigma=0.33):
    # medyane göre otomatik eşik (Canny için pratik)
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper, L2gradient=True)

# ================== GÖRSELİ YÜKLE ==================
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Görüntü bulunamadı: {IMG_PATH}")
H, W = img.shape[:2]

# ================== 1) SEGMENTASYON (YOLOv8-seg) ==================
# İlk kullanımda ağırlık indirir (internet gerekir)
model = YOLO("yolov8n-seg.pt")
results = model.predict(source=img, conf=CONF_THRES, verbose=False)

seg_overlay = img.copy()
seg_contours = img.copy()
det_count = {"person": 0, "cat": 0, "dog": 0, "car": 0}

if len(results):
    res = results[0]
    names = res.names

    if res.masks is not None and res.boxes is not None:
        masks = res.masks.data.cpu().numpy()        # [N, Hm, Wm] (model mask boyutu)
        boxes = res.boxes
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        rng = np.random.default_rng(42)

        for i, cls_id in enumerate(cls_ids):
            name = names.get(int(cls_id), str(cls_id))
            if name not in INTEREST_CLASSES:
                continue

            det_count[name] = det_count.get(name, 0) + 1

            # --- MASK BOYUTU DÜZELTME: orijinal HxW boyutuna ölçekle ---
            m_small = masks[i]  # (Hm, Wm)
            m_resized = cv2.resize(m_small, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (m_resized > 0.5).astype(np.uint8)  # (H, W) ikili maske

            color = rng.integers(50, 255, size=3, dtype=np.uint8).tolist()

            # Yarı saydam maske bindirme
            colored = np.zeros_like(img, dtype=np.uint8)
            colored[m == 1] = color
            seg_overlay = cv2.addWeighted(seg_overlay, 1.0, colored, 0.45, 0)

            # Kontur + bbox + etiket
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_contours, contours, -1, color, 2)

            b = boxes.xyxy[i].cpu().numpy().astype(int)  # orijinal boyutta bbox
            x1, y1, x2, y2 = b
            cv2.rectangle(seg_contours, (x1, y1), (x2, y2), color, 2)
            cv2.putText(seg_contours, name, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    else:
        print("UYARI: Segmentasyon maskesi üretilemedi.")
else:
    print("UYARI: YOLO sonuç döndürmedi.")

cv2.imwrite(os.path.join(OUT_DIR, "1_segmentation_overlay.png"), seg_overlay)
cv2.imwrite(os.path.join(OUT_DIR, "2_segmentation_contours.png"), seg_contours)

# ================== 2) KENAR ÇIKARIMI ==================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

# 2.1) Canny (otomatik eşik)
edges_canny = auto_canny(blur, sigma=0.33)
cv2.imwrite(os.path.join(OUT_DIR, "3_edges_canny.png"), edges_canny)

# 2.2) Sobel büyüklük
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobelx**2 + sobely**2)
sobel_mag = (255 * (sobel_mag / (sobel_mag.max() + 1e-8))).astype(np.uint8)
cv2.imwrite(os.path.join(OUT_DIR, "4_edges_sobel_magnitude.png"), sobel_mag)

# 2.3) Laplacian
lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
lap = np.absolute(lap)
lap = (255 * (lap / (lap.max() + 1e-8))).astype(np.uint8)
cv2.imwrite(os.path.join(OUT_DIR, "5_edges_laplacian.png"), lap)

# ================== 2.b) HİBRİT KENAR ==================
# Sobel (Otsu ile ikili)  OR  yumuşak Canny -> sonra morfolojik temizlik
_, sobel_bin = cv2.threshold(sobel_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
edges_canny_soft = cv2.Canny(blur, 60, 120, L2gradient=True)
hybrid = cv2.bitwise_or(sobel_bin, edges_canny_soft)

kernel = np.ones((3, 3), np.uint8)
hybrid_clean = cv2.morphologyEx(hybrid, cv2.MORPH_OPEN, kernel, iterations=1)
cv2.imwrite(os.path.join(OUT_DIR, "6_edges_hybrid.png"), hybrid_clean)

# Kenarları orijinale bindirme (renk kodlu görsel)
overlay_edges = img.copy()
overlay_edges[edges_canny > 0] = (0, 0, 255)   # Canny: kırmızı
overlay_edges[sobel_bin > 0] = (0, 255, 0)     # Sobel: yeşil
overlay_edges[hybrid_clean > 0] = (255, 0, 0)  # Hibrit: mavi
cv2.imwrite(os.path.join(OUT_DIR, "7_edges_overlay_color.png"), overlay_edges)

# ================== LOG ==================
print("\nBitti ✅  Çıktılar 'outputs/' klasöründe:")
print(" 1_segmentation_overlay.png")
print(" 2_segmentation_contours.png")
print(" 3_edges_canny.png")
print(" 4_edges_sobel_magnitude.png")
print(" 5_edges_laplacian.png")
print(" 6_edges_hybrid.png")
print(" 7_edges_overlay_color.png")
