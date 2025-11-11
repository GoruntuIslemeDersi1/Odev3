import cv2
import numpy as np
import matplotlib.pyplot as plt

# Görüntüyü yükle (Bu yolu kendi dosya yolunuzla değiştirin)
# Örn: 'gorsel.jpg'
try:
    img = cv2.imread('test_gorsel.jpg')
    if img is None:
        raise FileNotFoundError("Görüntü dosyası bulunamadı.")
except FileNotFoundError as e:
    print(e)
    # Yerine basit bir siyah görüntü oluştur (Demo amaçlı)
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    cv2.putText(img, "Demo Gorsel", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Basit İkili Eşikleme (Segmentasyon Benzetimi)
# Pikselleri iki sınıfa ayırır: 0 (siyah) veya 255 (beyaz)
esik_degeri = 120
_, mask = cv2.threshold(img_gray, esik_degeri, 255, cv2.THRESH_BINARY)

# Orijinal görüntüyü ve maskeyi göster
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Orijinal Görüntü')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Basit Segmentasyon Maskesi (Eşik: {esik_degeri})')
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.show()



# 1. Sobel Kenar Çıkarımı
sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5) # Yatay kenarlar
sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5) # Dikey kenarlar
sobel_mag = cv2.magnitude(sobel_x, sobel_y) # Toplam büyüklük

# 2. Canny Kenar Çıkarımı
# 100 ve 200, histerezis eşikleri (T1 ve T2)
canny_edges = cv2.Canny(img_gray, 100, 200)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Orijinal Görüntü')
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Sobel Kenar (Büyüklük)')
plt.imshow(sobel_mag, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Canny Kenar Çıkarımı')
plt.imshow(canny_edges, cmap='gray')
plt.axis('off')
plt.show()



# Laplacian Uygulama
# cv2.CV_64F: Çıktı derinliği
laplacian = cv2.Laplacian(img, cv2.CV_64F)

# Çıktı değerlerini görselleştirme için normalize et
laplacian_abs = cv2.convertScaleAbs(laplacian)

plt.figure(figsize=(6, 4))
plt.title('Laplacian Kenar Çıkarımı')
plt.imshow(laplacian_abs, cmap='gray')
plt.axis('off')
plt.show()