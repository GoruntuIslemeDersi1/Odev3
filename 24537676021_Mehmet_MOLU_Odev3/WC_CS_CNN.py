import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from torchvision import models
import os


klasor_yolu = "resimler"    
dosyalar = sorted([f for f in os.listdir(klasor_yolu)])



# --- CNN Modeli ---
model = models.segmentation.fcn_resnet50(pretrained=True).eval()
transform = T.Compose([
    T.ToTensor(),
    T.Resize((256, 256)),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def canny_sobel_hibrit(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_norm = cv2.convertScaleAbs(sobel_mag)
    canny = cv2.Canny(gray, 50, 150)
    combined = cv2.bitwise_or(sobel_norm, canny)
    return combined

def wavelet_canny_hibrit(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs2 = pywt.dwt2(gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    edges = cv2.convertScaleAbs(LH + HL + HH)
    edges = cv2.resize(edges, (gray.shape[1], gray.shape[0]))
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = cv2.Canny(blur, 50, 150)
    combined = cv2.bitwise_or(edges, canny)
    return combined

def cnn_kenar(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    pred = torch.argmax(output, dim=0).byte().cpu().numpy()
    pred_resized = cv2.resize(pred, (img.shape[1], img.shape[0]))
    cnn_edges = cv2.Canny(np.uint8(pred_resized * 255), 50, 150)
    return cnn_edges


originals = []
canny_sobel_results = []
wavelet_canny_results = []
cnn_results = []

for dosya in dosyalar:
    yol = os.path.join(klasor_yolu, dosya)
    img = cv2.imread(yol)
    if img is None:
        print(f"Yüklenemedi: {dosya}")
        continue

    originals.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    canny_sobel_results.append(canny_sobel_hibrit(img))
    wavelet_canny_results.append(wavelet_canny_hibrit(img))
    cnn_results.append(cnn_kenar(img))



plt.figure(figsize=(18, 10))

for i in range(len(originals)):
    # Orijinal
    plt.subplot(4, len(originals), i + 1)
    plt.imshow(originals[i])
    plt.title(f"Orijinal {i+1}")
    plt.axis('off')

    # Canny + Sobel
    plt.subplot(4, len(originals), i + 1 + len(originals))
    plt.imshow(canny_sobel_results[i], cmap='gray')
    plt.title(f"Canny + Sobel {i+1}")
    plt.axis('off')

    # Wavelet + Canny
    plt.subplot(4, len(originals), i + 1 + 2 * len(originals))
    plt.imshow(wavelet_canny_results[i], cmap='gray')
    plt.title(f"Wavelet + Canny {i+1}")
    plt.axis('off')

    # CNN
    plt.subplot(4, len(originals), i + 1 + 3 * len(originals))
    plt.imshow(cnn_results[i], cmap='gray')
    plt.title(f"CNN {i+1}")
    plt.axis('off')

plt.tight_layout(pad=1.0)
plt.show()
