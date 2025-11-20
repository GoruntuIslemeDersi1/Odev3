import cv2
import numpy as np
import pywt 
 
img = cv2.imread("resimler/img1.jpg", cv2.IMREAD_GRAYSCALE)
 
blur = cv2.GaussianBlur(img, (5, 5), 1)
 
coeffs2 = pywt.dwt2(blur, 'haar')
cA, (cH, cV, cD) = coeffs2   
 
wavelet_edges = np.sqrt(cH**2 + cV**2 + cD**2)
wavelet_edges = cv2.normalize(wavelet_edges, None, 0, 255, cv2.NORM_MINMAX)
wavelet_edges = np.uint8(wavelet_edges)
 
edges = cv2.Canny(wavelet_edges, 50, 150)
 
edges_resized = cv2.resize(edges, (img.shape[1], img.shape[0]))
 
combined = cv2.bitwise_or(edges_resized, cv2.Canny(blur, 50, 150))


cv2.imshow("Orijinal", img)
cv2.imshow("Wavelet Kenarları", wavelet_edges)
cv2.imshow("Hibrit (Wavelet + Canny)", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()
