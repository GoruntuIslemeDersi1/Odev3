import cv2
import numpy as np
import os


resimler = "resimler" 
for dosya in os.listdir(resimler):    

    giris_yolu = os.path.join(resimler, dosya) 
    img = cv2.imread(giris_yolu, cv2.IMREAD_GRAYSCALE)
 
    blur = cv2.GaussianBlur(img, (5, 5), 1) 
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3) 
    sobel_mag = cv2.magnitude(sobelx, sobely)
 
    sobel_norm = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX)
    sobel_uint8 = np.uint8(sobel_norm)
 
    edges = cv2.Canny(sobel_uint8, 50, 150)
 
    cv2.imshow(dosya+"_Orijinal", img)
    cv2.imshow(dosya+"_Sobel", sobel_uint8)
    cv2.imshow(dosya+"_Hibrit (Sobel + Canny)", edges)



cv2.waitKey(0)
cv2.destroyAllWindows()

 