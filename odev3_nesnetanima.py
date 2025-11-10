 
import numpy as np
import tensorflow as tf
import cv2 as cv
import os # Yeni eklenen kütüphane

# Dosya adını tam olarak buraya yazın:
MODEL_ADI = 'frozen_inference_graph.pb'
# ----------------- UYUMLULUK KONTROLLERİ -----------------
try:
    with tf.compat.v1.gfile.FastGFile(MODEL_ADI, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
except Exception as e:
    # Dosya yolu hatası verirse, kullanıcıyı yönlendir.
    print(f"\n[HATA UYARISI] Dosya yolu hatası veya ikili okuma sorunu: {e}")
    print(f"Lütfen '{MODEL_ADI}' dosyasının aynı dizinde olduğundan VE klasör adınızda Türkçe karakter (ö, ç, ş, ğ, ı, ü) OLMADIĞINDAN emin olun.")
    # Programı burada durdurun.
    exit()
with tf.compat.v1.Session() as sess:
   
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    # resim oku ve işle 
    img = cv.imread('test_gorsel.jpg')
    if img is None:
        print("Hata: 'test_gorsel.jpg' dosyası bulunamadı veya okunamadı.")
        exit()
        
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Model Çalıştır
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
    # tanınan kutuda göster
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()
