# make_report.py  (Görsellerin ÜSTÜNE paragraf ekleyen sürüm)
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, PageBreak
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ----------------- FONT KAYDI (TÜRKÇE) -----------------
WIN_ARIAL_REG = r"C:\Windows\Fonts\arial.ttf"
WIN_ARIAL_BOLD = r"C:\Windows\Fonts\arialbd.ttf"
MAC_DEJAVU = "/Library/Fonts/DejaVuSans.ttf"
MAC_DEJAVU_B = "/Library/Fonts/DejaVuSans-Bold.ttf"
MAC_ARIAL = "/Library/Fonts/Arial.ttf"
MAC_ARIAL_B = "/Library/Fonts/Arial Bold.ttf"
LIN_DEJAVU = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LIN_DEJAVU_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def register_tr_fonts():
    candidates_regular = [WIN_ARIAL_REG, MAC_ARIAL, MAC_DEJAVU, LIN_DEJAVU]
    candidates_bold    = [WIN_ARIAL_BOLD, MAC_ARIAL_B, MAC_DEJAVU_B, LIN_DEJAVU_B]
    regular_path = next((p for p in candidates_regular if os.path.isfile(p)), None)
    bold_path    = next((p for p in candidates_bold    if os.path.isfile(p)), None)
    if not regular_path:
        raise FileNotFoundError("Türkçe destekli TTF bulunamadı. Örn: C:\\Windows\\Fonts\\arial.ttf")
    if not bold_path:
        bold_path = regular_path
    pdfmetrics.registerFont(TTFont("TR-Regular", regular_path))
    pdfmetrics.registerFont(TTFont("TR-Bold", bold_path))

register_tr_fonts()

# ----------------- YOLLAR & SABİTLER -----------------
OUT_DIR = "outputs"
OUTPUT_PDF = os.path.join(OUT_DIR, "rapor.pdf")

TITLE = "Görüntü İşleme Çıktıları – Segmentasyon ve Kenar Çıkarımı"
SUBTITLE = "YOLOv8-seg; Canny, Sobel, Laplacian ve Hibrit kenar çıkarımı"

# Rapor sırası
SECTIONS = [
    # (dosya, başlık, paragraf)
    ("1_segmentation_overlay.png", "Segmentation – Maske Bindirme",
     "Bu aşamada görüntü üzerinde nesne segmentasyonu uygulanmıştır. YOLOv8-seg (CNN tabanlı) model, "
     "görüntü içindeki insan, kedi, köpek ve araba nesnelerini piksel düzeyinde tanımlayarak her nesnenin alanını "
     "renkli maskelerle vurgulamıştır. Böylece nesne bölgeleri belirginleşmiş ve sahne içindeki ayrımlar görsel olarak ortaya konmuştur."),

    ("2_segmentation_contours.png", "Segmentation – Kontur + BBox + Etiket",
     "Segmentasyon maskelerinden çıkarılan konturlar nesnelerin sınırlarının çizilmesinde kullanılmış; ek olarak her nesneye ait "
     "dikdörtgen sınır kutuları (bounding box) ve sınıf etiketleri yerleştirilmiştir. Bu görsel, tespit ve segmentasyonun doğruluğunu "
     "gözle değerlendirmeye imkân verir."),

    ("3_edges_canny.png", "Kenar Çıkarımı – Canny",
     "Canny algoritması gri seviyeye indirgenmiş ve bulanıklaştırılmış görüntü üzerinde otomatik alt/üst eşik değerleriyle çalıştırılmıştır. "
     "Amaç, sahnedeki en belirgin ve keskin kenarları gürültüye dayanıklı şekilde yakalamaktır."),

    ("4_edges_sobel_magnitude.png", "Kenar Çıkarımı – Sobel Büyüklük",
     "Sobel operatörü yatay ve dikey gradyanları hesaplayarak kenar adaylarını belirler. Üretilen büyüklük haritası, yoğunluk değişimlerinin "
     "şiddetini gösterir ve yönsel bilgi açısından anlamlıdır."),

    ("5_edges_laplacian.png", "Kenar Çıkarımı – Laplacian",
     "Laplacian ikinci dereceden türev kullanır ve yön bağımsızdır; bu nedenle tüm yönlerdeki ani geçişleri aynı anda vurgular. "
     "Yüksek frekanslı bölgelere (keskin değişimlere) duyarlıdır."),

    ("6_edges_hybrid.png", "Kenar Çıkarımı – Hibrit",
     "Hibrit yaklaşımda Sobel büyüklük görüntüsü Otsu eşikleme ile ikilendirilmiş, düşük eşikli Canny çıktısı ile birleştirilmiş "
     "(lojik OR) ve morfolojik açma ile gürültü temizliği yapılmıştır. Böylece Canny’nin keskin kenarları ile Sobel’in yönsel avantajları "
     "tek bir haritada birleştirilmiştir."),

    ("7_edges_overlay_color.png", "Renk Kodlu Kenar Bindirme",
     "Bu görselde farklı yöntemlerle çıkarılan kenarlar orijinal görüntüye renk kodlu olarak bindirilmiştir: Kırmızı=Canny, Yeşil=Sobel, Mavi=Hibrit. "
     "Bu sayede hangi yöntemin sahnenin hangi bölgelerinde daha etkili olduğu görsel olarak karşılaştırılabilir.")
]

# Sonda metin eklemek istersen:
TEXT_FILES = [
    ("README_CNN_NOTES.txt", "CNN Kısa Açıklama"),
    ("DETECTIONS.txt", "Tespit Sayıları"),
]

# ----------------- STİLLER -----------------
styles = getSampleStyleSheet()
style_title   = ParagraphStyle("TitleTR", parent=styles["Title"], fontName="TR-Bold", fontSize=18, spaceAfter=6)
style_subtt   = ParagraphStyle("SubTR", parent=styles["Normal"], fontName="TR-Regular", fontSize=11, spaceAfter=18)
style_h2      = ParagraphStyle("H2TR", parent=styles["Heading2"], fontName="TR-Bold", fontSize=14, spaceAfter=6, spaceBefore=6)
style_body    = ParagraphStyle("BodyTR", parent=styles["Normal"], fontName="TR-Regular", fontSize=11, leading=15, spaceAfter=8)
style_small   = ParagraphStyle("SmallTR", parent=styles["Normal"], fontName="TR-Regular", fontSize=9, textColor="#555555")

# ----------------- ARAÇLAR -----------------
def image_flowable(path, max_w=A4[0]-3*cm, max_h=A4[1]-7*cm):
    img_reader = ImageReader(path)
    iw, ih = img_reader.getSize()
    scale = min(max_w / iw, max_h / ih)
    return Image(path, width=iw*scale, height=ih*scale)

# ----------------- ANA İŞ -----------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    doc = SimpleDocTemplate(
        OUTPUT_PDF, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    flow = []

    # Üst bilgi (kapak tarzı tek sayfa değil; aynı sayfada metin)
    flow.append(Paragraph(TITLE, style_title))
    flow.append(Paragraph(SUBTITLE, style_subtt))
    flow.append(Paragraph(f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}", style_small))
    flow.append(Spacer(1, 0.5*cm))

    # Görsel bölümleri (metin ÜSTTE, sonra görsel)
    for fname, head, para in SECTIONS:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            flow.append(Paragraph(head, style_h2))
            flow.append(Paragraph(para, style_body))
            flow.append(image_flowable(fpath))
            flow.append(Spacer(1, 0.6*cm))
        else:
            # Dosya yoksa bilgi notu
            flow.append(Paragraph(head, style_h2))
            flow.append(Paragraph("(Not: İlgili görsel bulunamadı: %s)" % fname, style_body))
            flow.append(Spacer(1, 0.4*cm))

    # Ek metin sayfaları (varsa)
    for fname, title in TEXT_FILES:
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.isfile(fpath):
            flow.append(PageBreak())
            flow.append(Paragraph(title, style_h2))
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().replace("\t", "    ")
            # Uzun metinleri de tek Paragraph’ta bırakabiliriz; satır sonları korunur
            flow.append(Paragraph(text.replace("\n", "<br/>"), style_body))

    doc.build(flow)
    print(f"PDF hazır: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
