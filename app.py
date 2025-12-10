import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ExifTags
import easyocr

# ------------------ PAGE SETUP ------------------
st.set_page_config(page_title="Number Plate Detection", page_icon="🚘", layout="wide")
st.title("🚘 Number Plate Detection of Vehicles (YOLO + EasyOCR)")

# ------------------ AUTO IMAGE ROTATION FIX ------------------
def fix_rotation(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation = exif.get(orientation)
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
    except:
        pass
    return img

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    weights_path = "best.pt"
    if os.path.exists(weights_path):
        st.success("✅ Loaded custom YOLO model (best.pt)")
        return YOLO(weights_path)
    else:
        st.warning("⚠️ best.pt not found. Loading default YOLOv8n")
        return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

model = load_model()
reader = load_ocr()

# ------------------ OCR PROCESSING ------------------
def preprocess_plate(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def easy_ocr_text(img_gray):
    texts = reader.readtext(img_gray, detail=0)
    return " ".join(t.strip() for t in texts if t.strip())

INDIA_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$')

def clean_plate_text(raw_text):
    if not raw_text:
        return ""
    t = raw_text.upper().replace(" ", "")
    t = t.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
    return t

# ------------------ DETECTION FUNCTION ------------------
def detect_plates(image_bgr, conf_thr, min_w, min_h):

    start = time.time()
    results = model(image_bgr, conf=conf_thr)
    elapsed = time.time() - start

    overlay = results[0].plot()
    rows = []
    confs = []

    if results[0].boxes is None:
        return overlay, rows, elapsed, confs

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), c in zip(boxes, scores):

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1

        if w < min_w or h < min_h:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        processed = preprocess_plate(crop)
        raw = easy_ocr_text(processed)
        text = clean_plate_text(raw)

        cv2.putText(
            overlay,
            f"{text} ({c:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        confs.append(c)
        rows.append({
            "Plate No": text,
            "Confidence": round(float(c), 2)
        })

    return overlay, rows, elapsed, confs

# ------------------ SIDEBAR CONTROLS ------------------
st.sidebar.header("Controls")

conf_thr = st.sidebar.slider("Detection confidence", 0.01, 0.90, 0.10, 0.01)
min_w = st.sidebar.number_input("Min plate width (px)", 5, 2000, 20, 5)
min_h = st.sidebar.number_input("Min plate height (px)", 5, 2000, 10, 5)

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    img = Image.open(uploaded_file).convert("RGB")
    img = fix_rotation(img)

    img_np = np.array(img)[:, :, ::-1]

    overlay, rows, elapsed, confs = detect_plates(img_np, conf_thr, min_w, min_h)

    st.image(overlay[:, :, ::-1], caption="Detections", use_container_width=True)

    st.subheader("📊 Performance")
    st.metric("Processing Time (s)", f"{elapsed:.3f}")

    if len(confs):
        st.metric("Average Confidence", f"{np.mean(confs):.2f}")
    else:
        st.info("No detections found")

    if rows:
        df = pd.DataFrame(rows)
        st.subheader("📑 OCR Results")
        st.dataframe(df)

        st.download_button(
            "📥 Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            "detected_plates.csv",
            "text/csv"
        )
