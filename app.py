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
from streamlit_lottie import st_lottie
import json

# -------------------------------------------------------
# PAGE SETUP
# -------------------------------------------------------
st.set_page_config(page_title="Number Plate Detection", page_icon="🚘", layout="wide")
# --- Try to load car animation at the very top (above title) ---
def load_lottie(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

car_anim = load_lottie("car.json")
scan_anim = load_lottie("scan.json")

# Show car animation above the title if available, otherwise small emoji title
if car_anim:
    st_lottie(car_anim, height=180, key="car_top")
    st.title("🚘 Number Plate Detection of Vehicles (YOLO + EasyOCR)")
else:
    st.title("🚘 Number Plate Detection of Vehicles (YOLO + EasyOCR)")

# -------------------------------------------------------
# SIDEBAR: Controls + Scanning Animation (recommended)
# -------------------------------------------------------
st.sidebar.header("⚙️ Controls")

# show scan animation in sidebar (recommended)
if scan_anim:
    st.sidebar.subheader("🔍 Scanning Animation")
    st_lottie(scan_anim, height=140, key="scan_side")
else:
    st.sidebar.subheader("🔍 Scanning")
    st.sidebar.write("Scanning animation not found. Add `scan.json` to the project folder.")

# -------------------------------------------------------
# AUTO-ROTATION FIX
# -------------------------------------------------------
def fix_rotation(img: Image.Image) -> Image.Image:
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif is not None:
            orientation_val = exif.get(orientation)
            if orientation_val == 3:
                img = img.rotate(180, expand=True)
            elif orientation_val == 6:
                img = img.rotate(270, expand=True)
            elif orientation_val == 8:
                img = img.rotate(90, expand=True)
    except Exception:
        pass
    return img

# -------------------------------------------------------
# MODEL & OCR LOADING (cached)
# -------------------------------------------------------
@st.cache_resource
def load_model(weights_path="best.pt"):
    if os.path.exists(weights_path):
        st.sidebar.success("✅ Loaded custom YOLO model (best.pt)")
        return YOLO(weights_path)
    else:
        st.sidebar.warning("⚠ best.pt not found. Loading YOLOv8n (default).")
        return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr(lang_list=None):
    if lang_list is None:
        lang_list = ['en']
    return easyocr.Reader(lang_list, gpu=False)  # set gpu=True if available

model = load_model()
reader = load_ocr()

# -------------------------------------------------------
# OCR & PREPROCESS HELPERS
# -------------------------------------------------------
def preprocess_plate(crop_bgr):
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text(img_gray):
    # reader.readtext accepts numpy arrays as grayscale or color
    try:
        texts = reader.readtext(img_gray, detail=0)
        return " ".join(t.strip() for t in texts if t.strip())
    except Exception:
        return ""

def clean_text(t):
    if not t:
        return ""
    t = t.upper().replace(" ", "")
    # common substitutions to reduce OCR mistakes
    t = t.replace('O', '0').replace('I', '1').replace('Z', '2').replace('S', '5')
    return t

# -------------------------------------------------------
# DETECTION FUNCTION
# -------------------------------------------------------
def detect_plates(img_bgr, conf_thr=0.1, min_w=20, min_h=10):
    start = time.time()
    results = model(img_bgr, conf=conf_thr)
    elapsed = time.time() - start

    overlay = results[0].plot().astype(np.uint8)
    rows = []
    confs = []
    crops = []

    if not hasattr(results[0], "boxes") or results[0].boxes is None:
        return overlay, rows, elapsed, confs, crops

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scores = results[0].boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1

        if w < min_w or h < min_h:
            continue

        # ensure indexes are inside image bounds
        y1c, y2c = max(0, y1), max(0, min(overlay.shape[0], y2))
        x1c, x2c = max(0, x1), max(0, min(overlay.shape[1], x2))

        crop = img_bgr[y1c:y2c, x1c:x2c].copy()
        if crop.size == 0:
            continue
        crops.append(crop)

        processed = preprocess_plate(crop)
        raw = extract_text(processed)
        plate_text = clean_text(raw)

        # annotate overlay
        cv2.putText(
            overlay,
            f"{plate_text} ({score:.2f})",
            (x1c, max(15, y1c - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        rows.append({"Plate No": plate_text, "Confidence": round(float(score), 2)})
        confs.append(float(score))

    return overlay, rows, elapsed, confs, crops

# -------------------------------------------------------
# SIDEBAR PARAMETERS
# -------------------------------------------------------
conf_thr = st.sidebar.slider("Detection confidence", 0.01, 0.90, 0.10, 0.01)
min_w = st.sidebar.number_input("Min plate width (px)", 5, 2000, 20, 5)
min_h = st.sidebar.number_input("Min plate height (px)", 5, 2000, 10, 5)

# Optionally add logo in sidebar (uncomment if you have logo.png)
# if os.path.exists("logo.png"):
#     st.sidebar.image("logo.png", width=160)

# -------------------------------------------------------
# MAIN UI: Tabs - Upload / Webcam
# -------------------------------------------------------
tab1, tab2 = st.tabs(["📤 Upload Image", "📸 Webcam Capture"])

with tab1:
    uploaded = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img = fix_rotation(img)
        img_bgr = np.array(img)[:, :, ::-1]

        # show scanning animation while detecting (optional): spinner + message
        with st.spinner("Detecting number plates..."):
            overlay, rows, elapsed, confs, crops = detect_plates(img_bgr, conf_thr, min_w, min_h)

        st.image(overlay[:, :, ::-1], caption="Detections", use_container_width=True)
        st.write(f"Processing Time: **{elapsed:.3f} s**")
        st.write(f"Average Confidence: **{np.mean(confs):.2f}**" if confs else "Average Confidence: **N/A**")

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("📑 OCR Results")
            st.dataframe(df)
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), "detected_upload.csv", "text/csv")

        # show cropped plates
        if crops:
            st.subheader("🔍 Cropped Plates")
            # display side-by-side using columns
            cols = st.columns(min(4, len(crops)))
            for i, crop in enumerate(crops):
                col = cols[i % len(cols)]
                col.image(crop[:, :, ::-1], caption=f"Plate {i+1}", use_column_width=True)

with tab2:
    webcam_img = st.camera_input("Take a picture")
    if webcam_img:
        img = Image.open(webcam_img).convert("RGB")
        img = fix_rotation(img)
        img_bgr = np.array(img)[:, :, ::-1]

        # Optionally show a smaller scanning animation in main area while detecting
        with st.spinner("Detecting number plates..."):
            overlay, rows, elapsed, confs, crops = detect_plates(img_bgr, conf_thr, min_w, min_h)

        st.image(overlay[:, :, ::-1], caption="Webcam Detections", use_container_width=True)
        st.write(f"Processing Time: **{elapsed:.3f} s**")
        st.write(f"Average Confidence: **{np.mean(confs):.2f}**" if confs else "Average Confidence: **N/A**")

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("📑 OCR Results")
            st.dataframe(df)
            st.download_button("📥 Download CSV", df.to_csv(index=False).encode("utf-8"), "detected_webcam.csv", "text/csv")

        if crops:
            st.subheader("🔍 Cropped Plates")
            cols = st.columns(min(4, len(crops)))
            for i, crop in enumerate(crops):
                col = cols[i % len(cols)]
                col.image(crop[:, :, ::-1], caption=f"Plate {i+1}", use_column_width=True)

# -------------------------------------------------------
# FOOTER / HELP
# -------------------------------------------------------
st.markdown("---")
st.write("Tip: Put `best.pt` in the project folder to use your trained model. Add `car.json` and `scan.json` for animations.")
