import os
import re
import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import easyocr

st.set_page_config(page_title="Number Plate Detection of Vehicles", page_icon="🚘", layout="wide")
st.title("🚘 Number Plate Detection of Vehicles (YOLO + EasyOCR)")

@st.cache_resource
def load_model():
    
    weights_path = "best.pt"
    if os.path.exists(weights_path):
        st.success("✅ Loaded custom YOLO model (best.pt)")
        return YOLO(weights_path)
    st.warning("⚠️ best.pt not found. Loading YOLOv8n pretrained weights…")
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

model = load_model()
reader = load_ocr()

def preprocess_plate(crop_bgr: np.ndarray) -> np.ndarray:
    """Grayscale → Otsu threshold → CLAHE."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(thresh)
    return enhanced

def easy_ocr_text(img_gray: np.ndarray) -> str:
    """Run EasyOCR and combine results."""
    texts = reader.readtext(img_gray, detail=0)
    return " ".join(t.strip() for t in texts if t and t.strip())

INDIA_PLATE_REGEX = re.compile(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,2}[0-9]{3,4}$')

def clean_plate_text(raw_text: str) -> str:
    """
    Normalize common OCR confusions and validate Indian plate pattern.
    """
    if not raw_text:
        return ""
    t = raw_text.upper()
    t = re.sub(r'\s+', '', t)
    
    t = t.replace('O', '0')
    t = t.replace('I', '1')
    t = t.replace('Z', '2')
    t = t.replace('S', '5')

    return t if INDIA_PLATE_REGEX.match(t) else t

def draw_label(img, text, x1, y1):
    """Draws a filled label above the bbox."""
    if not text:
        return
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    y = max(0, y1 - th - 6)
    cv2.rectangle(img, (x1, y), (x1 + tw + 8, y + th + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x1 + 4, y + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

def detect_plates(image_bgr: np.ndarray, conf_thr: float, min_w: int, min_h: int):
    """
    Run YOLO, return overlay image and a list of results dicts:
    {Plate No., Confidence, x1, y1, x2, y2}
    """
    t0 = time.time()
    results = model(image_bgr, conf=conf_thr)
    elapsed = time.time() - t0

    overlay = results[0].plot()  
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes is not None else np.empty((0, 4))
    confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else np.array([])

    rows = []
    for (x1, y1, x2, y2), c in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        w, h = x2 - x1, y2 - y1
        if w < min_w or h < min_h:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        enhanced = preprocess_plate(crop)
        raw = easy_ocr_text(enhanced)
        text = clean_plate_text(raw)

        draw_label(overlay, f"{text} ({c:.2f})" if text else f"{c:.2f}", x1, y1)

        rows.append({
            "Plate No.": text if text else raw if raw else "",
            "Confidence": round(float(c), 2),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "Processing Time (s)": round(elapsed, 3)
        })

    return overlay, rows, elapsed, confs

if "webcam_logs" not in st.session_state:
    st.session_state.webcam_logs = []
if "run_webcam" not in st.session_state:
    st.session_state.run_webcam = False

st.sidebar.header("Controls")
mode = st.sidebar.radio("Mode", ["Upload Image", "Webcam (real-time)"])
conf_thr = st.sidebar.slider("Detection confidence", 0.10, 0.90, 0.30, 0.05)
min_w = st.sidebar.number_input("Min plate width (px)", 20, 2000, 80, 10)
min_h = st.sidebar.number_input("Min plate height (px)", 10, 2000, 30, 10)

if mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload Vehicle Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]  # to BGR for OpenCV/YOLO

        overlay, rows, elapsed, confs = detect_plates(img_np, conf_thr, min_w, min_h)

        st.image(overlay[:, :, ::-1], caption="Detections", use_container_width=True)

        if rows:
            df = pd.DataFrame(rows)
            st.subheader("📑 OCR Results")
            st.dataframe(df, width="stretch")

            st.download_button(
                "📥 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="detected_plates.csv",
                mime="text/csv"
            )

        st.subheader("📊 Performance")
        st.metric("Processing Time (s)", f"{elapsed:.3f}")
        if len(confs):
            st.metric("Average Confidence", f"{np.mean(confs):.2f}")
        else:
            st.info("No detections to compute average confidence.")

else:
    left, right = st.columns([2, 1])
    video_area = left.empty()
    right.subheader("📝 Live OCR Log")

    start_btn = right.button("▶ Start Webcam", disabled=st.session_state.run_webcam)
    stop_btn = right.button("⏹ Stop Webcam", disabled=not st.session_state.run_webcam)
    clear_btn = right.button("🧹 Clear Log")

    if clear_btn:
        st.session_state.webcam_logs = []

    if start_btn:
        st.session_state.run_webcam = True
    if stop_btn:
        st.session_state.run_webcam = False

    FRAME_SKIP = 3
    frame_idx = 0

    if st.session_state.run_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        try:
            while st.session_state.run_webcam:
                ok, frame = cap.read()
                if not ok:
                    left.error("❌ Failed to access webcam")
                    break

                if frame_idx % FRAME_SKIP == 0:
                    overlay, rows, elapsed, confs = detect_plates(frame, conf_thr, min_w, min_h)
                else:

                    results = model(frame, conf=conf_thr)
                    overlay = results[0].plot()
                    rows = []

                video_area.image(overlay[:, :, ::-1], use_column_width=True)

                for r in rows:
                    st.session_state.webcam_logs.append({
                        "Timestamp": time.strftime("%H:%M:%S"),
                        "Plate No.": r["Plate No."],
                        "Confidence": r["Confidence"]
                    })

             
                if st.session_state.webcam_logs:
                    df_live = pd.DataFrame(st.session_state.webcam_logs[-300:]) 
                    right.dataframe(df_live, use_container_width=True, height=380)
                    right.download_button(
                        "📥 Download Log (CSV)",
                        df_live.to_csv(index=False).encode("utf-8"),
                        file_name="webcam_ocr_log.csv",
                        mime="text/csv"
                    )

                frame_idx += 1
                time.sleep(0.01)

        finally:
            cap.release()
            cv2.destroyAllWindows()
            right.info("Webcam stopped.")
