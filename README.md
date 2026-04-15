# Number-Plate-Detection-of-Vehicles
🚘 Automatic Number Plate Detection using YOLOv8

📌 Overview

This project focuses on real-time automatic number plate detection using YOLOv8. It trains a custom dataset with pre-annotated bounding boxes to accurately detect number plates under different:

🚗 Vehicle types & angles

🌞 Lighting conditions (day/night)

🌧️ Weather variations

⚙️ Tech Stack

Framework: Ultralytics YOLOv8

Language: Python 🐍

Libraries: OpenCV, Albumentations, Matplotlib

Environment: Google Colab (GPU-enabled)

📂 Dataset Format yolo_dataset/ │── dataset.yaml │── images/ │ ├── train/ │ └── val/ │── labels/ ├── train/ └── val/

Class → number_plate (single class)

Annotation format → YOLO .txt (class, x_center, y_center, w, h)

📊 Results

✅ Model successfully detects number plates in real-time

✅ Works across different lighting/weather conditions

📁 Best weights saved at:

/content/yolo_training/plate_detector/weights/yolov8n.pt

✅ Conclusion

YOLOv8 proved to be fast, efficient, and accurate for number plate detection.

The system is robust to angle, lighting, and environmental variations.

Future Scope:

🔤 Add OCR (EasyOCR/Tesseract) for text extraction

🌐 Deploy as a Streamlit Web App for real-world demo

About
This project implements an Automatic Number Plate Detection (ANPD) system using the YOLOv8 object detection framework. The goal is to accurately and efficiently detect vehicle license plates in real-time across diverse conditions such as different angles, lighting (day/night), and weather variations.

Resources
 Readme
 Activity
Stars
 1 star
Watchers
 0 watching
Forks
 0 forks
Report repository
Releases
No releases published
Packages
No packages published
Languages
Jupyter Notebook
99.6%
 
Python
0.4%
Footer
© 2025 GitHub, Inc.
Footer navigation
Term
test webhook 2
