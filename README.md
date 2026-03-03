# 👥 Real-Time Crowd Detection & Density Monitoring System

An AI-powered crowd monitoring system built using YOLOv8, OpenCV, and Streamlit that performs real-time person detection, crowd density estimation, heatmap visualization, and intelligent alert generation.

---

## 🎯 Problem Statement

Overcrowding in public places such as malls, railway stations, stadiums, and events can lead to safety risks and poor crowd management. Manual monitoring is inefficient and unreliable.

This system provides an automated, real-time AI-based solution for detecting crowd density and triggering alerts when predefined thresholds are exceeded.

---

## 🚀 Key Features

- 🎥 **Real-time Person Detection** using YOLOv8
- 📊 **Dynamic Crowd Density Monitoring**
- 🚨 **Threshold-Based Overcrowding Alerts**
- 🔴 **Visual Warning Overlay System**
- 🌡️ **Heatmap Visualization for Crowd Patterns**
- 📂 **Video File Upload & Analysis**
- 📈 **Analytics Dashboard with Historical Insights**
- 🌐 **Interactive Web Interface (Streamlit)**

---

## 🧠 Technical Highlights

- Implemented YOLOv8 deep learning model for accurate person detection
- Optimized frame-by-frame real-time video processing
- Designed modular `CrowdDetector` class architecture
- Integrated cumulative heatmap generation algorithm
- Built interactive analytics dashboard using Streamlit
- Implemented configurable crowd threshold logic
- Performance-tuned for low-latency detection

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Computer Vision:** OpenCV
- **Deep Learning Model:** YOLOv8
- **Web Framework:** Streamlit
- **Libraries:** NumPy, Ultralytics YOLO
- **Deployment Ready:** Local / Cloud compatible

---

## 📋 System Requirements

- Python 3.8+
- Webcam or video input source
- 4GB+ RAM (Recommended)
- Internet connection (First-time YOLO model download)

---

## 🏗️ System Architecture

1. Video Input (Webcam / Uploaded Video)
2. Frame Extraction
3. YOLOv8 Person Detection
4. Person Counting
5. Density Calculation
6. Alert Trigger System
7. Heatmap Update
8. Dashboard Visualization

---

## 🛠️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/VINOTH3014/crowd-detection.git

2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Model Download

YOLOv8 model downloads automatically on first run.

▶️ Quick Start
🌐 Option 1: Streamlit Web Interface (Recommended)
streamlit run streamlit_app.py

Open browser:

http://localhost:8501
💻 Option 2: Command Line Interface
python crowd_detector.py

Controls:

Press q → Quit

Press r → Reset heatmap

⚙️ Configuration
Crowd Threshold

Default: 5 people

Adjustable via Streamlit sidebar

Triggers alert when exceeded

YOLO Model Options
Model	Speed	Accuracy
yolov8n.pt	Fastest	Lower
yolov8s.pt	Balanced	Medium
yolov8m.pt	Slower	High
Detection Parameters

Confidence Threshold: 0.5

Person Class Only (COCO Class ID 0)

Real-time optimized pipeline

📊 Core Components
1️⃣ crowd_detector.py

YOLO model loading

Person detection & counting

Bounding box rendering

Heatmap generation

Alert logic handling

2️⃣ streamlit_app.py

Web dashboard UI

Live webcam streaming

Video upload processing

Real-time metrics display

Historical data analytics

📸 Sample Output

(Add screenshots here)

Live bounding box detection

Overcrowding red alert overlay

Heatmap density visualization

Analytics dashboard view

🔧 Troubleshooting

Camera Not Detected

Check permissions

Close other camera apps

Try different camera index (0,1,2...)

Model Download Fails

Check internet connection

Manual download: yolo download yolov8n.pt

Performance Issues

Use smaller model (yolov8n.pt)

Reduce resolution

Close background applications

📁 Project Structure
project/
├── crowd_detector.py
├── streamlit_app.py
├── requirements.txt
└── README.md
🔮 Future Enhancements

Multi-camera support

Cloud deployment

Email / SMS alerts

Database logging

AI-based anomaly detection

IoT integration

Advanced predictive analytics

📄 License

This project is licensed under the MIT License.

👨‍💻 Author

Vinoth N
Aspiring Software Developer | Computer Vision Enthusiast
Chennai, India

⭐ If you found this project useful, consider giving it a star!

