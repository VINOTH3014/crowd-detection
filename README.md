# 👥 Real-Time Crowd Detection System

A comprehensive crowd detection and monitoring system using YOLOv8, OpenCV, and Streamlit. This system provides real-time person detection, crowd density monitoring, and alert capabilities.

## 🚀 Features

- **Real-time Detection**: Live webcam feed with person detection
- **Crowd Monitoring**: Configurable threshold-based alerting
- **Visual Alerts**: Overcrowding warnings with visual indicators
- **Heatmap Visualization**: Track crowded areas over time
- **Video Analysis**: Upload and analyze video files
- **Analytics Dashboard**: Historical data and statistics
- **Web Interface**: User-friendly Streamlit dashboard

## 📋 Requirements

- Python 3.8+
- Webcam or video input source
- 4GB+ RAM recommended

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLO model (automatic on first run):**
   The system will automatically download the YOLOv8 model on first use.

## 🎯 Quick Start

### Option 1: Streamlit Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Option 2: Command Line Interface

```bash
python crowd_detector.py
```

## 📖 Usage Guide

### Streamlit Dashboard

1. **Initialize Detector**: Click "🚀 Initialize Detector" in the sidebar
2. **Configure Settings**: 
   - Set crowd threshold (default: 5 people)
   - Choose YOLO model size
3. **Live Camera**: 
   - Click "🎥 Start Camera" to begin real-time detection
   - Monitor person count and alerts
   - View heatmap visualization
4. **Upload Video**: 
   - Upload video files for analysis
   - View frame-by-frame results
   - Analyze crowd patterns over time
5. **Analytics**: 
   - View historical trends
   - Check summary statistics
   - Monitor alert rates

### Command Line Interface

- **Start detection**: Run `python crowd_detector.py`
- **Controls**:
  - Press `q` to quit
  - Press `r` to reset heatmap
- **Display**: Shows live feed with bounding boxes and alerts

## ⚙️ Configuration

### Crowd Threshold
- Default: 5 people
- Adjustable in Streamlit sidebar
- Triggers visual alert when exceeded

### YOLO Models
- **yolov8n.pt**: Fastest, lower accuracy
- **yolov8s.pt**: Balanced speed/accuracy
- **yolov8m.pt**: Most accurate, slower

### Detection Parameters
- Confidence threshold: 0.5 (hardcoded)
- Person class only (COCO class 0)
- Real-time processing optimized

## 📊 System Components

### 1. CrowdDetector Class (`crowd_detector.py`)
- YOLO model integration
- Person detection and counting
- Bounding box visualization
- Heatmap generation
- Alert management

### 2. Streamlit App (`streamlit_app.py`)
- Web-based interface
- Live camera feed
- Video upload and analysis
- Analytics dashboard
- Real-time metrics

### 3. Key Features
- **Real-time Processing**: Optimized for live video streams
- **Visual Alerts**: Red overlay when threshold exceeded
- **Heatmap Tracking**: Cumulative crowd density visualization
- **Historical Data**: Track trends and patterns
- **Multi-format Support**: Webcam, video files, various formats

## 🔧 Troubleshooting

### Common Issues

1. **Camera not detected**:
   - Check camera permissions
   - Ensure no other applications are using the camera
   - Try different camera index (0, 1, 2...)

2. **Model download fails**:
   - Check internet connection
   - YOLO models download automatically on first use
   - Manual download: `yolo download yolov8n.pt`

3. **Performance issues**:
   - Use smaller YOLO model (yolov8n.pt)
   - Reduce video resolution
   - Close other applications

4. **Import errors**:
   - Ensure all dependencies installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Tips

- **For better accuracy**: Use yolov8m.pt model
- **For faster processing**: Use yolov8n.pt model
- **For large videos**: Process every Nth frame
- **For real-time**: Use webcam instead of video files

## 📁 File Structure

```
project/
├── crowd_detector.py      # Core detection logic
├── streamlit_app.py       # Web interface
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎥 Demo Scenarios

### Testing with Webcam
1. Position 1-2 people in front of camera
2. Gradually increase to 6+ people
3. Observe alert activation
4. Monitor heatmap development

### Testing with Videos
1. Upload crowd videos (concerts, events, etc.)
2. Analyze frame-by-frame results
3. Check analytics for patterns
4. Verify threshold detection

## 🔮 Future Enhancements

- [ ] IoT sensor integration
- [ ] Database logging
- [ ] Email/SMS alerts
- [ ] Multi-camera support
- [ ] Advanced analytics
- [ ] Mobile app interface
- [ ] Cloud deployment

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Test with different video sources

---

**Happy Crowd Monitoring! 👥📊**
