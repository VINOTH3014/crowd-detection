"""
Installation script for the Crowd Detection System
"""
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing required packages...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_camera():
    """Check if camera is available"""
    print("📹 Checking camera availability...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera is working!")
                return True
            else:
                print("⚠️ Camera detected but not responding")
                return False
        else:
            print("❌ No camera detected")
            return False
    except ImportError:
        print("❌ OpenCV not installed yet")
        return False
    except Exception as e:
        print(f"❌ Camera check failed: {e}")
        return False

def download_yolo_model():
    """Download YOLO model"""
    print("🤖 Downloading YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading YOLO model: {e}")
        return False

def test_installation():
    """Test the installation"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import streamlit as st
        from PIL import Image
        import matplotlib.pyplot as plt
        
        print("✅ All imports successful!")
        
        # Test YOLO model
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully!")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🚀 Crowd Detection System Installation")
    print("=" * 40)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Installation failed at package installation step")
        return False
    
    # Step 2: Download YOLO model
    if not download_yolo_model():
        print("❌ Installation failed at YOLO model download step")
        return False
    
    # Step 3: Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    # Step 4: Check camera
    camera_ok = check_camera()
    
    print("\n" + "=" * 40)
    print("🎉 Installation Complete!")
    print("=" * 40)
    
    if camera_ok:
        print("✅ Camera: Working")
    else:
        print("⚠️ Camera: Not detected or not working")
    
    print("\n📖 Next Steps:")
    print("1. Run the Streamlit app: streamlit run streamlit_app.py")
    print("2. Or run the command line version: python crowd_detector.py")
    print("3. For demo videos: python demo_video.py")
    
    print("\n🌐 Web Interface:")
    print("   Open your browser to: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Installation cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
