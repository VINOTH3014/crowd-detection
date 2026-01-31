"""
Verify that the entire crowd detection system is working correctly
"""

import sys
import os

def test_imports():
    """Test all imports"""
    print("🔍 Testing imports...")
    
    try:
        import cv2
        print("✅ OpenCV")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO")
    except ImportError as e:
        print(f"❌ Ultralytics YOLO: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    return True

def test_crowd_detector():
    """Test crowd detector initialization"""
    print("\n🤖 Testing CrowdDetector...")
    
    try:
        from crowd_detector import CrowdDetector
        detector = CrowdDetector()
        print("✅ CrowdDetector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ CrowdDetector failed: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\n📹 Testing camera...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print("✅ Camera working")
                return True
            else:
                print("⚠️ Camera detected but not responding")
                return False
        else:
            print("❌ No camera detected")
            return False
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
        return False

def test_yolo_model():
    """Test YOLO model loading"""
    print("\n🧠 Testing YOLO model...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ YOLO model failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Crowd Detection System Verification")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test crowd detector
    if not test_crowd_detector():
        print("\n❌ CrowdDetector test failed")
        return False
    
    # Test YOLO model
    if not test_yolo_model():
        print("\n❌ YOLO model test failed")
        return False
    
    # Test camera
    camera_ok = test_camera()
    
    print("\n" + "=" * 50)
    print("🎉 SYSTEM VERIFICATION COMPLETE!")
    print("=" * 50)
    
    if camera_ok:
        print("✅ Camera: Working")
    else:
        print("⚠️ Camera: Not detected or not working")
    
    print("✅ All core components: Working")
    print("✅ YOLO model: Loaded")
    print("✅ CrowdDetector: Functional")
    
    print("\n📋 Your system is ready to use!")
    print("🌐 Web interface: streamlit run streamlit_app.py")
    print("💻 Command line: python crowd_detector.py")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Verification cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
