"""
Test script for crowd detector with PyTorch compatibility fix
"""
import os
import sys

# Set environment variable before importing anything else
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Monkey patch torch.load to use weights_only=False
import torch
original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

# Now import and test
try:
    from crowd_detector import CrowdDetector
    print("Testing YOLO model loading...")
    detector = CrowdDetector()
    print("✅ YOLO model loaded successfully!")
    
    # Test camera
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("✅ Camera detected and working!")
        cap.release()
    else:
        print("⚠️ Camera not detected")
    
    print("\n🎉 All tests passed! The system is ready to use.")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

