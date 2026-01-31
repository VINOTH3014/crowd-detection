"""
Test script to verify all imports work correctly
"""

print("Testing imports...")

try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import failed: {e}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

try:
    from ultralytics import YOLO
    print("✅ Ultralytics YOLO imported successfully")
except ImportError as e:
    print(f"❌ Ultralytics YOLO import failed: {e}")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    from PIL import Image
    print("✅ PIL imported successfully")
except ImportError as e:
    print(f"❌ PIL import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import failed: {e}")

try:
    import seaborn as sns
    print("✅ Seaborn imported successfully")
except ImportError as e:
    print(f"❌ Seaborn import failed: {e}")

print("\n🎉 All imports completed!")
