"""
Crowd Detection System using YOLOv8
Real-time person detection and crowd monitoring
"""

import os
import sys
from collections import deque
import time

# Fix for PyTorch 2.6+ compatibility before importing torch
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

try:
    import cv2
    import numpy as np
    import torch
    
    # Patch torch.load for compatibility
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    torch.load = patched_load
    
    from ultralytics import YOLO
    import streamlit as st
    from PIL import Image
    import matplotlib.pyplot as plt
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)

class CrowdDetector:
    def __init__(self, model_path='yolov8n.pt', crowd_threshold=5):
        """
        Initialize the crowd detection system
        
        Args:
            model_path (str): Path to YOLO model
            crowd_threshold (int): Maximum number of people before alert
        """
        self.model = YOLO(model_path)
        self.crowd_threshold = crowd_threshold
        self.heatmap_data = deque(maxlen=100)  # Store last 100 frames for heatmap
        self.alert_active = False
        self.alert_start_time = None
        self.alert_sent = False  # Track if alert was sent
        
    def detect_people(self, frame):
        """
        Detect people in a frame using YOLO
        
        Args:
            frame: Input frame (numpy array)
            
        Returns:
            tuple: (annotated_frame, person_count, bounding_boxes)
        """
        # Run YOLO detection
        results = self.model(frame, classes=[0])  # class 0 is 'person' in COCO dataset
        
        person_count = 0
        bounding_boxes = []
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Only consider detections with confidence > 0.5
                    if confidence > 0.5:
                        person_count += 1
                        bounding_boxes.append((x1, y1, x2, y2, confidence))
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Add confidence label
                        label = f"Person {confidence:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated_frame, person_count, bounding_boxes
    
    def update_heatmap(self, frame, bounding_boxes):
        """
        Update heatmap data with current detections
        
        Args:
            frame: Current frame
            bounding_boxes: List of bounding boxes from current detection
        """
        # Create a heatmap layer for this frame
        heatmap_layer = np.zeros(frame.shape[:2], dtype=np.float32)
        
        for x1, y1, x2, y2, confidence in bounding_boxes:
            # Add Gaussian-like heat around each detection
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            radius = max((x2 - x1) // 2, (y2 - y1) // 2)
            
            # Create circular heat pattern
            y, x = np.ogrid[:frame.shape[0], :frame.shape[1]]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            heatmap_layer[mask] += confidence
        
        self.heatmap_data.append(heatmap_layer)
    
    def generate_heatmap(self, frame_shape):
        """
        Generate cumulative heatmap from stored data
        
        Args:
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            numpy array: Heatmap visualization
        """
        if not self.heatmap_data:
            return np.zeros(frame_shape[:2])
        
        # Sum all heatmap layers
        cumulative_heatmap = np.sum(list(self.heatmap_data), axis=0)
        
        # Normalize to 0-1 range
        if cumulative_heatmap.max() > 0:
            cumulative_heatmap = cumulative_heatmap / cumulative_heatmap.max()
        
        return cumulative_heatmap
    
    def check_crowd_threshold(self, person_count):
        """
        Check if crowd threshold is exceeded and manage alerts
        
        Args:
            person_count: Number of people detected
            
        Returns:
            bool: True if alert should be shown
        """
        if person_count > self.crowd_threshold:
            if not self.alert_active:
                self.alert_active = True
                self.alert_start_time = time.time()
            # Alert logic (no SMS)
            return True
        else:
            self.alert_active = False
            self.alert_start_time = None
            return False
    
    def draw_alert(self, frame, person_count):
        """
        Draw alert overlay on frame
        
        Args:
            frame: Input frame
            person_count: Number of people detected
            
        Returns:
            numpy array: Frame with alert overlay
        """
        alert_frame = frame.copy()
        
        if self.alert_active:
            # Create semi-transparent red overlay
            overlay = alert_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            alert_frame = cv2.addWeighted(alert_frame, 0.7, overlay, 0.3, 0)
            
            # Add alert text
            alert_text = f"⚠ OVERCROWDING DETECTED! ({person_count} people)"
            text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            
            cv2.putText(alert_frame, alert_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return alert_frame
    
    def process_frame(self, frame):
        """
        Process a single frame through the complete pipeline
        
        Args:
            frame: Input frame
            
        Returns:
            tuple: (processed_frame, person_count, heatmap)
        """
        # Detect people
        annotated_frame, person_count, bounding_boxes = self.detect_people(frame)
        
        # Update heatmap
        self.update_heatmap(frame, bounding_boxes)
        
        # Check threshold and add alert if needed
        self.check_crowd_threshold(person_count)
        alert_frame = self.draw_alert(annotated_frame, person_count)
        
        # Add person count display
        count_text = f"People Count: {person_count}"
        cv2.putText(alert_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Add threshold info
        threshold_text = f"Threshold: {self.crowd_threshold}"
        cv2.putText(alert_frame, threshold_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Generate heatmap
        heatmap = self.generate_heatmap(frame.shape)
        
        return alert_frame, person_count, heatmap

def main():
    """Main function for testing the crowd detector"""
    detector = CrowdDetector(crowd_threshold=5)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Crowd Detection System Started!")
    print("Press 'q' to quit, 'r' to reset heatmap")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Warning: Failed to read frame from camera. Attempting to reinitialize...")
            cap.release()
            time.sleep(1)  # Wait a moment before retrying
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not reinitialize webcam. Exiting.")
                cv2.destroyAllWindows()
                break
            continue  # Try reading the next frame
        
        # Process frame
        processed_frame, person_count, heatmap = detector.process_frame(frame)
        
        # Display results
        cv2.imshow('Crowd Detection', processed_frame)
        
        # Display heatmap
        if len(detector.heatmap_data) > 0:
            heatmap_display = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
            cv2.imshow('Crowd Heatmap', heatmap_colored)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.heatmap_data.clear()
            print("Heatmap reset!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
