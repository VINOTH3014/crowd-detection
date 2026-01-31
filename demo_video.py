"""
Demo script to create sample crowd detection videos for testing
"""
import cv2
import numpy as np
import os
from crowd_detector import CrowdDetector

def create_demo_video():
    """Create a demo video with simulated crowd scenarios"""
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 30  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('demo_crowd_video.mp4', fourcc, fps, (width, height))
    
    print("Creating demo video...")
    
    for frame_num in range(total_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50
        
        # Add some background elements
        cv2.rectangle(frame, (50, 50), (width-50, height-50), (100, 100, 100), 2)
        cv2.putText(frame, "Demo Crowd Scene", (width//2-100, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Simulate different crowd scenarios over time
        time_phase = frame_num / total_frames
        
        if time_phase < 0.2:  # 0-6 seconds: 1-2 people
            num_people = 1 + (frame_num % 2)
        elif time_phase < 0.4:  # 6-12 seconds: 3-4 people
            num_people = 3 + (frame_num % 2)
        elif time_phase < 0.6:  # 12-18 seconds: 5-6 people (threshold)
            num_people = 5 + (frame_num % 2)
        elif time_phase < 0.8:  # 18-24 seconds: 7-8 people (overcrowded)
            num_people = 7 + (frame_num % 2)
        else:  # 24-30 seconds: 2-3 people (back to normal)
            num_people = 2 + (frame_num % 2)
        
        # Draw simulated people as rectangles
        for i in range(num_people):
            # Random positions
            x = 100 + (i * 80) + (frame_num % 20) - 10
            y = 150 + (frame_num % 30) - 15
            
            # Keep within bounds
            x = max(50, min(width-100, x))
            y = max(100, min(height-150, y))
            
            # Draw person (rectangle)
            cv2.rectangle(frame, (x, y), (x+40, y+80), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"People: {num_people}", (10, height-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add threshold indicator
        threshold = 5
        if num_people > threshold:
            cv2.putText(frame, "OVER CROWDED!", (width//2-80, height-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        out.write(frame)
        
        # Progress indicator
        if frame_num % (fps * 5) == 0:  # Every 5 seconds
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}%")
    
    out.release()
    print("Demo video created: demo_crowd_video.mp4")

def test_with_demo_video():
    """Test the crowd detector with the demo video"""
    
    if not os.path.exists('demo_crowd_video.mp4'):
        print("Demo video not found. Creating it first...")
        create_demo_video()
    
    print("Testing crowd detector with demo video...")
    
    # Initialize detector
    detector = CrowdDetector(crowd_threshold=5)
    
    # Open video
    cap = cv2.VideoCapture('demo_crowd_video.mp4')
    
    if not cap.isOpened():
        print("Error: Could not open demo video")
        return
    
    frame_count = 0
    person_counts = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, person_count, heatmap = detector.process_frame(frame)
        person_counts.append(person_count)
        
        # Display results
        cv2.imshow('Demo Video Analysis', processed_frame)
        
        # Show heatmap
        if len(detector.heatmap_data) > 0:
            heatmap_display = (heatmap * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_display, cv2.COLORMAP_JET)
            cv2.imshow('Heatmap', heatmap_colored)
        
        frame_count += 1
        
        # Print statistics every 30 frames
        if frame_count % 30 == 0:
            avg_count = np.mean(person_counts[-30:])
            print(f"Frame {frame_count}: Current={person_count}, Avg(30)={avg_count:.1f}")
        
        # Break on 'q' key
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final statistics
    print(f"\nDemo Analysis Complete!")
    print(f"Total frames processed: {frame_count}")
    print(f"Average people count: {np.mean(person_counts):.1f}")
    print(f"Maximum people count: {max(person_counts)}")
    print(f"Overcrowding events: {sum(1 for count in person_counts if count > 5)}")

if __name__ == "__main__":
    print("Crowd Detection Demo")
    print("===================")
    print("1. Create demo video")
    print("2. Test with demo video")
    print("3. Both")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        create_demo_video()
    elif choice == "2":
        test_with_demo_video()
    elif choice == "3":
        create_demo_video()
        test_with_demo_video()
    else:
        print("Invalid choice. Running both...")
        create_demo_video()
        test_with_demo_video()
