import os
import streamlit as st
import cv2
import numpy as np
import torch

# Fix for PyTorch 2.6+ compatibility
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from crowd_detector import CrowdDetector
import time
import tempfile

# Page configuration
st.set_page_config(
    page_title="Crowd Detection System",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'person_count_history' not in st.session_state:
        st.session_state.person_count_history = []
    if 'alert_history' not in st.session_state:
        st.session_state.alert_history = []

def create_detector(crowd_threshold):
    """Create a new detector instance"""
    try:
        detector = CrowdDetector(crowd_threshold=crowd_threshold)
        return detector, None
    except Exception as e:
        return None, str(e)

def process_uploaded_video(uploaded_file, detector):
    """Process uploaded video file"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        person_counts = []
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame to speed up processing
            if frame_count % 5 == 0:
                processed_frame, person_count, heatmap = detector.process_frame(frame)
                frames.append(processed_frame)
                person_counts.append(person_count)
                
                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            frame_count += 1
        
        cap.release()
        progress_bar.empty()
        status_text.empty()
        
        return frames, person_counts
    
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def main():
    """Main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">👥 Real-Time Crowd Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Crowd threshold setting
    crowd_threshold = st.sidebar.slider(
        "Crowd Threshold (Max People)",
        min_value=1,
        max_value=20,
        value=5,
        help="Alert when number of people exceeds this threshold"
    )
    
    # Model selection
    model_choice = st.sidebar.selectbox(
        "YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="Choose YOLO model size (n=fastest, m=most accurate)"
    )
    
    # Initialize detector button
    if st.sidebar.button("🚀 Initialize Detector", type="primary"):
        with st.spinner("Loading YOLO model..."):
            detector, error = create_detector(crowd_threshold)
            if detector:
                st.session_state.detector = detector
                st.sidebar.success("✅ Detector initialized successfully!")
            else:
                st.sidebar.error(f"❌ Error: {error}")
    
    # Main content area
    if st.session_state.detector is None:
        st.warning("⚠️ Please initialize the detector from the sidebar first.")
        st.info("""
        ## How to use:
        1. **Initialize Detector**: Click the button in the sidebar to load the YOLO model
        2. **Live Camera**: Use your webcam for real-time detection
        3. **Upload Video**: Test with your own video files
        4. **Monitor**: Watch the live count and alerts
        """)
        return
    
    # Tabs for different modes
    tab1, tab2, tab3 = st.tabs(["📹 Live Camera", "📁 Upload Video", "📊 Analytics"])
    
    with tab1:
        st.header("Live Camera Feed")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera controls
            if st.button("🎥 Start Camera", disabled=st.session_state.camera_active):
                st.session_state.camera_active = True
                st.rerun()
            
            if st.button("⏹️ Stop Camera", disabled=not st.session_state.camera_active):
                st.session_state.camera_active = False
                st.rerun()
            
            # Camera feed placeholder
            camera_placeholder = st.empty()
            
            if st.session_state.camera_active:
                # Initialize camera
                cap = cv2.VideoCapture(0)
                
                if not cap.isOpened():
                    st.error("❌ Could not access camera. Please check your camera connection.")
                    st.session_state.camera_active = False
                else:
                    # Process camera feed
                    while st.session_state.camera_active:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process frame
                        processed_frame, person_count, heatmap = st.session_state.detector.process_frame(frame)
                        
                        # Update history
                        st.session_state.person_count_history.append(person_count)
                        if person_count > crowd_threshold:
                            st.session_state.alert_history.append(time.time())
                        
                        # Convert to RGB for display
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        camera_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.1)
                    
                    cap.release()
        
        with col2:
            # Real-time metrics
            if st.session_state.person_count_history:
                current_count = st.session_state.person_count_history[-1]
                
                # Current count display
                st.metric("Current People Count", current_count)
                
                # Alert status
                if current_count > crowd_threshold:
                    st.markdown("""
                    <div class="alert-box">
                        <h3>⚠️ OVERCROWDING ALERT!</h3>
                        <p>People count exceeds threshold of {}</p>
                    </div>
                    """.format(crowd_threshold), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="success-box">
                        <h3>✅ Normal Crowd Level</h3>
                        <p>People count within safe limits</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Statistics
                st.subheader("📈 Statistics")
                avg_count = np.mean(st.session_state.person_count_history[-10:])  # Last 10 readings
                max_count = max(st.session_state.person_count_history)
                alert_count = len([t for t in st.session_state.alert_history if time.time() - t < 60])  # Alerts in last minute
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Average (Last 10)", f"{avg_count:.1f}")
                    st.metric("Maximum", max_count)
                with col_b:
                    st.metric("Alerts (Last min)", alert_count)
    
    with tab2:
        st.header("Upload Video for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze crowd density"
        )
        
        if uploaded_file is not None:
            st.info("📹 Processing uploaded video...")
            
            # Process video
            frames, person_counts = process_uploaded_video(uploaded_file, st.session_state.detector)
            
            if frames:
                st.success(f"✅ Processed {len(frames)} frames")
                
                # Video analysis results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Video Analysis")
                    st.write(f"**Total Frames Processed:** {len(frames)}")
                    st.write(f"**Average People Count:** {np.mean(person_counts):.1f}")
                    st.write(f"**Maximum People Count:** {max(person_counts)}")
                    st.write(f"**Overcrowding Events:** {sum(1 for count in person_counts if count > crowd_threshold)}")
                
                with col2:
                    # People count over time
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(person_counts, color='blue', linewidth=2)
                    ax.axhline(y=crowd_threshold, color='red', linestyle='--', label=f'Threshold ({crowd_threshold})')
                    ax.set_xlabel('Frame Number')
                    ax.set_ylabel('People Count')
                    ax.set_title('People Count Over Time')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                
                # Sample frames
                st.subheader("🎬 Sample Frames")
                sample_indices = np.linspace(0, len(frames)-1, min(6, len(frames)), dtype=int)
                
                cols = st.columns(3)
                for i, idx in enumerate(sample_indices):
                    with cols[i % 3]:
                        frame_rgb = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                        st.image(frame_rgb, caption=f"Frame {idx} - {person_counts[idx]} people", use_column_width=True)
    
    with tab3:
        st.header("Analytics Dashboard")
        
        if st.session_state.person_count_history:
            # Historical data visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # People count trend
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(st.session_state.person_count_history, color='blue', alpha=0.7)
                ax.axhline(y=crowd_threshold, color='red', linestyle='--', label=f'Threshold ({crowd_threshold})')
                ax.fill_between(range(len(st.session_state.person_count_history)), 
                               st.session_state.person_count_history, 
                               alpha=0.3, color='blue')
                ax.set_xlabel('Time (frames)')
                ax.set_ylabel('People Count')
                ax.set_title('Real-time People Count Trend')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Distribution histogram
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(st.session_state.person_count_history, bins=20, alpha=0.7, color='green', edgecolor='black')
                ax.axvline(x=crowd_threshold, color='red', linestyle='--', label=f'Threshold ({crowd_threshold})')
                ax.set_xlabel('People Count')
                ax.set_ylabel('Frequency')
                ax.set_title('People Count Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Summary statistics
            st.subheader("📈 Summary Statistics")
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Total Readings", len(st.session_state.person_count_history))
            
            with stats_col2:
                st.metric("Average Count", f"{np.mean(st.session_state.person_count_history):.1f}")
            
            with stats_col3:
                st.metric("Peak Count", max(st.session_state.person_count_history))
            
            with stats_col4:
                alert_percentage = (sum(1 for count in st.session_state.person_count_history if count > crowd_threshold) / 
                                  len(st.session_state.person_count_history)) * 100
                st.metric("Alert Rate", f"{alert_percentage:.1f}%")
            
            # Clear data button
            if st.button("🗑️ Clear All Data"):
                st.session_state.person_count_history = []
                st.session_state.alert_history = []
                st.rerun()
        
        else:
            st.info("📊 No data available yet. Start the camera or upload a video to see analytics.")

if __name__ == "__main__":
    main()
