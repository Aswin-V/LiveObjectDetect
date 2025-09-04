# --- Installation ---
# Before running, please install the required libraries by running the following command in your terminal:
# pip install streamlit opencv-python-headless requests numpy Pillow ultralytics deepface python-dotenv
import streamlit as st
import cv2
import tempfile
import time
import logging
import os
from dotenv import load_dotenv

from core import (AppController, YOLOConfig, create_yolo_analyzer_params, shutdown_thread_pool)

# --- Register shutdown hook ---
atexit.register(shutdown_thread_pool)

# --- Load Environment Variables ---
# Load environment variables from a .env file if it exists.
# This is useful for managing API keys and other secrets.
load_dotenv()

# --- Logging Configuration ---
# Configure logging to display the time, log level, and message.
# This will output to the console where streamlit is running.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- AppController Initialization ---
if 'controller' not in st.session_state:
    st.session_state.controller = AppController()

controller = st.session_state.controller

# --- UI Setup ---
# Configure the Streamlit page with a title and wide layout for better video display.
st.set_page_config(page_title="AI Video Analysis", layout="wide")
logging.info("Application started and page configured.")

# Display the main title of the application.
st.title("🤖 AI Video Analysis with Python & Streamlit")

# --- Sidebar for Configuration ---
# Create a title for the sidebar section.
st.sidebar.title("Configuration")

# Create a dropdown menu (selectbox) in the sidebar for choosing the analysis model.
model_selection = st.sidebar.selectbox(
    "Choose the analysis model",
    ("Gemini", "YOLO", "DeepFace")
)

st.sidebar.markdown("---")

# --- Model-specific Configurations ---
# Initialize configuration variables with default values to prevent NameError.
api_key_input = ""
confidence_threshold = 0.25
yolo_model_name = "yolov8n.pt"
yolo_task = "detect"
yolo_version = "v8"
yolo_size = "n"

# Display different UI elements in the sidebar based on the selected model.
if model_selection == "Gemini":
    # Get the API key from environment variables, with a fallback to an empty string.
    default_api_key = os.getenv("GEMINI_API_KEY", "")

    # Add a password input field for the Gemini API Key for security.
    api_key_input = st.sidebar.text_input(
        "Gemini API Key", 
        value=st.session_state.get("gemini_api_key", default_api_key),
        type="password",
        help="You can get your key from Google AI Studio. For convenience, you can also set it as GEMINI_API_KEY in a .env file."
    )
    if api_key_input:
        st.session_state.gemini_api_key = api_key_input

    # Display informational text about the Gemini model's capabilities and requirements.
    st.sidebar.info("""
    **Gemini Model:**
    - Cloud-based (requires internet & API Key).
    - Detects objects, human emotions, and activities.
    - Slower due to API calls.
    """)
elif model_selection == "YOLO":
    yolo_version = st.sidebar.selectbox("YOLO Version", YOLOConfig.VERSIONS, help="Choose the YOLO architecture.")
    
    # --- YOLO Task Selection ---
    valid_tasks = YOLOConfig.VALID_TASKS_BY_VERSION.get(yolo_version, [])
    task_index = valid_tasks.index(st.session_state.yolo_task) if st.session_state.get('yolo_task') in valid_tasks else 0
    yolo_task = st.sidebar.selectbox("Task", valid_tasks, index=task_index, help="Choose the task for the YOLO model.")
    st.session_state.yolo_task = yolo_task
    
    # --- YOLO Size Selection ---
    valid_sizes = YOLOConfig.VALID_SIZES_BY_VERSION.get(yolo_version, [])
    size_index = valid_sizes.index(st.session_state.yolo_size) if st.session_state.get('yolo_size') in valid_sizes else 0
    yolo_size = st.sidebar.selectbox("Model Size", valid_sizes, index=size_index, help="Nano is fastest, X is most accurate.")
    st.session_state.yolo_size = yolo_size

    # Add a slider to control the confidence threshold for YOLO detections.
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
        help="Adjust to show more or fewer detections."
    )
    model_display_name = f"YOLO-{yolo_task.capitalize()}" if yolo_task != "detect" else "YOLO"
    # Display informational text about the YOLO model.
    st.sidebar.info(f"""
    **{model_selection} Model:**
    - **Selected:** `{model_display_name}`
    - Runs locally (very fast).
    - Detects a wide range of objects.
    - Does **not** detect emotions or activities.
    - The model file will be downloaded automatically on the first run.
    """)
elif model_selection == "DeepFace":
    st.sidebar.info("""
    **DeepFace Model:**
    - Runs locally.
    - Detects faces and analyzes age, gender, and ethnicity.
    - The model files for face detection and analysis will be downloaded automatically on the first run.
    """)

# --- Analyzer Setup ---
analyzer_params = {}
model_type_to_set = model_selection

if model_selection == "YOLO":
    analyzer_params = create_yolo_analyzer_params(
        version=yolo_version,
        size=yolo_size,
        task=yolo_task,
        confidence=confidence_threshold
    )
elif model_selection == "Gemini":
    analyzer_params = {"api_key": st.session_state.get("gemini_api_key")}
elif model_selection == "DeepFace":
    pass # No params needed

analyzer_ready = controller.set_analyzer(model_selection, **analyzer_params)

if model_selection == "Gemini" and not analyzer_ready:
    st.warning("Please enter your Gemini API Key in the sidebar to proceed.")

def process_image_st(image_file):
    """
    Processes a single uploaded image file for Streamlit UI.
    """
    st.markdown("### Analyzed Image")
    image_placeholder = st.empty()
    results_placeholder = st.empty()

    with st.spinner("Analyzing image..."):
        raw_frame, annotated_frame, analysis = controller.process_single_image(image_file)
        
        if "error" in analysis:
            image_placeholder.empty()
            results_placeholder.error(f"An error occurred during analysis: {analysis['error']}")
            return

        if annotated_frame is not None:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_frame, caption="Analyzed Image", use_container_width=True)
        
        with results_placeholder.container():
            st.subheader("Analysis Results")
            st.json(analysis)
    st.success("Image analysis complete!")

# --- Main Application Logic ---

st.markdown("---")

# --- Input Source Selection and State Reset ---
if 'current_input_source' not in st.session_state:
    st.session_state.current_input_source = "Upload an image file"

input_source = st.radio(
    "Select Input Source",
    ("Upload an image file", "Upload a video file", "Use live webcam feed"),
    horizontal=True,
    key="input_source_radio"
)

if input_source != st.session_state.current_input_source:
    controller.stop_processing()
    st.session_state.current_input_source = input_source
    # Clear video-specific session state
    st.session_state.video_file_path = None
    st.session_state.uploaded_video_id = None
    st.rerun()

logging.info(f"Input source selected: {input_source}")

# --- Session State for UI only ---
if 'video_file_path' not in st.session_state:
    st.session_state.video_file_path = None
if 'uploaded_video_id' not in st.session_state:
    st.session_state.uploaded_video_id = None

@st.fragment
def video_playback_and_processing():
    """
    This fragment contains the main video processing loop.
    It runs a `while` loop to continuously update video frames without
    rerunning the entire Streamlit app, providing a smooth playback experience.
    """
    # Create placeholders INSIDE the fragment. This is key to isolating updates.
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### {'Live Feed' if controller.is_live else 'Video Playback'}")
        playback_placeholder = st.empty()
    with col2:
        st.markdown("### Processed Feed")
        processed_placeholder = st.empty()
    
    results_placeholder = st.empty()
    
    sleep_duration = controller.get_sleep_duration()

    # The main loop. This will run inside the fragment, updating placeholders.
    while controller.processing_state in ('running', 'paused'):
        # If the user clicks Start/Pause/Stop, the main script reruns,
        # breaking this loop. The fragment will then be re-executed
        # with the new state.
        if not controller.process_next_frame():
            break

        raw_frame, annotated_frame = controller.get_display_frames()

        # --- Display Section ---
        if raw_frame is not None:
            playback_placeholder.image(raw_frame, use_container_width=True)
        
        if annotated_frame is not None:
            caption = f"Analyzed Frame: {controller.frame_count}" if not controller.is_live else "Last Analyzed Frame"
            processed_placeholder.image(annotated_frame, caption=caption, use_container_width=True)
        else:
            processed_placeholder.info("Processing is running... waiting for first analysis.")

        if controller.latest_analysis is not None:
            with results_placeholder.container():
                st.subheader("Latest Analysis Results")
                st.json(controller.latest_analysis)

        # Control the frame rate
        time.sleep(sleep_duration)

    # --- Cleanup after loop exits ---
    if controller.processing_state == 'stopped':
        st.info(f"{'Webcam' if controller.is_live else 'Video'} feed stopped.")
        # Rerun the whole app to hide the player UI
        st.rerun()

if input_source == "Upload an image file":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file and analyzer_ready:
        logging.info(f"File uploaded: {uploaded_file.name}")
        process_image_st(uploaded_file)

elif input_source == "Upload a video file":
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], key="video_uploader")

    # When a new file is uploaded, reset the playback state
    if uploaded_file and uploaded_file.file_id != st.session_state.get('uploaded_video_id'):
        controller.stop_processing()
        st.session_state.video_file_path = None
        st.session_state.uploaded_video_id = uploaded_file.file_id

    if uploaded_file and analyzer_ready:
        if not controller.video_capture:
            if not st.session_state.video_file_path:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_file.read())
                    st.session_state.video_file_path = tfile.name
            try:
                controller.start_video_file(st.session_state.video_file_path)
            except IOError as e:
                st.error(str(e))
                controller.stop_processing()

elif input_source == "Use live webcam feed":
    if analyzer_ready:
        if not controller.video_capture:
            try:
                controller.start_webcam()
            except IOError as e:
                st.error(str(e))
                controller.stop_processing()

# --- Unified Controls and Display for Video and Webcam ---
if controller.video_capture:
    st.markdown("### Processing Controls")
    col1, col2, col3 = st.columns(3)

    if col1.button("▶️ Start Processing", use_container_width=True, disabled=controller.processing_state == 'running'):
        controller.start_processing()
        st.rerun()

    if col2.button("⏸️ Pause Processing", use_container_width=True, disabled=controller.processing_state != 'running'):
        controller.pause_processing()
        st.rerun()

    if col3.button("⏹️ Stop Processing", use_container_width=True, disabled=controller.processing_state == 'stopped'):
        controller.stop_processing()
        st.rerun()
    
    st.info(f"Processing State: **{controller.processing_state.upper()}**")

    if controller.processing_state in ('running', 'paused'):
        video_playback_and_processing()
    else:
        # Handle paused or stopped state by drawing the last known frames
        raw_frame, annotated_frame = controller.get_display_frames()
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### {'Live Feed' if controller.is_live else 'Video Playback'}")
            if raw_frame is not None:
                st.image(raw_frame, use_container_width=True)
            else:
                st.info(f"Playback is {controller.processing_state}.")
        with col2:
            st.markdown("### Processed Feed")
            if annotated_frame is not None:
                st.image(annotated_frame, caption=f"Processing {controller.processing_state}", use_container_width=True)
            else:
                st.info(f"Processing is {controller.processing_state}.")
        if controller.latest_analysis is not None:
            st.subheader("Latest Analysis Results")
            st.json(controller.latest_analysis)