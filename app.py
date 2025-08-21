# --- Installation ---
# Before running, please install the required libraries by running the following command in your terminal:
# pip install streamlit opencv-python-headless requests numpy Pillow ultralytics deepface python-dotenv
import streamlit as st
import cv2
import tempfile
import numpy as np
import logging
import os
from PIL import Image
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from analyzers import GeminiAnalyzer, YoloAnalyzer, DeepfaceAnalyzer

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

# --- Model Loading ---
# Using st.cache_resource ensures the model is loaded only once.
@st.cache_resource
def load_yolo_model(model_name):
    """
    Loads a YOLO model from the specified path.
    The model is cached to avoid reloading on every app rerun.
    """
    logging.info(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)
    logging.info(f"YOLO model '{model_name}' loaded successfully.")
    return model

@st.cache_resource
def get_thread_pool():
    """Creates and returns a thread pool executor."""
    return ThreadPoolExecutor(max_workers=1)

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
logging.info(f"Model selected: {model_selection}")

st.sidebar.markdown("---")

# --- Model-specific Configurations ---
# Initialize configuration variables with default values to prevent NameError.
api_key_input = ""
confidence_threshold = 0.5
yolo_model_name = "yolov8n.pt" # Default model

# Display different UI elements in the sidebar based on the selected model.
if model_selection == "Gemini":
    # Get the API key from environment variables, with a fallback to an empty string.
    default_api_key = os.getenv("GEMINI_API_KEY", "")

    # Add a password input field for the Gemini API Key for security.
    api_key_input = st.sidebar.text_input(
        "Gemini API Key", 
        value=default_api_key,
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
    # --- YOLO Version and Size Selection ---
    yolo_version = st.sidebar.selectbox("YOLO Version", ["v8", "v9", "v10"], help="Choose the YOLO architecture.")
    
    yolo_sizes = {
        "v8": ['n', 's', 'm', 'l', 'x'],
        "v9": ['c', 'e'],
        "v10": ['n', 's', 'm', 'l', 'x']
    }
    yolo_size = st.sidebar.selectbox("Model Size", yolo_sizes[yolo_version], help="Nano is fastest, X is most accurate.")
    
    # Construct the model name based on user selection
    yolo_model_name = f"yolo{yolo_version}{yolo_size}.pt"

    # Add a slider to control the confidence threshold for YOLO detections.
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
        help="Adjust to show more or fewer detections."
    )
    # Display informational text about the YOLO model.
    st.sidebar.info(f"""
    **YOLO Model:**
    - **Selected:** `{yolo_model_name}`
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


# --- Helper Functions ---

def _get_label_text(detection: dict) -> str:
    """Creates a label string from a detection dictionary."""
    parts = [detection.get("label", "Unknown")]
    if "emotion" in detection:
        parts.append(f"({detection['emotion']})")
    elif "confidence" in detection:
        parts.append(f"{detection['confidence']:.2f}")
    return " ".join(parts)

def draw_annotations(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draws bounding boxes and labels on a frame for detections from either model.
    """
    height, width, _ = frame.shape
    for detection in detections:
        if "box" in detection and isinstance(detection["box"], list) and len(detection["box"]) == 4:
            box = detection["box"]
            x_min, y_min, x_max, y_max = box
            start_point = (int(x_min * width), int(y_min * height))
            end_point = (int(x_max * width), int(y_max * height))
            cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
            label = _get_label_text(detection)
            text_y = start_point[1] - 10 if start_point[1] > 20 else start_point[1] + 20
            cv2.putText(frame, label, (start_point[0], text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def _analyze_frame_in_thread(analyzer, frame, frame_num):
    """
    Helper to run analysis in a separate thread.
    Returns the analysis result, the original frame, and the frame number.
    """
    analysis = analyzer.analyze_frame(frame)
    return analysis, frame, frame_num

def process_image(image_file):
    """
    Processes a single uploaded image file.
    """
    st.markdown("### Analyzed Image")
    image_placeholder = st.empty()
    results_placeholder = st.empty()

    # Convert uploaded file to an OpenCV image
    image = Image.open(image_file).convert("RGB")
    frame = np.array(image)
    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    with st.spinner("Analyzing image..."):
        logging.info(f"Processing uploaded image: {image_file.name}")
        try:
            analysis = analyzer.analyze_frame(frame)
        except Exception as e:
            logging.error(f"Analysis failed for uploaded image: {e}", exc_info=True)
            image_placeholder.empty() # Clear the spinner
            results_placeholder.error(f"An error occurred during analysis: {e}")
            st.stop()


        if analysis and analysis.get("detections"):
            logging.info(f"Found {len(analysis['detections'])} detections in the image.")
            annotated_frame = frame.copy()
            annotated_frame = draw_annotations(annotated_frame, analysis["detections"])
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_frame, caption="Analyzed Image", use_container_width=True)

            with results_placeholder.container():
                st.subheader("Analysis Results")
                st.json(analysis)
        else:
            logging.warning("No analysis results for the image.")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_frame, caption="Uploaded Image (no detections)", use_container_width=True)
            with results_placeholder.container():
                st.info("No detections found in the image.")
    st.success("Image analysis complete!")

# --- Main Application Logic ---

analyzer = None

st.markdown("---")
input_source = st.radio("Select Input Source", ("Upload an image file", "Upload a video file", "Use live webcam feed"), horizontal=True)
logging.info(f"Input source selected: {input_source}")

prompt = (
    "Analyze this image. Identify all objects and provide their bounding boxes "
    "in the format [x_min, y_min, x_max, y_max] as normalized coordinates (0.0 to 1.0). "
    "If humans are present, identify their emotions and describe what they are doing. "
    "Provide the output as a JSON object with a key 'detections' which is an array of objects."
)

try:
    if model_selection == "Gemini":
        api_key = st.session_state.get("gemini_api_key")
        if api_key:
            analyzer = GeminiAnalyzer(api_key=api_key, prompt=prompt)
        else:
            st.warning("Please enter your Gemini API Key in the sidebar to proceed.")
    elif model_selection == "YOLO":
        yolo_model = load_yolo_model(yolo_model_name)
        if yolo_model:
            analyzer = YoloAnalyzer(model=yolo_model, confidence_threshold=confidence_threshold)
    elif model_selection == "DeepFace":
        analyzer = DeepfaceAnalyzer()
except ValueError as e:
    st.error(f"Error initializing analyzer: {e}")
    logging.error(f"Error initializing analyzer: {e}")
    st.stop()

# --- Session State Initialization ---
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'analysis_future' not in st.session_state:
    st.session_state.analysis_future = None
# 'stopped', 'running', 'paused'
if 'processing_state' not in st.session_state:
    st.session_state.processing_state = 'stopped'
if 'latest_analysis' not in st.session_state:
    st.session_state.latest_analysis = None
if 'video_capture' not in st.session_state:
    st.session_state.video_capture = None
if 'latest_annotated_frame' not in st.session_state:
    st.session_state.latest_annotated_frame = None

def process_video(is_live=False):
    """
    A generic function to process video from either a file or a webcam.
    It displays every frame for smooth playback and overlays the latest analysis.
    """
    # Create placeholders for the video feeds and results
    if is_live:
        col1, col2 = st.columns(2)
        col1.markdown("### Live Feed")
        live_placeholder = col1.empty()
        col2.markdown("### Processed Feed")
        processed_placeholder = col2.empty()
        if st.session_state.processing_state == 'stopped':
            processed_placeholder.info("Processing is stopped. Press 'Start Processing' to begin.")
    else:
        st.markdown("### Processed Feed")
        processed_placeholder = st.empty()
        processed_placeholder.info("Processing video, please wait...")
    
    results_placeholder = st.empty()

    # Set the analysis interval. For live video, analyze more frequently.
    if is_live:
        frame_interval = 10  # Analyze every 10 frames for a responsive feel
    else:
        fps = st.session_state.video_capture.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps)  # Analyze once per second for uploaded videos

    # Get the thread pool for running analysis in the background
    executor = get_thread_pool()

    frame_num = 0
        
    spinner_text = "Live analysis in progress..." if is_live else "Processing video..."
    # For live, the spinner is less useful since we have explicit controls.
    if not is_live:
        st.spinner(spinner_text)

    while True if not is_live else st.session_state.webcam_running:
        success, frame = st.session_state.video_capture.read()
        if not success:
            logging.info("End of video file or stream.")
            st.session_state.webcam_running = False # Stop loop if webcam fails
            break
        
        # --- Check for completed analysis (runs for every frame) ---
        if st.session_state.analysis_future and st.session_state.analysis_future.done():
            try:
                analysis, analyzed_frame, analyzed_frame_num = st.session_state.analysis_future.result()
                if analysis is not None:
                    st.session_state.latest_analysis = analysis
                    with results_placeholder.container():
                        st.subheader("Latest Analysis Results")
                        st.json(st.session_state.latest_analysis)
                    
                    # Draw annotations on the frame that was actually analyzed
                    annotated_frame = draw_annotations(analyzed_frame.copy(), analysis.get("detections", []))
                    st.session_state.latest_annotated_frame = annotated_frame
                    
                    # Update the processed feed placeholder with the new static annotated image
                    caption = f"Analyzed Frame: {analyzed_frame_num}" if not is_live else "Last Analyzed Frame"
                    processed_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)

                    num_detections = len(analysis.get("detections", []))
                    if num_detections > 0:
                        logging.info(f"Found {num_detections} detections in async result.")
                else:
                    logging.warning("Async analysis returned no result.")
            except Exception as e:
                logging.error(f"Async analysis failed: {e}", exc_info=True)
                results_placeholder.error(f"An error occurred during analysis: {e}")
            st.session_state.analysis_future = None # Clear the future to allow the next one

        # --- Display Section (runs for every frame) ---
        if is_live:
            live_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        # When paused, we need to continuously redraw the last annotated frame
        if st.session_state.processing_state == 'paused':
            if st.session_state.latest_annotated_frame is not None:
                processed_placeholder.image(
                    cv2.cvtColor(st.session_state.latest_annotated_frame, cv2.COLOR_BGR2RGB),
                    caption="Processing Paused",
                    use_container_width=True
                )
            else:
                processed_placeholder.info("Processing is paused.")
        else: # For uploaded video, we need to show something on every frame
            annotated_frame = frame.copy()
            if st.session_state.latest_analysis:
                annotated_frame = draw_annotations(annotated_frame, st.session_state.latest_analysis.get("detections", []))
            processed_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_num}", use_container_width=True)

        # --- Analysis Section (runs periodically) ---
        # For webcam, check the processing state
        should_process = st.session_state.processing_state == 'running' if is_live else True
        if should_process and frame_num % frame_interval == 0 and st.session_state.analysis_future is None:
            logging.info(f"Submitting frame {frame_num} for async analysis.")
            # Submit analysis to the thread pool. It will run in the background.
            st.session_state.analysis_future = executor.submit(_analyze_frame_in_thread, analyzer, frame.copy(), frame_num)
        elif is_live and frame_num % frame_interval == 0:
            logging.info(f"Skipping analysis for frame {frame_num}, previous one still running.")

        frame_num += 1

    if st.session_state.video_capture:
        st.session_state.video_capture.release()
    st.session_state.video_capture = None
    logging.info("Video capture released.")
    if not is_live:
        st.success("Video processing complete!")
    else:
        st.info("Webcam feed stopped.")
        # Clear placeholders when webcam stops
        live_placeholder.empty()
        processed_placeholder.empty()
        results_placeholder.empty()

if input_source == "Upload an image file":
    st.session_state.webcam_running = False
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file and analyzer:
        logging.info(f"File uploaded: {uploaded_file.name}")
        process_image(uploaded_file)

elif input_source == "Upload a video file":
    st.session_state.webcam_running = False
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file and analyzer:
        logging.info(f"File uploaded: {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            st.session_state.video_capture = cv2.VideoCapture(tfile.name)
        process_video(is_live=False)

elif input_source == "Use live webcam feed":
    # Using a single checkbox to control the webcam state is cleaner
    run_webcam = st.checkbox("Run live webcam feed", value=st.session_state.webcam_running)

    if run_webcam and not st.session_state.webcam_running:
        # Just started
        st.session_state.webcam_running = True
        st.session_state.processing_state = 'stopped'
        st.rerun()
    elif not run_webcam and st.session_state.webcam_running:
        # Just stopped
        st.session_state.webcam_running = False
        st.session_state.processing_state = 'stopped'
        st.rerun()

    if st.session_state.webcam_running:
        st.markdown("### Processing Controls")
        col1, col2, col3 = st.columns(3)

        if col1.button("Start Processing", use_container_width=True, disabled=st.session_state.processing_state == 'running'):
            st.session_state.processing_state = 'running'
            st.rerun()

        if col2.button("Pause Processing", use_container_width=True, disabled=st.session_state.processing_state != 'running'):
            st.session_state.processing_state = 'paused'
            st.rerun()

        if col3.button("Stop Processing", use_container_width=True, disabled=st.session_state.processing_state == 'stopped'):
            st.session_state.processing_state = 'stopped'
            st.session_state.latest_analysis = None
            st.session_state.latest_annotated_frame = None
            # We can't easily cancel a future. We'll just ignore its result.
            st.session_state.analysis_future = None
            st.rerun()
        
        if analyzer:
            st.info(f"Processing State: **{st.session_state.processing_state.upper()}**")
            if not st.session_state.video_capture:
                st.session_state.video_capture = cv2.VideoCapture(0)
            
            if not st.session_state.video_capture.isOpened():
                logging.error("Could not open webcam.")
                st.error("Could not open webcam. Please grant access and refresh.")
                st.session_state.webcam_running = False
                st.session_state.video_capture = None
            else:
                logging.info("Webcam opened successfully.")
                process_video(is_live=True)
        else:
            st.warning("Please select a model to start the webcam feed.")
            st.session_state.webcam_running = False

    else:
        # This block runs when the checkbox is unchecked, ensuring cleanup
        if st.session_state.video_capture:
            st.session_state.video_capture.release()
            st.session_state.video_capture = None
            logging.info("Webcam capture released on stop.")