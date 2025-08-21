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

def process_image(image_file):
    """
    Processes a single uploaded image file.
    """
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
                st.warning("No detections found in the image.")
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


if 'stop' not in st.session_state:
    st.session_state.stop = False

def process_video(video_capture, is_live=False):
    """
    A generic function to process video from either a file or a webcam.
    It displays every frame for smooth playback and overlays the latest analysis.
    """
    image_placeholder = st.empty()
    results_placeholder = st.empty()

    # Set the analysis interval. For live video, analyze more frequently.
    if is_live:
        frame_interval = 10  # Analyze every 10 frames for a responsive feel
    else:
        fps = video_capture.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = int(fps)  # Analyze once per second for uploaded videos

    frame_num = 0
    latest_analysis = None  # To store the most recent analysis results

    spinner_text = "Live analysis in progress..." if is_live else "Processing video..."
    with st.spinner(spinner_text):
        while True:
            if is_live and st.session_state.stop:
                break
            success, frame = video_capture.read()
            if not success:
                logging.info("End of video file or stream.")
                break

            # --- Analysis Section (runs periodically) ---
            if frame_num % frame_interval == 0:
                logging.info(f"Processing frame number: {frame_num}")
                try:
                    analysis = analyzer.analyze_frame(frame)
                    # If analysis was successful, update the latest results and the JSON display
                    if analysis and analysis.get("detections"):
                        logging.info(f"Found {len(analysis['detections'])} detections in frame {frame_num}.")
                        latest_analysis = analysis  # Store the new analysis
                        with results_placeholder.container():
                            st.subheader("Latest Analysis Results")
                            st.json(latest_analysis)
                    else:
                        logging.warning(f"No analysis results for frame {frame_num}.")
                        with results_placeholder.container():
                            st.warning("No new analysis results for this frame.")
                except Exception as e:
                    logging.error(f"Analysis failed for frame {frame_num}: {e}", exc_info=True)
                    with results_placeholder.container():
                        st.error(f"An error occurred during analysis: {e}")

            # --- Display Section (runs for every frame) ---
            annotated_frame = frame.copy()
            if latest_analysis and latest_analysis.get("detections"):
                # Draw annotations from the latest analysis onto the current frame
                annotated_frame = draw_annotations(annotated_frame, latest_analysis["detections"])

            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            caption = "Live Webcam Feed" if is_live else f"Frame {frame_num}"
            image_placeholder.image(rgb_frame, caption=caption, use_container_width=True)

            frame_num += 1

    video_capture.release()
    logging.info("Video capture released.")
    if not is_live:
        st.success("Video processing complete!")
    else:
        st.info("Webcam feed stopped.")

if input_source == "Upload an image file":
    st.session_state.stop = True
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file and analyzer:
        logging.info(f"File uploaded: {uploaded_file.name}")
        process_image(uploaded_file)

elif input_source == "Upload a video file":
    st.session_state.stop = True
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file and analyzer:
        logging.info(f"File uploaded: {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_capture = cv2.VideoCapture(tfile.name)
        process_video(video_capture)

elif input_source == "Use live webcam feed":
    # Use a checkbox for a more intuitive and stateful UI control
    run_webcam = st.checkbox("Start live webcam feed")

    if run_webcam and analyzer:
        logging.info("Starting webcam feed.")
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logging.error("Could not open webcam.")
            st.error("Could not open webcam. Please grant access and refresh.")
        else:
            st.session_state.stop = False # Ensure stop is False when starting
            logging.info("Webcam opened successfully.")
            process_video(video_capture, is_live=True)
    else:
        st.session_state.stop = True # Ensure stop is True when checkbox is unchecked