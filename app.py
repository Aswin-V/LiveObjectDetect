# --- Installation ---
# Before running, please install the required libraries by running the following command in your terminal:
# pip install streamlit opencv-python-headless requests numpy Pillow

import streamlit as st
import cv2
import base64
import requests
import tempfile
import numpy as np
import json
import os
import logging
from PIL import Image
from ultralytics import YOLO

# --- Logging Configuration ---
# Configure logging to display the time, log level, and message.
# This will output to the console where streamlit is running.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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
    ("Gemini", "YOLO")
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
    # Add a password input field for the Gemini API Key for security.
    api_key_input = st.sidebar.text_input(
        "Gemini API Key", 
        type="password", 
        help="You can get your key from Google AI Studio."
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

# --- Helper Functions ---

def get_gemini_analysis(frame: np.ndarray, prompt: str) -> dict | None:
    """
    Sends a single frame to the Gemini API for analysis.
    """
    logging.info("Attempting Gemini analysis.")
    api_key = st.session_state.get("gemini_api_key")
    if not api_key:
        logging.error("Gemini API Key is missing.")
        st.error("Please enter your Gemini API Key in the sidebar.")
        return None

    GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    
    is_success, buffer = cv2.imencode(".jpg", frame)
    if not is_success:
        logging.error("Failed to encode frame to JPEG.")
        st.error("Failed to encode frame.")
        return None
    
    image_b64 = base64.b64encode(buffer).decode("utf-8")

    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}}]}],
        "generationConfig": {"responseMimeType": "application/json"}
    }
    
    headers = {"Content-Type": "application/json"}
    full_api_url = f"{GEMINI_API_URL}?key={api_key}"

    try:
        logging.info("Sending request to Gemini API.")
        response = requests.post(full_api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        logging.info("Received successful response from Gemini API.")
        result = response.json()
        
        if (result.get("candidates") and result["candidates"][0].get("content") and 
            result["candidates"][0]["content"].get("parts")):
            json_text = result["candidates"][0]["content"]["parts"][0]["text"]
            if json_text.startswith("```json"):
                json_text = json_text[7:-3]
            logging.info("Successfully parsed Gemini response.")
            return json.loads(json_text)
        else:
            logging.warning("No valid content in Gemini API response.")
            st.warning("No valid content found in API response.")
            st.json(result)
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Gemini API request failed: {e}")
        st.error(f"API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse JSON from Gemini response: {e}")
        st.error(f"Failed to parse JSON from API response: {e}")
        st.text(f"Received text: {response.text}")
        return None

def run_yolo_detection(frame: np.ndarray, model, confidence_thresh: float):
    """
    Runs YOLOv8/v9/v10 object detection on a single frame using the ultralytics library.
    """
    logging.info(f"Running YOLO detection with model {model.ckpt_path.split('/')[-1]} and confidence {confidence_thresh}")
    # Perform inference, specifying confidence and disabling verbose output
    results = model(frame, conf=confidence_thresh, verbose=False)

    detections = []
    # Ultralytics returns a list of results, we take the first one for our single image
    result = results[0]

    # Get bounding boxes, confidences, and class IDs
    boxes = result.boxes.xyxyn.cpu().numpy()  # Normalized [x_min, y_min, x_max, y_max]
    confs = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(boxes)):
        detections.append({
            "label": model.names[class_ids[i]],
            "box": boxes[i].tolist(), # The box is already normalized
            "confidence": float(confs[i])
        })

    logging.info(f"YOLO found {len(detections)} objects.")
    return {"detections": detections}

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
            label = detection.get("label", "Unknown")
            if "emotion" in detection:
                label += f" ({detection['emotion']})"
            elif "confidence" in detection:
                label += f" {detection['confidence']:.2f}"
            text_y = start_point[1] - 10 if start_point[1] > 20 else start_point[1] + 20
            cv2.putText(frame, label, (start_point[0], text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

# --- Main Application Logic ---

# Load YOLO model if selected.
# Using st.cache_resource ensures the model is loaded only once.
yolo_model = None
if model_selection == "YOLO":
    @st.cache_resource
    def load_yolo_model(model_name):
        logging.info(f"Loading YOLO model: {model_name}")
        try:
            model = YOLO(model_name)
            logging.info(f"YOLO model '{model_name}' loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading YOLO model {model_name}: {e}")
            st.error(f"Error loading YOLO model: {e}")
            return None
    yolo_model = load_yolo_model(yolo_model_name)

st.markdown("---")
input_source = st.radio("Select Input Source", ("Upload a video file", "Use live webcam feed"), horizontal=True)
logging.info(f"Input source selected: {input_source}")

prompt = (
    "Analyze this video frame. Identify all objects and provide their bounding boxes "
    "in the format [x_min, y_min, x_max, y_max] as normalized coordinates (0.0 to 1.0). "
    "If humans are present, identify their emotions and describe what they are doing. "
    "Provide the output as a JSON object with a key 'detections' which is an array of objects."
)

if 'stop' not in st.session_state:
    st.session_state.stop = False

def process_video(video_capture, is_live=False):
    """
    A generic function to process video from either a file or a webcam.
    """
    image_placeholder, results_placeholder = st.empty(), st.empty()
    frame_interval = 30 if is_live else int(video_capture.get(cv2.CAP_PROP_FPS) or 1)
    frame_num = 0

    spinner_text = "Live analysis in progress..." if is_live else "Processing video..."
    with st.spinner(spinner_text):
        while True:
            if is_live and st.session_state.stop: break
            success, frame = video_capture.read()
            if not success:
                logging.info("End of video file or stream.")
                break

            if frame_num % frame_interval == 0:
                logging.info(f"Processing frame number: {frame_num}")
                analysis = None
                if model_selection == "Gemini":
                    analysis = get_gemini_analysis(frame, prompt)
                elif model_selection == "YOLO" and yolo_model:
                    analysis = run_yolo_detection(frame, yolo_model, confidence_threshold)

                annotated_frame = frame
                if analysis and analysis.get("detections"):
                    logging.info(f"Found {len(analysis['detections'])} detections in frame {frame_num}.")
                    annotated_frame = draw_annotations(frame.copy(), analysis["detections"])
                    with results_placeholder.container():
                        st.subheader("Analysis Results")
                        st.json(analysis)
                else:
                    logging.warning(f"No analysis results for frame {frame_num}.")
                    with results_placeholder.container():
                        st.warning("No analysis results for this frame.")
                
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                caption = "Live Webcam Feed" if is_live else f"Frame {frame_num}"
                image_placeholder.image(rgb_frame, caption=caption, use_column_width=True)
            
            frame_num += 1
    
    video_capture.release()
    logging.info("Video capture released.")
    if not is_live: st.success("Video processing complete!")
    else: st.info("Webcam feed stopped.")

if input_source == "Upload a video file":
    st.session_state.stop = True
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])
    if uploaded_file:
        logging.info(f"File uploaded: {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            video_capture = cv2.VideoCapture(tfile.name)
        process_video(video_capture)

elif input_source == "Use live webcam feed":
    # Use a checkbox for a more intuitive and stateful UI control
    run_webcam = st.checkbox("Start live webcam feed")

    if run_webcam:
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