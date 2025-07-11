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
    ("Gemini", "YOLOv3")
)
logging.info(f"Model selected: {model_selection}")

st.sidebar.markdown("---")

# --- Model-specific Configurations ---
# Initialize configuration variables with default values to prevent NameError.
api_key_input = ""
confidence_threshold = 0.5

# Display different UI elements in the sidebar based on the selected model.
if model_selection == "Gemini":
    # Add a password input field for the Gemini API Key for security.
    api_key_input = st.sidebar.text_input(
        "Gemini API Key", 
        type="password", 
        help="You can get your key from Google AI Studio."
    )
    # Display informational text about the Gemini model's capabilities and requirements.
    st.sidebar.info("""
    **Gemini Model:**
    - Cloud-based (requires internet & API Key).
    - Detects objects, human emotions, and activities.
    - Slower due to API calls.
    """)
elif model_selection == "YOLOv3":
    # Add a slider to control the confidence threshold for YOLO detections.
    confidence_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
        help="Adjust to show more or fewer detections."
    )
    # Display informational text about the YOLO model.
    st.sidebar.info("""
    **YOLOv3 Model:**
    - Runs locally (faster).
    - Detects a wide range of objects.
    - Does **not** detect emotions or activities.
    """)
    # Display a warning with setup instructions for the user.
    st.sidebar.warning("""
    **Setup Required for YOLO:**
    Please download the following files and place them in the same directory as this script:
    1. `yolov3.weights`
    2. `yolov3.cfg`
    3. `coco.names`
    """)

# --- Helper Functions ---

def get_gemini_analysis(frame: np.ndarray, prompt: str, api_key: str) -> dict | None:
    """
    Sends a single frame to the Gemini API for analysis.
    """
    logging.info("Attempting Gemini analysis.")
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

def run_yolo_detection(frame: np.ndarray, net, output_layers, classes, confidence_thresh: float):
    """
    Runs YOLO object detection on a single frame.
    """
    logging.info(f"Running YOLO detection with confidence threshold: {confidence_thresh}")
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes, detections = [], [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_thresh:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, 0.4)
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            norm_box = [x / width, y / height, (x + w) / width, (y + h) / height]
            detections.append({"label": label, "box": norm_box, "confidence": confidences[i]})
    
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

# Load YOLO model files if YOLO is selected
yolo_net, yolo_output_layers, yolo_classes = None, None, None
if model_selection == "YOLOv3":
    logging.info("Checking for YOLO model files.")
    weights_path, cfg_path, names_path = "yolov3.weights", "yolov3.cfg", "coco.names"
    if all(os.path.exists(p) for p in [weights_path, cfg_path, names_path]):
        logging.info("YOLO files found. Loading model.")
        yolo_net = cv2.dnn.readNet(weights_path, cfg_path)
        with open(names_path, "r") as f:
            yolo_classes = [line.strip() for line in f.readlines()]
        layer_names = yolo_net.getLayerNames()
        yolo_output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers().flatten()]
        logging.info("YOLO model loaded successfully.")
    else:
        logging.error("YOLO model files not found.")
        st.error("YOLO model files not found. Please follow setup instructions.")

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
                    analysis = get_gemini_analysis(frame, prompt, api_key_input)
                elif model_selection == "YOLOv3" and yolo_net:
                    analysis = run_yolo_detection(frame, yolo_net, yolo_output_layers, yolo_classes, confidence_threshold)

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
    col1, col2 = st.columns(2)
    if col1.button("Start Webcam"):
        logging.info("Start Webcam button clicked.")
        st.session_state.stop = False
    if col2.button("Stop Webcam"):
        logging.info("Stop Webcam button clicked.")
        st.session_state.stop = True

    if not st.session_state.stop:
        logging.info("Starting webcam feed.")
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            logging.error("Could not open webcam.")
            st.error("Could not open webcam. Please grant access and refresh.")
        else:
            logging.info("Webcam opened successfully.")
            process_video(video_capture, is_live=True)