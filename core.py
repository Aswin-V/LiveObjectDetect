import cv2
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import argparse
import os
from dotenv import load_dotenv

from analyzers import (
    GeminiAnalyzer, YoloAnalyzer, DeepfaceAnalyzer, YoloPoseAnalyzer, YoloBaseAnalyzer,
    YoloSegAnalyzer, YoloClsAnalyzer, YoloObbAnalyzer)
from ultralytics import YOLO

# --- Constants for Pose Estimation Drawing ---
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (11, 12), (5, 11), (6, 12),
    (11, 13), (12, 14), (13, 15), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)
]
SKELETON_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
    (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
    (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
    (255, 0, 255), (255, 0, 170), (255, 0, 85)
]

def create_argument_parser():
    """Creates and returns the argument parser for the standalone and tkinter apps."""
    # This needs to be called before creating the parser to ensure env vars are loaded.
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="AI Video Analysis Application.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="YOLO",
        choices=["YOLO", "DeepFace", "Gemini"],
        help="The analysis model to use. Default: YOLO"
    )
    parser.add_argument(
        "-v", "--video",
        type=str,
        default=None,
        help="Path to a video file to process. If not provided, the webcam will be used."
    )

    yolo_group = parser.add_argument_group('YOLO Options (only used if --model is YOLO)')
    yolo_group.add_argument("--yolo-task", type=str, default="detect", choices=['detect', 'segment', 'classify', 'pose', 'obb'], help="YOLO task to perform. Default: detect")
    yolo_group.add_argument("--yolo-version", type=str, default="v8", choices=["v8", "v9", "v10", "v11"], help="YOLO architecture version. Default: v8")
    yolo_group.add_argument("--yolo-size", type=str, default="n", help="YOLO model size. e.g., n,s,m,l,x for v8; c,e for v9; n,s,m,b,l,x for v10. Default: n")
    yolo_group.add_argument("-c", "--confidence", type=float, default=0.25, help="YOLO confidence threshold. Default: 0.25")

    gemini_group = parser.add_argument_group('Gemini Options (used if --model is Gemini)')
    gemini_group.add_argument("--api-key", type=str, default=os.getenv("GEMINI_API_KEY"), help="Gemini API Key. Can also be set via GEMINI_API_KEY environment variable in a .env file.")

    return parser

YOLO_VALID_SIZES_BY_VERSION = {
    "v8": ['n', 's', 'm', 'l', 'x'],
    "v9": ['c', 'e'],
    "v10": ['n', 's', 'm', 'b', 'l', 'x'],
    "v11": ['n', 's', 'm', 'l', 'x'],
    "v12": ['n', 's', 'm', 'l', 'x']
}

YOLO_VALID_TASKS_BY_VERSION = {
    "v8": ['detect', 'segment', 'pose', 'obb', 'classify'],
    "v9": ['detect', 'segment', 'classify'],
    "v10": ['detect'],
    "v11": ['detect', 'segment', 'pose', 'obb', 'classify'],
    "v12": ['detect', 'segment', 'pose', 'obb', 'classify']
}

YOLO_VERSIONS = list(YOLO_VALID_SIZES_BY_VERSION.keys())

def validate_yolo_args(parser, args):
    """Validates YOLO-specific arguments and calls parser.error() if invalid."""
    if args.model == "YOLO":
        if args.yolo_size not in YOLO_VALID_SIZES_BY_VERSION.get(args.yolo_version, []):
            parser.error(f"Invalid size '{args.yolo_size}' for YOLO {args.yolo_version}. Valid sizes are: {YOLO_VALID_SIZES_BY_VERSION.get(args.yolo_version, [])}")
        if args.yolo_task not in YOLO_VALID_TASKS_BY_VERSION.get(args.yolo_version, []):
            parser.error(f"Invalid task '{args.yolo_task}' for YOLO {args.yolo_version}. Valid tasks are: {YOLO_VALID_TASKS_BY_VERSION.get(args.yolo_version, [])}")

def create_yolo_analyzer_params(version: str, size: str, task: str, confidence: float) -> dict:
    """Creates a dictionary of parameters for the YOLO analyzer."""
    task_suffix = ""
    if task != "detect":
        if task == "classify":
            task_suffix = "-cls"
        else:
            task_suffix = f"-{task}"

    model_version_str = version
    if version in ["v11", "v12"]:
        model_version_str = version.lstrip('v')

    model_name = f"yolo{model_version_str}{size}{task_suffix}.pt"

    return {
        "yolo_model_name": model_name,
        "confidence_threshold": confidence,
        "yolo_task": task
    }

# --- Model Loading & Thread Pool (UI-Agnostic) ---
_yolo_model_cache = {}
_thread_pool_executor = None

def load_yolo_model(model_name):
    """Loads a YOLO model from the specified path."""
    if model_name not in _yolo_model_cache:
        logging.info(f"Loading YOLO model: {model_name}")
        try:
            model = YOLO(model_name)
            _yolo_model_cache[model_name] = model
            logging.info(f"YOLO model '{model_name}' loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load YOLO model '{model_name}'. It may not be an official Ultralytics model or a network error occurred. Error: {e}")
            _yolo_model_cache[model_name] = None # Cache failure
    return _yolo_model_cache[model_name]

def get_thread_pool():
    """Creates and returns a thread pool executor."""
    global _thread_pool_executor
    if _thread_pool_executor is None:
        _thread_pool_executor = ThreadPoolExecutor(max_workers=1)
    return _thread_pool_executor

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
    Draws bounding boxes, labels, and keypoints on a frame.
    """
    if not detections:
        return frame
        
    height, width, _ = frame.shape
    annotated_frame = frame.copy()
    for detection in detections:
        # Draw bounding box
        if "box" in detection and isinstance(detection["box"], list) and len(detection["box"]) == 4:
            box = detection["box"]
            x_min, y_min, x_max, y_max = box
            start_point = (int(x_min * width), int(y_min * height))
            end_point = (int(x_max * width), int(y_max * height))
            cv2.rectangle(annotated_frame, start_point, end_point, (0, 255, 0), 2)
            label = _get_label_text(detection)
            text_y = start_point[1] - 10 if start_point[1] > 20 else start_point[1] + 20
            cv2.putText(annotated_frame, label, (start_point[0], text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw segmentation mask
        if "mask" in detection and detection["mask"]:
            mask_points = np.array(detection["mask"], dtype=np.int32)
            # Rescale mask points from normalized (0-1) to image coordinates
            mask_points[:, 0] *= width
            mask_points[:, 1] *= height
            # Create a transparent overlay for the mask
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [mask_points], (0, 255, 100),-1)
            alpha = 0.4
            annotated_frame = cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0)
            # Draw mask contour
            cv2.polylines(annotated_frame, [mask_points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw oriented bounding box
        if "obb" in detection and detection["obb"]:
            obb_points = np.array(detection["obb"], dtype=np.int32)
            obb_points[:, 0] *= width
            obb_points[:, 1] *= height
            cv2.polylines(annotated_frame, [obb_points], isClosed=True, color=(255, 100, 0), thickness=2)

        # Draw keypoints for pose estimation
        if "keypoints" in detection and detection["keypoints"]:
            kpts = detection["keypoints"]["xy"]
            kpts_conf = detection["keypoints"]["confidence"]
            
            # Draw skeleton
            for i, (p1_idx, p2_idx) in enumerate(SKELETON_CONNECTIONS):
                if kpts_conf[p1_idx] > 0.5 and kpts_conf[p2_idx] > 0.5:
                    p1 = (int(kpts[p1_idx][0] * width), int(kpts[p1_idx][1] * height))
                    p2 = (int(kpts[p2_idx][0] * width), int(kpts[p2_idx][1] * height))
                    color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
                    cv2.line(annotated_frame, p1, p2, color, 2)

            # Draw keypoint circles
            for i, kpt in enumerate(kpts):
                if kpts_conf[i] > 0.5:
                    center = (int(kpt[0] * width), int(kpt[1] * height))
                    color = SKELETON_COLORS[i % len(SKELETON_COLORS)]
                    cv2.circle(annotated_frame, center, 5, color, -1)

        # Display classification results
        if "class_probs" in detection and detection["class_probs"]:
            probs = detection["class_probs"]
            text_y = 30
            for prob in probs:
                label = f"{prob['label']}: {prob['confidence']:.2f}"
                # Changed text color from cyan (0, 255, 255) to red (0, 0, 255)
                cv2.putText(annotated_frame, label, (15, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                text_y += 30

    return annotated_frame

def _analyze_frame_in_thread(analyzer, frame, frame_num):
    """
    Helper to run analysis in a separate thread.
    Returns the analysis result, the original frame, and the frame number.
    """
    analysis = analyzer.analyze_frame(frame)
    return analysis, frame, frame_num

class AppController:
    """
    Core logic for the AI Video Analysis application.
    This class is UI-agnostic and can be used by a web app (Streamlit)
    or a standalone desktop app.
    """
    def __init__(self):
        self.analyzer = None
        self.video_capture = None
        self.is_live = False
        
        # State variables
        self.processing_state = 'stopped'  # 'stopped', 'running', 'paused'
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_analysis = None
        self.analyzed_frame_number = 0
        self.frame_count = 0
        self.analysis_future = None
        self.executor = get_thread_pool()
        self.frame_interval = 30 # Default analysis interval

    def set_analyzer(self, model_selection, **kwargs):
        """Sets the analysis model."""
        logging.info(f"Setting analyzer to {model_selection}")
        prompt = (
            "Analyze this image. Identify all objects and provide their bounding boxes "
            "in the format [x_min, y_min, x_max, y_max] as normalized coordinates (0.0 to 1.0). "
            "If humans are present, identify their emotions and describe what they are doing. "
            "Provide the output as a JSON object with a key 'detections' which is an array of objects."
        )
        try:
            if model_selection == "Gemini":
                api_key = kwargs.get("api_key")
                if api_key:
                    self.analyzer = GeminiAnalyzer(api_key=api_key, prompt=prompt)
                else:
                    self.analyzer = None
                    logging.warning("Gemini analyzer requires an API key.")
            elif model_selection == "YOLO":
                yolo_model_name = kwargs.get("yolo_model_name", "yolov8n.pt")
                confidence_threshold = kwargs.get("confidence_threshold", 0.25)
                yolo_task = kwargs.get("yolo_task", "detect")

                yolo_model = load_yolo_model(yolo_model_name)
                if not yolo_model:
                    raise ValueError(f"Model '{yolo_model_name}' could not be loaded.")

                analyzer_map = {
                    "detect": YoloAnalyzer,
                    "pose": YoloPoseAnalyzer,
                    "segment": YoloSegAnalyzer,
                    "classify": YoloClsAnalyzer,
                    "obb": YoloObbAnalyzer
                }
                AnalyzerClass = analyzer_map.get(yolo_task, YoloAnalyzer)
                self.analyzer = AnalyzerClass(model=yolo_model, confidence_threshold=confidence_threshold)
            elif model_selection == "DeepFace":
                self.analyzer = DeepfaceAnalyzer()
            else:
                self.analyzer = None
        except ValueError as e:
            logging.error(f"Error initializing analyzer: {e}")
            self.analyzer = None
        return self.analyzer is not None

    def process_single_image(self, image_file):
        """Processes a single uploaded image file."""
        if not self.analyzer:
            return None, None, {"error": "Analyzer not set."}

        image = Image.open(image_file).convert("RGB")
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        try:
            analysis = self.analyzer.analyze_frame(frame)
            if analysis and analysis.get("detections"):
                annotated_frame = draw_annotations(frame.copy(), analysis["detections"])
                return frame, annotated_frame, analysis
            else:
                return frame, frame, {"info": "No detections found."}
        except Exception as e:
            logging.error(f"Analysis failed for uploaded image: {e}", exc_info=True)
            return frame, frame, {"error": str(e)}

    def start_video_file(self, video_path):
        """Starts processing a video file."""
        self.stop_processing()
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            self.video_capture = None
            raise IOError(f"Could not open video file: {video_path}")
        self.is_live = False
        fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
        self.frame_interval = int(fps) if fps > 0 else 30
        logging.info(f"Started video file processing: {video_path} at {fps} FPS.")

    def start_webcam(self):
        """Starts processing the live webcam feed."""
        self.stop_processing()
        self.video_capture = cv2.VideoCapture(0)
        if not self.video_capture.isOpened():
            self.video_capture = None
            raise IOError("Could not open webcam.")
        self.is_live = True
        self.frame_interval = 10 # Analyze every 10 frames for webcam
        logging.info("Started webcam processing.")

    def start_processing(self):
        if self.video_capture:
            self.processing_state = 'running'

    def pause_processing(self):
        self.processing_state = 'paused'

    def stop_processing(self):
        """Stops any ongoing processing and releases resources."""
        self.processing_state = 'stopped'
        if self.video_capture:
            self.video_capture.release()
        self.video_capture = None
        self.latest_frame = None
        self.latest_annotated_frame = None
        self.latest_analysis = None
        self.analyzed_frame_number = 0
        self.frame_count = 0
        self.analysis_future = None
        logging.info("Processing stopped and resources released.")

    def process_next_frame(self):
        """Processes a single frame from the current video source."""
        if self.processing_state == 'stopped' or not self.video_capture:
            return False

        # For video files, 'paused' means we don't read the next frame.
        # For live webcam, we always read to keep the feed live.
        if self.processing_state == 'paused' and not self.is_live:
            # For paused video, we don't advance the frame, but we need to
            # keep the loop in app.py alive to respond to controls.
            # We return True to keep the loop running, but do nothing else.
            return True

        success, frame = self.video_capture.read()
        if not success:
            self.stop_processing()
            return False
        
        self.frame_count += 1
        self.latest_frame = frame

        # Only perform analysis if the state is 'running'.
        # When 'paused', we still read the frame to update the live feed, but skip analysis.
        if self.processing_state == 'running':
            # --- Check for completed analysis ---
            if self.analysis_future and self.analysis_future.done():
                try:
                    analysis, analyzed_frame, frame_num = self.analysis_future.result()
                    if analysis is not None:
                        self.latest_analysis = analysis
                        self.analyzed_frame_number = frame_num
                        annotated_frame = draw_annotations(analyzed_frame.copy(), analysis.get("detections", []))
                        self.latest_annotated_frame = annotated_frame
                    else:
                        logging.warning("Async analysis returned no result.")
                except Exception as e:
                    logging.error(f"Async analysis failed: {e}", exc_info=True)
                    self.latest_analysis = {"error": str(e)}
                self.analysis_future = None

            # --- Submit new frame for analysis ---
            if self.analyzer and self.frame_count % self.frame_interval == 0 and self.analysis_future is None:
                logging.info(f"Submitting frame {self.frame_count} for async analysis.")
                self.analysis_future = self.executor.submit(
                    _analyze_frame_in_thread, self.analyzer, frame.copy(), self.frame_count
                )
        
        return True

    def get_display_frames(self):
        """Returns the latest raw and annotated frames for display."""
        raw_frame_rgb = None
        if self.latest_frame is not None:
            raw_frame_rgb = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)

        annotated_frame_rgb = None
        if self.latest_annotated_frame is not None:
            annotated_frame_rgb = cv2.cvtColor(self.latest_annotated_frame, cv2.COLOR_BGR2RGB)
        
        return raw_frame_rgb, annotated_frame_rgb

    def get_sleep_duration(self):
        """Calculates the sleep duration to maintain video frame rate."""
        if self.is_live or not self.video_capture:
            return 0.01
        
        fps = self.video_capture.get(cv2.CAP_PROP_FPS) or 30
        return 1 / fps if fps > 0 else 0.01

    def get_confidence_threshold(self) -> float:
        """Gets the confidence threshold from the current analyzer if applicable."""
        if isinstance(self.analyzer, YoloBaseAnalyzer):
            return self.analyzer.confidence_threshold
        return 0.25 # A sensible default

    def set_confidence_threshold(self, confidence: float):
        """Sets the confidence threshold for the current analyzer if applicable."""
        if isinstance(self.analyzer, YoloBaseAnalyzer):
            self.analyzer.confidence_threshold = confidence
            logging.info(f"YOLO confidence threshold set to {confidence}")
            return True
        return False