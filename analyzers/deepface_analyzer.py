import logging
import numpy as np
from deepface import DeepFace
from .analyzer_base import Analyzer

class DeepfaceAnalyzer(Analyzer):
    """Analyzer using the DeepFace model."""

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Runs DeepFace analysis on a single frame for age, gender, and ethnicity.
        """
        logging.info("Attempting DeepFace analysis.")
        try:
            # DeepFace expects BGR format, which is what OpenCV provides.
            # It will raise a ValueError if no face is found.
            results = DeepFace.analyze(
                img_path=frame,
                actions=['age', 'gender', 'race'],
                enforce_detection=True,
                detector_backend='opencv'
            )

            detections = []
            height, width, _ = frame.shape

            # DeepFace returns a list of dicts, one for each detected face
            for face_data in results:
                region = face_data['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                box = [x / width, y / height, (x + w) / width, (y + h) / height]
                label = f"{face_data['dominant_gender']}, {face_data['age']}, {face_data['dominant_race']}"
                
                detection_info = face_data.copy()
                detection_info['label'] = label
                detection_info['box'] = box
                del detection_info['region'] # Redundant since we have the normalized box
                detections.append(detection_info)

            logging.info(f"DeepFace found {len(detections)} faces.")
            return {"detections": detections}

        except ValueError as e:
            # This is an expected outcome when no face is found, not an error.
            logging.info("DeepFace: No face detected in the frame.")
            return {"detections": []}
        except Exception as e:
            logging.error(f"DeepFace analysis failed: {e}")
            return None