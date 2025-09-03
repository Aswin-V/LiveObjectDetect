import logging
import numpy as np
from .yolo_base_analyzer import YoloBaseAnalyzer

class YoloPoseAnalyzer(YoloBaseAnalyzer):
    """Analyzer using a YOLO-Pose model."""

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Runs YOLO-Pose estimation on a single frame.
        """
        logging.info(f"Running YOLO-Pose estimation with model {self.model.ckpt_path.split('/')[-1]} and confidence {self.confidence_threshold}")
        # Perform inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        # Results is a list of one result object
        result = results[0]

        # Handle case where no poses are detected, which makes result.keypoints None
        if result.keypoints is None:
            logging.info("YOLO-Pose: No keypoints detected in frame.")
            return {"detections": []}

        # Get bounding boxes, confidences, and keypoints
        boxes = result.boxes.xyxyn.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        keypoints_xyn = result.keypoints.xyn.cpu().numpy()
        keypoints_conf = result.keypoints.conf.cpu().numpy()

        for i in range(len(boxes)):
            detections.append({
                "label": "person",
                "box": boxes[i].tolist(),
                "confidence": float(confs[i]),
                "keypoints": {
                    "xy": keypoints_xyn[i].tolist(),
                    "confidence": keypoints_conf[i].tolist()
                }
            })

        logging.info(f"YOLO-Pose found {len(detections)} persons.")
        return {"detections": detections}