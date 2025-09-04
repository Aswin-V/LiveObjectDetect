import logging
import numpy as np
from .yolo_base_analyzer import YoloBaseAnalyzer

class YoloAnalyzer(YoloBaseAnalyzer):
    """Analyzer using the YOLO model."""

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Runs YOLO object detection on a single frame using the ultralytics library.
        """
        logging.info(f"Running YOLO detection with model {self.model.ckpt_path.split('/')[-1]} and confidence {self.confidence_threshold}")
        # Perform inference, specifying confidence and disabling verbose output
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        # Ultralytics returns a list of results, we take the first one for our single image
        result = results[0]

        # Get bounding boxes, confidences, and class IDs
        boxes = result.boxes.xyxyn.cpu().numpy()  # Normalized [x_min, y_min, x_max, y_max]
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        for i in range(len(boxes)):
            detections.append({
                "label": self.model.names[class_ids[i]],
                "box": boxes[i].tolist(), # The box is already normalized
                "confidence": float(confs[i])
            })

        logging.info(f"YOLO found {len(detections)} objects.")
        return {"detections": detections}