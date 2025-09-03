import logging
import numpy as np
from .yolo_base_analyzer import YoloBaseAnalyzer

class YoloObbAnalyzer(YoloBaseAnalyzer):
    """Analyzer using a YOLO model for Oriented Bounding Box detection."""

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Runs YOLO OBB detection on a single frame.
        """
        logging.info(f"Running YOLO-OBB detection with model {self.model.ckpt_path.split('/')[-1]} and confidence {self.confidence_threshold}")
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        result = results[0]

        if not result.obb:
            logging.info("YOLO-OBB: No oriented boxes detected in frame.")
            return {"detections": []}

        confs = result.obb.conf.cpu().numpy()
        class_ids = result.obb.cls.cpu().numpy().astype(int)
        # Get the corner points of the rotated bounding box (normalized)
        obb_polygons = result.obb.xyxyn

        for i in range(len(confs)):
            detections.append({
                "label": self.model.names[class_ids[i]],
                "obb": obb_polygons[i].cpu().numpy().tolist(), # Rotated box corners
                "confidence": float(confs[i])
            })

        logging.info(f"YOLO-OBB found {len(detections)} objects.")
        return {"detections": detections}