import logging
import numpy as np
from .yolo_base_analyzer import YoloBaseAnalyzer

class YoloSegAnalyzer(YoloBaseAnalyzer):
    """Analyzer using a YOLO model for segmentation."""

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Runs YOLO segmentation on a single frame.
        """
        logging.info(f"Running YOLO segmentation with model {self.model.ckpt_path.split('/')[-1]} and confidence {self.confidence_threshold}")
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        detections = []
        result = results[0]

        if not result.masks:
            logging.info("YOLO-Seg: No masks detected in frame.")
            return {"detections": []}

        boxes = result.boxes.xyxyn.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.xyn

        for i in range(len(boxes)):
            detections.append({
                "label": self.model.names[class_ids[i]],
                "box": boxes[i].tolist(),
                "confidence": float(confs[i]),
                "mask": masks[i].tolist() # Add mask polygon
            })

        logging.info(f"YOLO-Seg found {len(detections)} objects.")
        return {"detections": detections}