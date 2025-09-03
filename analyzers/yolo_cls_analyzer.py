import logging
import numpy as np
from .yolo_base_analyzer import YoloBaseAnalyzer

class YoloClsAnalyzer(YoloBaseAnalyzer):
    """Analyzer using a YOLO model for classification."""

    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Runs YOLO classification on a single frame.
        """
        logging.info(f"Running YOLO classification with model {self.model.ckpt_path.split('/')[-1]} and confidence {self.confidence_threshold}")
        # Confidence is not directly used for classification in the same way, but we run it anyway.
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)

        result = results[0]
        
        if not result.probs:
            logging.info("YOLO-Cls: No classification results.")
            return {"detections": []}

        # For classification, there's one result for the whole image.
        # We'll format it as a single "detection" for consistency.
        top5_indices = result.probs.top5
        top5_conf = result.probs.top5conf.cpu().numpy()
        
        class_probs = []
        for i in range(len(top5_indices)):
            class_probs.append({
                "label": self.model.names[top5_indices[i]],
                "confidence": float(top5_conf[i])
            })

        detection = {"label": "classification", "class_probs": class_probs}

        logging.info(f"YOLO-Cls top class: {class_probs[0]['label']} ({class_probs[0]['confidence']:.2f})")
        return {"detections": [detection]}