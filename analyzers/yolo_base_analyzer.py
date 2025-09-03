from abc import ABC
from .analyzer_base import Analyzer

class YoloBaseAnalyzer(Analyzer, ABC):
    """Base class for YOLO-based analyzers."""

    def __init__(self, model, confidence_threshold: float):
        if not model:
            raise ValueError("YOLO model is not loaded.")
        self.model = model
        self.confidence_threshold = confidence_threshold