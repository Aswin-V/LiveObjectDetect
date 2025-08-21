from abc import ABC, abstractmethod
import numpy as np

class Analyzer(ABC):
    """Abstract base class for analysis models."""

    @abstractmethod
    def analyze_frame(self, frame: np.ndarray) -> dict | None:
        """
        Analyzes a single frame and returns the results.

        Args:
            frame: The frame to analyze.

        Returns:
            A dictionary containing the analysis results, or None if analysis fails.
        """
        pass