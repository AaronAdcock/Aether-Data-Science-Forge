"""Production Model Monitoring."""

import numpy as np
import logging
from typing import Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """Detects data drift between training and production data."""

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect_drift(self, train_data: np.ndarray, prod_data: np.ndarray) -> bool:
        """Simple drift detection using mean shift comparison.

        Args:
            train_data: Baseline training data.
            prod_data: New production data.

        Returns:
            True if drift is detected, False otherwise.
        """
        train_mean = np.mean(train_data, axis=0)
        prod_mean = np.mean(prod_data, axis=0)
        
        drift = np.abs(train_mean - prod_mean) > self.threshold
        if np.any(drift):
            logger.warning("Data drift detected in features!")
            return True
        return False

class MetricLogger:
    """Simulates Prometheus-style metric logging."""

    def log_inference(self, metrics: Dict[str, float]):
        """Logs inference performance metrics.

        Args:
            metrics: Dictionary of metric names and values.
        """
        for name, value in metrics.items():
            logger.info(f"PROMETHEUS_METRIC_{name}: {value}")
