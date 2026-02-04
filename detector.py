import numpy as np
import pandas as pd
from typing import List, Dict, Union

class AnomalyDetector:
    def __init__(self, static_threshold: float = 5000.0, sigma_factor: float = 3.0):
        """
        Args:
            static_threshold: Absolute power limit in Watts.
            sigma_factor: Number of standard deviations for dynamic check.
        """
        self.static_threshold = static_threshold
        self.sigma_factor = sigma_factor
        self.history = []

    def check_static(self, current_power: float) -> bool:
        """Return True if power exceeds absolute limit."""
        return current_power > self.static_threshold

    def check_dynamic(self, current_power: float, predicted_mean: float, predicted_std: float) -> bool:
        """
        Return True if power is outside confidence interval [mean - k*std, mean + k*std].
        """
        lower = predicted_mean - self.sigma_factor * predicted_std
        upper = predicted_mean + self.sigma_factor * predicted_std
        
        # Check deviation
        if current_power < lower or current_power > upper:
            return True
        return False

    def detect(self, current_timestamp, current_power: float, predicted_mean: float = None, predicted_std: float = None) -> Dict:
        """
        Run full detection suite.
        """
        alerts = []
        is_anomaly = False
        
        # 1. Static Check
        if self.check_static(current_power):
            alerts.append(f"CRITICAL: Power {current_power:.2f}W exceeds safe limit {self.static_threshold}W")
            is_anomaly = True

        # 2. Dynamic Check (if prediction available)
        deviation = 0.0
        if predicted_mean is not None and predicted_std is not None:
            pm, ps = float(predicted_mean), float(predicted_std)
            if self.check_dynamic(current_power, pm, ps):
                alerts.append(f"WARNING: Abnormal behavior detected. Value {current_power:.2f}W is outside range [{pm - self.sigma_factor*ps:.2f}, {pm + self.sigma_factor*ps:.2f}]")
                is_anomaly = True
                deviation = abs(current_power - pm)

        result = {
            'timestamp': current_timestamp,
            'is_anomaly': is_anomaly,
            'power': current_power,
            'alerts': alerts,
            'deviation': deviation
        }
        
        if is_anomaly:
            self.history.append(result)
            
        return result

    def get_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
