import pandas as pd

from src.drift_detection import calculate_drift
from src.monitoring_logger import log_drift_results


MONITORING_WINDOW_SIZE = 100


class DataMonitor:

    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.observations = []

    def add_observation(self, observation):
        """
        Add one new machine observation.
        """
        self.observations.append(observation)

    def should_check_drift(self):
        """
        Check whether enough production observations
        have been collected.
        """
        return len(self.observations) >= MONITORING_WINDOW_SIZE

    def check_drift(self):
        """
        Compare the collected production observations
        against the reference dataset and log the drift results.
        """

        if not self.should_check_drift():
            return None

        production_data = pd.DataFrame(self.observations)

        results = calculate_drift(
            self.reference_data,
            production_data
        )

        log_drift_results(results)

        return results

    def reset(self):
        """
        Clear the current monitoring window.
        """
        self.observations = []