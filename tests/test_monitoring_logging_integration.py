import pandas as pd

from src.monitoring import DataMonitor
import src.monitoring_logger as logger


def test_monitoring_logs_drift_results(tmp_path):

    reference_data = pd.DataFrame({
        "Air temperature [K]": [298.0] * 100,
        "Process temperature [K]": [308.0] * 100,
        "Rotational speed [rpm]": [1500.0] * 100,
        "Torque [Nm]": [40.0] * 100,
        "Tool wear [min]": [100.0] * 100,
    })

    monitor = DataMonitor(reference_data)

    for _ in range(100):
        monitor.add_observation({
            "Air temperature [K]": 310.0,
            "Process temperature [K]": 320.0,
            "Rotational speed [rpm]": 1800.0,
            "Torque [Nm]": 55.0,
            "Tool wear [min]": 200.0,
        })

    log_file = tmp_path / "drift_log.csv"

    logger.LOG_FILE = str(log_file)

    results = monitor.check_drift()

    assert results is not None
    assert log_file.exists()

    logged_data = pd.read_csv(log_file)

    assert len(logged_data) == 5
    assert "feature" in logged_data.columns
    assert "drift_detected" in logged_data.columns