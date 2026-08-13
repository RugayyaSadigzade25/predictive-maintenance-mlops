import pandas as pd

from src.monitoring import DataMonitor


def test_monitoring_creates_drift_results():

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

    assert monitor.should_check_drift() is True

    results = monitor.check_drift()

    assert results is not None
    assert len(results) == 5

    assert "feature" in results.columns
    assert "drift_detected" in results.columns