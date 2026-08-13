import pandas as pd

from src.monitoring import DataMonitor


def test_monitor_starts_empty():

    reference_data = pd.DataFrame({
        "Air temperature [K]": [300, 301],
        "Process temperature [K]": [310, 311],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40, 50],
        "Tool wear [min]": [100, 110],
    })

    monitor = DataMonitor(reference_data)

    assert len(monitor.observations) == 0


def test_monitor_collects_observations():

    reference_data = pd.DataFrame({
        "Air temperature [K]": [300, 301],
        "Process temperature [K]": [310, 311],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40, 50],
        "Tool wear [min]": [100, 110],
    })

    monitor = DataMonitor(reference_data)

    observation = {
        "Air temperature [K]": 300,
        "Process temperature [K]": 310,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 40,
        "Tool wear [min]": 100,
    }

    monitor.add_observation(observation)

    assert len(monitor.observations) == 1


def test_monitor_window():

    reference_data = pd.DataFrame({
        "Air temperature [K]": [300, 301],
        "Process temperature [K]": [310, 311],
        "Rotational speed [rpm]": [1500, 1600],
        "Torque [Nm]": [40, 50],
        "Tool wear [min]": [100, 110],
    })

    monitor = DataMonitor(reference_data)

    for _ in range(100):

        observation = {
            "Air temperature [K]": 300,
            "Process temperature [K]": 310,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 40,
            "Tool wear [min]": 100,
        }

        monitor.add_observation(observation)

    assert monitor.should_check_drift() is True