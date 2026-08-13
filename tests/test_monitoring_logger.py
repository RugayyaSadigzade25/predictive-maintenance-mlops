import pandas as pd

from src.monitoring_logger import log_drift_results


def test_log_drift_results(tmp_path):

    drift_results = pd.DataFrame([
        {
            "feature": "Torque [Nm]",
            "reference_mean": 40.0,
            "new_mean": 56.0,
            "mean_difference": 0.4,
            "reference_std": 10.0,
            "new_std": 14.0,
            "std_difference": 0.4,
            "ks_statistic": 0.5,
            "p_value": 0.001,
            "drift_detected": True,
        }
    ])

    log_file = tmp_path / "drift_log.csv"

    # Temporarily replace the logger's output location
    import src.monitoring_logger as logger

    logger.LOG_FILE = str(log_file)

    log_drift_results(drift_results)

    assert log_file.exists()

    logged_data = pd.read_csv(log_file)

    assert len(logged_data) == 1
    assert logged_data.iloc[0]["feature"] == "Torque [Nm]"
    assert bool(logged_data.iloc[0]["drift_detected"]) is True