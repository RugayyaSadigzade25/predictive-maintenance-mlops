import os
from datetime import datetime

import pandas as pd


LOG_FILE = "logs/drift_log.csv"


def log_drift_results(drift_results):
    """
    Save drift detection results to a CSV log.
    """

    rows = []

    for _, result in drift_results.iterrows():

        rows.append({
            "timestamp": datetime.now().isoformat(),
            "feature": result["feature"],
            "reference_mean": result["reference_mean"],
            "new_mean": result["new_mean"],
            "mean_difference": result["mean_difference"],
            "reference_std": result["reference_std"],
            "new_std": result["new_std"],
            "std_difference": result["std_difference"],
            "ks_statistic": result["ks_statistic"],
            "p_value": result["p_value"],
            "drift_detected": result["drift_detected"],
        })

    new_data = pd.DataFrame(rows)

    os.makedirs("logs", exist_ok=True)

    if os.path.exists(LOG_FILE):
        new_data.to_csv(
            LOG_FILE,
            mode="a",
            header=False,
            index=False
        )
    else:
        new_data.to_csv(
            LOG_FILE,
            index=False
        )