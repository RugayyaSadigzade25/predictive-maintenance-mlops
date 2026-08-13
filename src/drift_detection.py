import pandas as pd
from scipy.stats import ks_2samp


REFERENCE_DATA_PATH = "data/ai4i2020.csv"
PRODUCTION_DATA_PATH = "data/production_data.csv"

DRIFT_THRESHOLD = 0.10
P_VALUE_THRESHOLD = 0.05

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def load_reference_data():
    return pd.read_csv(REFERENCE_DATA_PATH)


def load_production_data():
    return pd.read_csv(PRODUCTION_DATA_PATH)


def get_features(data):
    return data[FEATURES]


def calculate_drift(reference_data, new_data):
    results = []

    for feature in NUMERICAL_FEATURES:

        reference_values = reference_data[feature].dropna()
        new_values = new_data[feature].dropna()

        reference_mean = reference_values.mean()
        new_mean = new_values.mean()

        reference_std = reference_values.std()
        new_std = new_values.std()

        mean_difference = (
            abs(new_mean - reference_mean)
            / reference_mean
        )

        std_difference = (
            abs(new_std - reference_std)
            / reference_std
        )

        # Kolmogorov-Smirnov statistical test
        ks_statistic, p_value = ks_2samp(
            reference_values,
            new_values
        )

        drift_detected = (
            mean_difference > DRIFT_THRESHOLD
            or p_value < P_VALUE_THRESHOLD # type: ignore
        )

        results.append({
            "feature": feature,
            "reference_mean": reference_mean,
            "new_mean": new_mean,
            "mean_difference": mean_difference,
            "reference_std": reference_std,
            "new_std": new_std,
            "std_difference": std_difference,
            "ks_statistic": ks_statistic,
            "p_value": p_value,
            "drift_detected": drift_detected,
        })

    return pd.DataFrame(results)


def print_drift_report(results):

    print("\n========================================")
    print("           DATA DRIFT REPORT")
    print("========================================")

    overall_drift = False

    for _, row in results.iterrows():

        if row["drift_detected"]:
            status = "DRIFT ⚠️"
            overall_drift = True
        else:
            status = "NORMAL ✅"

        print(
            f"{row['feature']:<25} "
            f"Mean change: {row['mean_difference'] * 100:>6.2f}%   "
            f"p-value: {row['p_value']:.4f}   "
            f"{status}"
        )

    print("----------------------------------------")

    if overall_drift:
        print("Overall status: DRIFT DETECTED ⚠️")
    else:
        print("Overall status: NO DRIFT DETECTED ✅")

    print("========================================")


if __name__ == "__main__":

    reference_data = get_features(
        load_reference_data()
    )

    production_data = get_features(
        load_production_data()
    )

    drift_results = calculate_drift(
        reference_data,
        production_data
    )

    print_drift_report(drift_results)