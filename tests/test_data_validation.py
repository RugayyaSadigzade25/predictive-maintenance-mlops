import pandas as pd
import pytest

from src.data_validation import validate_data


def valid_machine_data():
    return pd.DataFrame([
        {
            "Type": "L",
            "Air temperature [K]": 300.1,
            "Process temperature [K]": 310.2,
            "Rotational speed [rpm]": 1500,
            "Torque [Nm]": 45.0,
            "Tool wear [min]": 100,
        }
    ])


def test_valid_data_passes():
    df = valid_machine_data()

    validate_data(df)


def test_missing_column_fails():
    df = valid_machine_data()

    df = df.drop(columns=["Torque [Nm]"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_data(df)


def test_missing_value_fails():
    df = valid_machine_data()

    df.loc[0, "Torque [Nm]"] = None

    with pytest.raises(ValueError, match="Missing values detected"):
        validate_data(df)


def test_invalid_machine_type_fails():
    df = valid_machine_data()

    df.loc[0, "Type"] = "X"

    with pytest.raises(ValueError, match="Invalid machine type"):
        validate_data(df)


def test_non_numeric_value_fails():
    df = valid_machine_data()

    df.loc[0, "Torque [Nm]"] = "banana"

    with pytest.raises(
        ValueError,
        match="must contain numerical values"
    ):
        validate_data(df)