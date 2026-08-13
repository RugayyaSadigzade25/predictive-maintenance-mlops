import pandas as pd


EXPECTED_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def validate_data(df: pd.DataFrame) -> None:
    """
    Validate the structure and basic quality of machine data.

    Raises:
        ValueError: If the data does not meet the expected schema.
    """

    # 1. Check that all required columns exist
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 2. Check for missing values
    missing_values = df[EXPECTED_COLUMNS].isnull().sum()

    columns_with_missing_values = missing_values[
        missing_values > 0
    ]

    if not columns_with_missing_values.empty:
        raise ValueError(
            "Missing values detected: "
            f"{columns_with_missing_values.to_dict()}"
        )

    # 3. Check the machine type
    valid_types = {"L", "M", "H"}

    invalid_types = set(df["Type"].unique()) - valid_types

    if invalid_types:
        raise ValueError(
            f"Invalid machine type(s): {invalid_types}"
        )

    # 4. Check numerical columns
    numerical_columns = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    for column in numerical_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"Column '{column}' must contain numerical values."
            )

    print("Data validation passed.")