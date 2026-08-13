import pandas as pd


REFERENCE_DATA_PATH = "data/ai4i2020.csv"
PRODUCTION_DATA_PATH = "data/production_data.csv"


def create_production_data():
    data = pd.read_csv(REFERENCE_DATA_PATH)

    production_data = data.copy()

    # Simulate a change in machine operating conditions
    production_data["Rotational speed [rpm]"] = (
        production_data["Rotational speed [rpm]"] * 1.3
    )

    production_data["Torque [Nm]"] = (
        production_data["Torque [Nm]"] * 1.4
    )

    production_data.to_csv(
        PRODUCTION_DATA_PATH,
        index=False
    )

    print("Production data created successfully.")
    print(f"Rows: {len(production_data)}")
    print(f"Saved to: {PRODUCTION_DATA_PATH}")


if __name__ == "__main__":
    create_production_data()