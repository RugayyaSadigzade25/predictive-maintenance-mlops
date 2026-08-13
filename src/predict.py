from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]
from pydantic import BaseModel
import pandas as pd
import joblib

from src.data_validation import validate_data
from src.monitoring import DataMonitor


# Load trained model
model = joblib.load("model/model.joblib")


# Load reference data for monitoring
reference_data = pd.read_csv("data/ai4i2020.csv")


# Create data monitor
monitor = DataMonitor(reference_data)


# Create API
app = FastAPI(title="Predictive Maintenance API")


# Input data structure
class MachineData(BaseModel):
    Type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: int
    torque: float
    tool_wear: int


@app.get("/")
def home():
    return {
        "message": "Predictive Maintenance API is running"
    }


@app.post("/predict")
def predict(machine: MachineData):

    # Convert API input into the same column structure
    # used during model training
    input_data = pd.DataFrame([{
        "Type": machine.Type,
        "Air temperature [K]": machine.air_temperature,
        "Process temperature [K]": machine.process_temperature,
        "Rotational speed [rpm]": machine.rotational_speed,
        "Torque [Nm]": machine.torque,
        "Tool wear [min]": machine.tool_wear,
    }])

    # Validate incoming machine data
    try:
        validate_data(input_data)

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # Add valid observation to monitoring system
    monitor.add_observation(
        input_data.iloc[0].to_dict()
    )

    # Check whether enough observations
    # have been collected for drift detection
    drift_detected = False

    if monitor.should_check_drift():

        drift_results = monitor.check_drift()

        if drift_results is not None:
            drift_detected = bool(
                drift_results["drift_detected"].any()
            )

        # Start a new monitoring window
        monitor.reset()

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "machine_failure": int(prediction),
        "drift_detected": drift_detected
    }