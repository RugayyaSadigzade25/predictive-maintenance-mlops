from fastapi import FastAPI # pyright: ignore[reportMissingImports]
from pydantic import BaseModel
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/model.joblib")

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

    # Make prediction
    prediction = model.predict(input_data)[0]

    return {
        "machine_failure": int(prediction)
    }