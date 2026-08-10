from fastapi.testclient import TestClient # pyright: ignore[reportMissingImports]
from src.predict import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Predictive Maintenance API is running"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "Type": "L",
            "air_temperature": 300.1,
            "process_temperature": 310.2,
            "rotational_speed": 1500,
            "torque": 45.0,
            "tool_wear": 100,
        },
    )

    assert response.status_code == 200
    assert "machine_failure" in response.json()
    assert response.json()["machine_failure"] in [0, 1]