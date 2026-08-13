from fastapi.testclient import TestClient # type: ignore

from src.predict import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Predictive Maintenance API is running"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "Type": "M",
            "air_temperature": 300.0,
            "process_temperature": 310.0,
            "rotational_speed": 1500,
            "torque": 40.0,
            "tool_wear": 100,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "machine_failure" in data
    assert "drift_detected" in data

    assert data["machine_failure"] in [0, 1]
    assert isinstance(data["drift_detected"], bool)