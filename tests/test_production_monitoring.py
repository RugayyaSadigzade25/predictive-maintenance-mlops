from fastapi.testclient import TestClient

from src.predict import app, monitor


client = TestClient(app)


def test_production_monitoring_detects_drift():

    monitor.reset()

    for _ in range(100):

        response = client.post(
            "/predict",
            json={
                "Type": "M",
                "air_temperature": 300.0,
                "process_temperature": 310.0,
                "rotational_speed": 2000,
                "torque": 56.0,
                "tool_wear": 100,
            },
        )

        assert response.status_code == 200

    result = response.json()

    assert "machine_failure" in result
    assert "drift_detected" in result

    assert result["drift_detected"] is True