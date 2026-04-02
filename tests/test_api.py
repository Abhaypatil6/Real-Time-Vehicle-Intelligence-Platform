from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {"feature1": 0.5, "feature2": -0.8}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert "latency_ms" in body
    assert body["model_version"] == "v1"


def test_batch_predict():
    payload = {
        "instances": [
            {"feature1": 0.2, "feature2": 0.8},
            {"feature1": -1.2, "feature2": -0.4},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 2
