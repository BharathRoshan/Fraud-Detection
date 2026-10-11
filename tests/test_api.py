import numpy as np
from fastapi.testclient import TestClient

from src.api import create_app


class DummyModel:
    feature_names_in_ = np.array(["Amount", "Time"])

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


def make_client():
    app = create_app(model_path="missing-test-model.joblib")
    app.state.model = DummyModel()
    app.state.feature_names = ["Amount", "Time"]
    return TestClient(app)


def test_predict_returns_probability_and_class():
    response = make_client().post(
        "/predict", json={"features": {"Amount": 12.5, "Time": 42.0}}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == 1
    assert payload["fraud_probability"] == 0.8
    assert payload["model_version"] == "in-memory"


def test_predict_rejects_missing_feature():
    response = make_client().post("/predict", json={"features": {"Amount": 12.5}})
    assert response.status_code == 422
    assert "missing_features" in response.json()["detail"]


def test_predict_rejects_extra_feature():
    response = make_client().post(
        "/predict", json={"features": {"Amount": 12.5, "Time": 42.0, "unexpected": 1}}
    )
    assert response.status_code == 422


def test_health_reports_ready_when_model_is_injected():
    response = make_client().get("/health")
    assert response.status_code == 200
    assert response.json()["model_available"] is True
