import os
import pytest
from fastapi.testclient import TestClient
from api.main import app

SAMPLE = {
    "Age": 35, "Gender": "F", "Fever": 1, "Cough": 1, "Fatigue": 0,
    "Breathlessness": 0, "Chest_Pain": 0, "Nausea": 1, "Headache": 1,
    "BMI": 24.5, "BP_Systolic": 118, "BP_Diastolic": 78, "SpO2": 98,
    "Glucose_mg_dL": 95, "HbA1c": 5.4
}

@pytest.fixture(scope="module")
def client():
    # Load the real Week 6 artifact for integration tests.
    from api.predictor import HealthcarePredictor
    model_path = os.getenv("MEDSCAN_MODEL_PATH", "models/medscan_week6_best.joblib")
    app.state.test_predictor = HealthcarePredictor(model_path)
    import api.main as main_module
    main_module.predictor = app.state.test_predictor
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_prediction_returns_class_and_probabilities(client):
    response = client.post("/predict", json=SAMPLE)
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"]
    assert "probabilities" in body
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-5
    assert "not a medical diagnosis" in body["warning"]

def test_invalid_feature_is_rejected(client):
    invalid = dict(SAMPLE)
    invalid["Fever"] = 3
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
