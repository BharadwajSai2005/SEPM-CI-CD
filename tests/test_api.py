import pytest
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import app

client = TestClient(app)

HIGH_RISK_PATIENT = {
    "age": 63, "sex": 1, "cp": 0, "trestbps": 145,
    "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
    "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}

LOW_RISK_PATIENT = {
    "age": 35, "sex": 0, "cp": 1, "trestbps": 110,
    "chol": 185, "fbs": 0, "restecg": 0, "thalach": 175,
    "exang": 0, "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 2
}

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_metrics_endpoint():
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "auc_roc" in data
    assert "recall"  in data

def test_predict_returns_valid_structure():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ["disease", "no disease"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_category"] in ["low", "moderate", "high"]
    assert "no_disease" in body["probabilities"]
    assert "disease"    in body["probabilities"]

def test_probabilities_sum_to_one():
    r = client.post("/predict", json=HIGH_RISK_PATIENT)
    p = r.json()["probabilities"]
    total = p["no_disease"] + p["disease"]
    assert abs(total - 1.0) < 0.001

def test_invalid_age_rejected():
    bad = HIGH_RISK_PATIENT.copy()
    bad["age"] = 5   # below minimum of 20
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_invalid_cholesterol_rejected():
    bad = HIGH_RISK_PATIENT.copy()
    bad["chol"] = 50  # below minimum of 100
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_out_of_range_bp_rejected():
    bad = HIGH_RISK_PATIENT.copy()
    bad["trestbps"] = 300  # above max of 200
    r = client.post("/predict", json=bad)
    assert r.status_code == 422

def test_missing_field_rejected():
    incomplete = {k: v for k, v in HIGH_RISK_PATIENT.items() if k != "chol"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422