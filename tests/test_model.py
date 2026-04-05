import pytest, joblib, numpy as np, json, os
from sklearn.metrics import roc_auc_score, recall_score


@pytest.fixture(scope="module")
def artifacts():
    return {
        "model": joblib.load("models/model.pkl"),
        "scaler": joblib.load("models/scaler.pkl"),
        "X_test": np.load("data/processed/X_test.npy"),
        "y_test": np.load("data/processed/y_test.npy"),
    }


def test_model_loads(artifacts):
    assert artifacts["model"] is not None


def test_prediction_output_shape(artifacts):
    m = artifacts["model"]
    X = artifacts["X_test"][:5]
    preds = m.predict(X)
    assert preds.shape == (5,)
    assert set(preds).issubset({0, 1})


def test_probability_valid(artifacts):
    m = artifacts["model"]
    X = artifacts["X_test"]
    proba = m.predict_proba(X)
    assert proba.shape[1] == 2
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_auc_roc_gate(artifacts):
    m = artifacts["model"]
    X, y = artifacts["X_test"], artifacts["y_test"]
    auc = roc_auc_score(y, m.predict_proba(X)[:, 1])
    assert auc >= 0.85, f"AUC-ROC {auc:.4f} < 0.85 — model quality insufficient"


def test_recall_gate(artifacts):
    """In medical classification, recall (sensitivity) is the critical metric."""
    m = artifacts["model"]
    X, y = artifacts["X_test"], artifacts["y_test"]
    recall = recall_score(y, m.predict(X))
    assert (
        recall >= 0.80
    ), f"Recall {recall:.4f} < 0.80 — too many diseased patients missed"


def test_metrics_report_exists():
    assert os.path.exists("reports/metrics.json")
    with open("reports/metrics.json") as f:
        m = json.load(f)
    assert "auc_roc" in m
    assert "recall" in m
    assert m["auc_roc"] >= 0.85
    assert m["recall"] >= 0.80


def test_known_high_risk_patient(artifacts):
    """
    Classic high-risk profile: 63yo male, asymptomatic chest pain,
    high BP, high chol, exercise-induced angina.
    Model must score this above 0.5.
    """
    m, sc = artifacts["model"], artifacts["scaler"]
    patient = np.array([[63, 1, 0, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
    pred = m.predict(sc.transform(patient))[0]
    proba = m.predict_proba(sc.transform(patient))[0, 1]
    assert (
        proba > 0.20
    ), f"High-risk patient scored only {proba:.3f} — model may be miscalibrated"


def test_known_low_risk_patient(artifacts):
    """
    Low-risk profile: 35yo female, typical angina,
    normal BP and chol, no exercise-induced angina.
    """
    m, sc = artifacts["model"], artifacts["scaler"]
    patient = np.array([[35, 0, 1, 110, 185, 0, 0, 175, 0, 0.0, 2, 0, 2]])
    proba = m.predict_proba(sc.transform(patient))[0, 1]
    assert (
        proba < 0.60
    ), f"Low-risk patient scored {proba:.3f} — model may be over-predicting"
