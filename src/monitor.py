import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping
import json, os

FEATURE_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def run_drift_report(
    reference_path="data/processed/X_test_raw.npy",
    current_path="data/processed/X_test_raw.npy",
):
    """
    In production, current_path would point to a live inference log.
    Here we simulate by adding small Gaussian noise to the test set.
    """
    X_ref = np.load(reference_path)
    X_cur = np.load(current_path).copy()

    # Simulate production drift: small noise on numerical features
    rng = np.random.default_rng(seed=99)
    num_idx = [FEATURE_NAMES.index(f) for f in NUMERICAL_FEATURES]
    X_cur[:, num_idx] += rng.normal(0, 0.5, (X_cur.shape[0], len(num_idx)))

    ref_df = pd.DataFrame(X_ref, columns=FEATURE_NAMES)
    cur_df = pd.DataFrame(X_cur, columns=FEATURE_NAMES)

    col_map = ColumnMapping(
        numerical_features=NUMERICAL_FEATURES, categorical_features=CATEGORICAL_FEATURES
    )

    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=col_map)

    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/drift_report.html")

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]

    drifted_features = [
        col
        for col, stats in result["metrics"][0]["result"]["drift_by_columns"].items()
        if stats["drift_detected"]
    ]

    summary = {
        "dataset_drift": drift_detected,
        "drifted_features": drifted_features,
        "n_drifted": len(drifted_features),
    }

    with open("reports/drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DRIFT] Dataset drift detected: {drift_detected}")
    print(f"[DRIFT] Drifted features ({len(drifted_features)}): {drifted_features}")

    if drift_detected:
        print("[WARN] Drift threshold exceeded — retrain recommended.")

    return summary


if __name__ == "__main__":
    run_drift_report()
