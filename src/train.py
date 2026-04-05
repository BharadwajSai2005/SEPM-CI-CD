import numpy as np
import mlflow, mlflow.sklearn
import joblib, json, os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

EXPERIMENT = "heart-disease-classification"


def train_model():
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment(EXPERIMENT)

    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    candidates = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        ),
    }

    best_model, best_name, best_auc = None, None, 0.0

    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params(model.get_params())
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "recall": round(recall_score(y_test, y_pred), 4),
                "f1": round(f1_score(y_test, y_pred), 4),
                "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
            }
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")

            cm = confusion_matrix(y_test, y_pred)
            print(f"\n[{name}]")
            print(f"  AUC-ROC : {metrics['auc_roc']}")
            print(f"  Recall  : {metrics['recall']}")
            print(f"  F1      : {metrics['f1']}")
            print(f"  Accuracy: {metrics['accuracy']}")
            print(f"  Confusion matrix:\n{cm}")
            print(
                classification_report(
                    y_test, y_pred, target_names=["no disease", "disease"]
                )
            )

            # Select best by AUC-ROC (clinically appropriate)
            if metrics["auc_roc"] > best_auc:
                best_auc = metrics["auc_roc"]
                best_model = model
                best_name = name
                best_metrics = metrics

    # Persist best model
    joblib.dump(best_model, "models/model.pkl")

    os.makedirs("reports", exist_ok=True)
    report = {"model": best_name, **best_metrics}
    with open("reports/metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[BEST] {best_name}")
    print(f"       AUC-ROC={best_auc:.4f}")
    return best_model, report


if __name__ == "__main__":
    train_model()
