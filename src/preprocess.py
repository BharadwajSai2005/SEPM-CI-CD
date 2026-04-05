import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib, os, json

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# Clinical valid ranges for validation
FEATURE_RANGES = {
    "age":      (20,  80),
    "trestbps": (80,  200),
    "chol":     (100, 600),
    "thalach":  (60,  220),
    "oldpeak":  (0.0, 7.0),
    "ca":       (0,   4),
    "sex":      (0,   1),
    "cp":       (0,   3),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "exang":    (0,   1),
    "slope":    (0,   2),
    "thal":     (0,   3),
}

def load_and_validate(path="data/heart.csv"):
    """Load Cleveland Heart Disease CSV and run clinical sanity checks."""
    col_names = FEATURE_NAMES + ["target"]
    df = pd.read_csv(path, header=None, names=col_names, na_values="?")

    # Replace multi-class target with binary (0=no disease, 1=disease)
    df["target"] = (df["target"] > 0).astype(int)

    print(f"[INFO] Loaded {df.shape[0]} rows, {df.shape[1]} cols")
    print(f"[INFO] Missing values:\n{df.isnull().sum()[df.isnull().sum()>0]}")

    # --- Validation checks ---
    assert df.shape[1] == 14, f"Expected 14 cols, got {df.shape[1]}"
    assert df.shape[0] >= 200, "Dataset too small — check the CSV"
    assert df["target"].nunique() == 2, "Target must be binary after conversion"

    # Check clinical ranges (ignoring NaN)
    for col, (lo, hi) in FEATURE_RANGES.items():
        valid = df[col].dropna()
        assert valid.between(lo, hi).all(), \
            f"Feature '{col}' has out-of-range values: min={valid.min()}, max={valid.max()}"

    disease_rate = df["target"].mean()
    print(f"[OK] Disease prevalence: {disease_rate:.1%}")
    print(f"[OK] All {len(FEATURE_RANGES)} clinical range checks passed")
    return df

def preprocess(df):
    """Impute, encode, scale, SMOTE-balance and split."""
    # Impute median for ca and thal (only ~6 rows affected)
    df["ca"]   = df["ca"].fillna(df["ca"].median())
    df["thal"] = df["thal"].fillna(df["thal"].median())

    X = df[FEATURE_NAMES].values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # SMOTE on training set only — never touch the test set
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_sc, y_train)

    print(f"[OK] After SMOTE — Train: {X_train_res.shape}, Test: {X_test_sc.shape}")
    print(f"[OK] Class balance post-SMOTE: "
          f"{np.bincount(y_train_res)}")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train.npy", X_train_res)
    np.save("data/processed/X_test.npy",  X_test_sc)
    np.save("data/processed/y_train.npy", y_train_res)
    np.save("data/processed/y_test.npy",  y_test)

    # Save raw (unscaled) test set for drift monitoring later
    np.save("data/processed/X_test_raw.npy", X_test)

    joblib.dump(scaler, "models/scaler.pkl")
    print("[OK] Scaler saved to models/scaler.pkl")

    return X_train_res, X_test_sc, y_train_res, y_test

if __name__ == "__main__":
    df = load_and_validate()
    preprocess(df)