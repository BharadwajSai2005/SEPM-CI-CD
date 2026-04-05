import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocess import FEATURE_NAMES, FEATURE_RANGES, load_and_validate


@pytest.fixture(scope="module")
def df():
    return load_and_validate("data/heart.csv")


def test_row_count(df):
    assert df.shape[0] >= 200, "Need at least 200 patient records"


def test_column_count(df):
    assert df.shape[1] == 14


def test_binary_target(df):
    assert set(df["target"].unique()).issubset({0, 1})


def test_no_nulls_after_load(df):
    # Only ca and thal may have NaN — everything else must be complete
    optional_null_cols = {"ca", "thal"}
    for col in df.columns:
        if col not in optional_null_cols:
            assert df[col].isnull().sum() == 0, f"Unexpected nulls in '{col}'"


def test_clinical_ranges(df):
    for col, (lo, hi) in FEATURE_RANGES.items():
        valid = df[col].dropna()
        assert valid.between(lo, hi).all(), f"'{col}' has values outside [{lo}, {hi}]"


def test_disease_prevalence_reasonable(df):
    rate = df["target"].mean()
    assert 0.30 <= rate <= 0.70, f"Disease prevalence {rate:.1%} looks unrealistic"


def test_age_distribution(df):
    assert df["age"].median() > 40, "Median age should be > 40 for this dataset"
    assert df["age"].median() < 70


def test_sex_encoding(df):
    assert set(df["sex"].unique()).issubset({0, 1})


def test_processed_files_created():
    for fname in ["X_train.npy", "X_test.npy", "y_train.npy", "y_test.npy"]:
        assert os.path.exists(f"data/processed/{fname}"), f"Missing {fname}"
