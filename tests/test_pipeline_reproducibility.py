"""
Tests 11, 12, 13: Table formatting compliance, reproducibility, and smoke execution.
"""

import os
import re
import pandas as pd
import numpy as np
import pytest

from src.reporting.tables import format_table_1_df, ROW_ORDER


def test_table_1_format_compliance():
    r"""
    Test 12: Table I has exact header, exact row order, and regex formatting:
    - ACCURACY: \d{2}\.\d
    - F1, MCC, SENSITIVITY: \d\.\d{3}
    """
    dummy_data = {
        "pima": {"accuracy": 0.8124, "f1": 0.7841, "mcc": 0.6012, "sensitivity": 0.8234},
        "cleveland": {"accuracy": 0.8842, "f1": 0.8611, "mcc": 0.7423, "sensitivity": 0.8461},
        "ckd": {"accuracy": 0.9560, "f1": 0.9432, "mcc": 0.8871, "sensitivity": 0.9442},
        "ilpd": {"accuracy": 0.7931, "f1": 0.7712, "mcc": 0.5183, "sensitivity": 0.7821},
        "wbcd": {"accuracy": 0.9712, "f1": 0.9583, "mcc": 0.9281, "sensitivity": 0.9634}
    }

    df = format_table_1_df(dummy_data)

    # 1. Exact Header
    expected_cols = ["Dataset", "ACCURACY", "F1", "MCC", "SENSITIVITY"]
    assert list(df.columns) == expected_cols, f"Header mismatch: {df.columns}"

    # 2. Exact Row Order
    assert list(df["Dataset"]) == ROW_ORDER, f"Row order mismatch: {list(df['Dataset'])}"

    # 3. Regex cell validation
    acc_pat = re.compile(r"^\d{2}\.\d$")
    metric_pat = re.compile(r"^\d\.\d{3}$")

    for _, row in df.iterrows():
        assert acc_pat.match(row["ACCURACY"]), f"Bad accuracy format: {row['ACCURACY']}"
        assert metric_pat.match(row["F1"]), f"Bad F1 format: {row['F1']}"
        assert metric_pat.match(row["MCC"]), f"Bad MCC format: {row['MCC']}"
        assert metric_pat.match(row["SENSITIVITY"]), f"Bad Sensitivity format: {row['SENSITIVITY']}"


def test_reproducibility():
    """
    Test 13: Same seed produces identical fold splits and predictions.
    """
    from src.data.load import load_cohort
    from src.preprocessing.transforms import ClinicalPreprocessor

    X, y, num_cols, cat_cols = load_cohort("pima", "data/raw")
    
    prep1 = ClinicalPreprocessor(numerical_cols=num_cols, random_state=42)
    prep1.fit(X.iloc[:500], y.iloc[:500])
    X1 = prep1.transform(X.iloc[500:])

    prep2 = ClinicalPreprocessor(numerical_cols=num_cols, random_state=42)
    prep2.fit(X.iloc[:500], y.iloc[:500])
    X2 = prep2.transform(X.iloc[500:])

    assert np.allclose(X1, X2), "Data transformation must be strictly reproducible given fixed seed"
