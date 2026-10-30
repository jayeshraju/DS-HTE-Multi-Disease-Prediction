"""
Cross-Table Invariants & Audit Consistency Suite.
Follows Section A5:
- Invariant 1: "Full DS-HTE" row in ablation == Table II values for the same metrics (exactly).
- Invariant 2: Table I CV mean == Table II (accuracy, F1, MCC, sensitivity).
- Invariant 3: Calibration table "after" ECE/Brier == Table II ECE/Brier.
- Invariant 4: For every metric, the reported mean lies inside its bootstrap CI.
- Invariant 5: Table I follows exact regex formatting and row ordering.
"""

import os
import re
import json
import pytest
import pandas as pd
import numpy as np


TABLES_DIR = "results/tables"
ROW_ORDER = ["PIMA", "Cleveland", "CKD", "ILPD", "WBCD"]


@pytest.fixture(scope="module")
def loaded_tables():
    if not os.path.exists(os.path.join(TABLES_DIR, "table_1_cv_mean.csv")):
        pytest.skip("Results tables not yet generated.")

    t1_cv = pd.read_csv(os.path.join(TABLES_DIR, "table_1_cv_mean.csv"), dtype=str)
    t1_ext = pd.read_csv(os.path.join(TABLES_DIR, "table_1_extended.csv"), dtype=str)
    t2 = pd.read_csv(os.path.join(TABLES_DIR, "table_2_dshte_metrics.csv"))
    t4 = pd.read_csv(os.path.join(TABLES_DIR, "table_4_statistical_tests.csv")) if os.path.exists(os.path.join(TABLES_DIR, "table_4_statistical_tests.csv")) else None
    t5 = pd.read_csv(os.path.join(TABLES_DIR, "table_5_ablation.csv")) if os.path.exists(os.path.join(TABLES_DIR, "table_5_ablation.csv")) else None
    t8 = pd.read_csv(os.path.join(TABLES_DIR, "table_8_calibration.csv")) if os.path.exists(os.path.join(TABLES_DIR, "table_8_calibration.csv")) else None

    return {
        "t1_cv": t1_cv,
        "t1_ext": t1_ext,
        "t2": t2,
        "t4": t4,
        "t5": t5,
        "t8": t8
    }


def test_table_1_format_and_order(loaded_tables):
    df_t1 = loaded_tables["t1_cv"]
    expected_cols = ["Dataset", "ACCURACY", "F1", "MCC", "SENSITIVITY"]
    assert list(df_t1.columns) == expected_cols
    assert list(df_t1["Dataset"]) == ROW_ORDER

    acc_pat = re.compile(r"^\d{2}\.\d$")
    metric_pat = re.compile(r"^\d\.\d{3}$")

    for _, row in df_t1.iterrows():
        assert acc_pat.match(str(row["ACCURACY"]))
        assert metric_pat.match(str(row["F1"]))
        assert metric_pat.match(str(row["MCC"]))
        assert metric_pat.match(str(row["SENSITIVITY"]))


def test_table_1_cv_mean_equals_table_2(loaded_tables):
    """
    Invariant 2: Table I CV mean == Table II (accuracy, F1, MCC, sensitivity).
    """
    t1 = loaded_tables["t1_cv"]
    t2 = loaded_tables["t2"]

    for c_name in ROW_ORDER:
        row1 = t1[t1["Dataset"] == c_name].iloc[0]
        row2 = t2[t2["Dataset"] == c_name].iloc[0]

        # Table I accuracy is percentage: e.g., 74.3 vs 0.743
        acc_t1 = float(row1["ACCURACY"])
        acc_t2_mean = float(str(row2["ACCURACY"]).split("+-")[0].strip()) * 100.0
        assert np.isclose(acc_t1, acc_t2_mean, atol=0.15), f"Accuracy mismatch in {c_name}: {acc_t1} vs {acc_t2_mean}"

        f1_t1 = float(row1["F1"])
        f1_t2_mean = float(str(row2["F1"]).split("+-")[0].strip())
        assert np.isclose(f1_t1, f1_t2_mean, atol=0.002), f"F1 mismatch in {c_name}: {f1_t1} vs {f1_t2_mean}"

        mcc_t1 = float(row1["MCC"])
        mcc_t2_mean = float(str(row2["MCC"]).split("+-")[0].strip())
        assert np.isclose(mcc_t1, mcc_t2_mean, atol=0.002), f"MCC mismatch in {c_name}: {mcc_t1} vs {mcc_t2_mean}"

        sens_t1 = float(row1["SENSITIVITY"])
        sens_t2_mean = float(str(row2["SENSITIVITY"]).split("+-")[0].strip())
        assert np.isclose(sens_t1, sens_t2_mean, atol=0.002), f"Sensitivity mismatch in {c_name}: {sens_t1} vs {sens_t2_mean}"


def test_ablation_full_dshte_equals_table_2(loaded_tables):
    """
    Invariant 1: "Full DS-HTE" row in ablation == Table II values for the same metrics.
    """
    t5 = loaded_tables["t5"]
    t2 = loaded_tables["t2"]
    if t5 is None:
        pytest.skip("Table 5 not generated yet.")

    for c_name in ROW_ORDER:
        sub_t5 = t5[(t5["Dataset"] == c_name) & (t5["Ablation Variant"] == "Full DS-HTE")]
        if len(sub_t5) == 0:
            continue
        row5 = sub_t5.iloc[0]
        row2 = t2[t2["Dataset"] == c_name].iloc[0]

        acc5 = float(row5["Accuracy (%)"])
        acc2 = float(str(row2["ACCURACY"]).split("+-")[0].strip()) * 100.0
        assert np.isclose(acc5, acc2, atol=0.15)

        f1_5 = float(row5["F1"])
        f1_2 = float(str(row2["F1"]).split("+-")[0].strip())
        assert np.isclose(f1_5, f1_2, atol=0.002)

        mcc5 = float(row5["MCC"])
        mcc2 = float(str(row2["MCC"]).split("+-")[0].strip())
        assert np.isclose(mcc5, mcc2, atol=0.002)

        auc5 = float(row5["ROC-AUC"])
        auc2 = float(str(row2["ROC_AUC"]).split("+-")[0].strip())
        assert np.isclose(auc5, auc2, atol=0.002)

        brier5 = float(row5["Brier"])
        brier2 = float(str(row2["BRIER"]).split("+-")[0].strip())
        assert np.isclose(brier5, brier2, atol=0.002)


def test_calibration_after_equals_table_2(loaded_tables):
    """
    Invariant 3: Calibration table "after" ECE/Brier == Table II ECE/Brier.
    """
    t8 = loaded_tables["t8"]
    t2 = loaded_tables["t2"]
    if t8 is None:
        pytest.skip("Table 8 not generated yet.")

    for c_name in ROW_ORDER:
        sub_t8 = t8[t8["Dataset"] == c_name]
        if len(sub_t8) == 0:
            continue
        row8 = sub_t8.iloc[0]
        row2 = t2[t2["Dataset"] == c_name].iloc[0]

        ece_after = float(row8["ECE After"])
        ece_t2 = float(str(row2["ECE"]).split("+-")[0].strip())
        assert np.isclose(ece_after, ece_t2, atol=0.002)

        brier_after = float(row8["Brier After"])
        brier_t2 = float(str(row2["BRIER"]).split("+-")[0].strip())
        assert np.isclose(brier_after, brier_t2, atol=0.002)


def test_mean_inside_ci_for_all_metrics(loaded_tables):
    """
    Invariant 4: For every metric in Table II, lower <= mean <= upper holds.
    """
    t2 = loaded_tables["t2"]
    metric_cols = [c for c in t2.columns if c != "Dataset"]

    ci_pattern = re.compile(r"^([\d\.]+)\s*\+-\s*[\d\.]+\s*\[([\d\.]+),\s*([\d\.]+)\]$")

    for _, row in t2.iterrows():
        for col in metric_cols:
            val_str = str(row[col]).strip()
            m = ci_pattern.match(val_str)
            assert m is not None, f"String '{val_str}' in {col} does not match CI pattern"
            mean_v = float(m.group(1))
            lower_v = float(m.group(2))
            upper_v = float(m.group(3))
            assert lower_v <= mean_v <= upper_v, f"Violation in {row['Dataset']} {col}: {lower_v} <= {mean_v} <= {upper_v}"
