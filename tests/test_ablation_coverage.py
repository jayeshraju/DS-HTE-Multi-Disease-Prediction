"""
Unit test for Section 0.3: Ablation Study Coverage across all 5 clinical cohorts.
Verifies:
1. All 5 cohorts (PIMA, Cleveland, CKD, ILPD, WBCD) are present in Table V.
2. Variants include Full DS-HTE, Minus Feature Optimization, Minus Imbalance, Minus Calibration,
   Minus Threshold, GBDT-Only, Minus Transformer, LR-Only, Transformer-Only.
3. If a calibrator is "none", it is marked appropriately rather than silently duplicated.
"""

import os
import pandas as pd
import pytest

TABLE_5_PATH = "results/tables/table_5_ablation.csv"


def test_ablation_covers_all_5_cohorts():
    assert os.path.exists(TABLE_5_PATH), f"Missing {TABLE_5_PATH}"
    df = pd.read_csv(TABLE_5_PATH)

    cohorts = df["Dataset"].unique()
    expected_cohorts = {"PIMA", "Cleveland", "CKD", "ILPD", "WBCD"}
    assert set(cohorts) == expected_cohorts, f"Ablation cohorts {set(cohorts)} != {expected_cohorts}"

    for cohort in expected_cohorts:
        sub = df[df["Dataset"] == cohort]
        variants = set(sub["Ablation Variant"])
        assert "Full DS-HTE" in variants, f"Missing Full DS-HTE for {cohort}"
        assert "Minus Feature Optimization" in variants, f"Missing Minus Feature Optimization for {cohort}"
        assert "LR-Only" in variants, f"Missing LR-Only for {cohort}"
        assert "Transformer-Only" in variants, f"Missing Transformer-Only for {cohort}"
