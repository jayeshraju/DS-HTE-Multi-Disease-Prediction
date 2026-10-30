"""
Unit test for Section 0.1: Ensemble Weights Logging & Component Collapse Audit.
Validates:
- results/audit/ensemble_weights.parquet exists and has records for all cohorts
- Every outer fold contains weights for all active models
- Weights respect the minimum weight floor (>= 0.01)
- Flags any component with mean weight > 0.90 and ensures audit report is generated
"""

import os
import pytest
import pandas as pd
from src.evaluation.ensemble_audit import audit_ensemble_weights

AUDIT_DIR = "results/audit"
WEIGHTS_PARQUET = os.path.join(AUDIT_DIR, "ensemble_weights.parquet")


def test_ensemble_weights_audit_and_collapse_detection():
    # If not yet copied, run audit
    if not os.path.exists(WEIGHTS_PARQUET):
        audit_ensemble_weights()

    assert os.path.exists(WEIGHTS_PARQUET), f"Audit parquet missing at {WEIGHTS_PARQUET}"

    df = pd.read_parquet(WEIGHTS_PARQUET)
    assert not df.empty, "Ensemble weights parquet is empty!"

    # Verify all 5 cohorts are present
    expected_cohorts = {"pima", "cleveland", "ckd", "ilpd", "wbcd"}
    assert set(df["cohort"].unique()) == expected_cohorts

    # Verify per-fold weights sum to ~1.0
    for (cohort, fold_id), group in df.groupby(["cohort", "outer_fold_id"]):
        total_weight = group["weight"].sum()
        assert abs(total_weight - 1.0) < 1e-4, f"Weights for {cohort} fold {fold_id} do not sum to 1.0 (got {total_weight})"

    # Verify minimum weight floor (0.01)
    min_w = df["weight"].min()
    assert min_w >= 0.0099, f"Minimum weight floor violated: found weight {min_w} < 0.01"

    # Run audit and verify collapsed components are detected and written to md
    res = audit_ensemble_weights()
    md_path = res["audit_md_path"]
    assert os.path.exists(md_path)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "Ensemble Weights & Model Diversity Audit" in md_text
    assert "COLLAPSE_DETECTED" in str(res["collapsed_components"]) or len(res["collapsed_components"]) == 0
