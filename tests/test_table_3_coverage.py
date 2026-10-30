"""
Unit test for Section 0.2: Table III Full Model and Cohort Coverage Assertion.
Ensures Table III and outer fold metrics encompass every model x every cohort x 50 folds:
n_cohorts (5) * n_models (11) * 50 = 2,750 fold runs.
"""

import os
import pytest
import pandas as pd

FOLD_METRICS_PATH = "results/fold_metrics/outer_fold_metrics.parquet"
TABLE_3_PATH = "results/tables/table_3_baselines_comparison.csv"


def test_table_3_complete_coverage():
    assert os.path.exists(FOLD_METRICS_PATH), f"Missing fold metrics at {FOLD_METRICS_PATH}"
    df_folds = pd.read_parquet(FOLD_METRICS_PATH)

    cohorts = df_folds["cohort"].unique()
    models = df_folds["model"].unique()

    n_cohorts = len(cohorts)
    n_models = len(models)

    assert n_cohorts == 5, f"Expected 5 cohorts, found {n_cohorts}: {cohorts}"
    assert n_models == 11, f"Expected 11 models, found {n_models}: {models}"

    expected_total_folds = n_cohorts * n_models * 50
    assert len(df_folds) == expected_total_folds, (
        f"Expected {expected_total_folds} outer fold runs, but got {len(df_folds)}"
    )

    # Verify every cohort x model has exactly 50 folds
    for c in cohorts:
        for m in models:
            n_entries = len(df_folds[(df_folds["cohort"] == c) & (df_folds["model"] == m)])
            assert n_entries == 50, f"Expected 50 folds for {c} x {m}, got {n_entries}"

    # Verify Table 3 summary has all 5 cohorts x 11 models
    if os.path.exists(TABLE_3_PATH):
        df_tab3 = pd.read_csv(TABLE_3_PATH)
        assert len(df_tab3) == n_cohorts * n_models, (
            f"Table 3 summary rows {len(df_tab3)} != {n_cohorts * n_models}"
        )
