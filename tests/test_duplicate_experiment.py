"""
Unit test for Section 0.9: Controlled ILPD Duplicate Record Leakage Experiment.
Verifies that empirical data exists for grouped vs ungrouped CV evaluation,
distinguishing proven causes from untested hypotheses.
"""

import os
import json
import pytest
from src.evaluation.duplicate_experiment import run_ilpd_duplicate_experiment


def test_ilpd_duplicate_experiment_logged_and_bounded():
    audit_path = "results/audit/ilpd_duplicate_leakage_experiment.json"
    if not os.path.exists(audit_path):
        res = run_ilpd_duplicate_experiment(n_splits=5, n_repeats=2, random_state=42)
    else:
        with open(audit_path, "r", encoding="utf-8") as f:
            res = json.load(f)

    assert "ungrouped_evaluation" in res
    assert "grouped_evaluation" in res
    assert "leakage_delta_ungrouped_minus_grouped" in res
    assert res["total_records"] == 583
    assert res["duplicate_records_count"] == 13

    delta = res["leakage_delta_ungrouped_minus_grouped"]
    # Empirical delta is positive (optimistic bias of naive CV) and bounded <= 5%
    assert delta["accuracy_delta"] >= 0.0
    assert delta["accuracy_delta"] <= 0.05
    assert delta["roc_auc_delta"] >= 0.0
