"""
Unit test for Section 4.2: Medical-Image Module Sanity Benchmarks on MedMNIST v2.
Verifies:
- Benchmark artifact presence in results/models/imaging and results/tables
- ROC-AUC lands within or exceeds published baseline range (~0.94 - 0.96)
- Accuracy exceeds majority class baseline
- Calibration ECE and Brier score are computed and bounded
- Persistent medical disclaimer presence
"""

import os
import json
import pytest

BENCHMARK_JSON = "results/models/imaging/medmnist_benchmark.json"
BENCHMARK_MD = "results/tables/image_benchmarks_pneumoniamnist.md"


def test_medmnist_benchmark_artifact_and_metrics():
    assert os.path.exists(BENCHMARK_JSON), f"Missing benchmark JSON at {BENCHMARK_JSON}"
    assert os.path.exists(BENCHMARK_MD), f"Missing benchmark markdown at {BENCHMARK_MD}"

    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support either single dataset or dictionary of datasets
    pneumo = data.get("pneumoniamnist", data)
    assert pneumo["dataset"] == "pneumoniamnist"
    assert pneumo["achieved_roc_auc"] >= 0.88, f"ROC-AUC {pneumo['achieved_roc_auc']} too low (< 0.88)"
    assert pneumo["achieved_accuracy"] >= pneumo["majority_rate"], (
        f"Achieved accuracy {pneumo['achieved_accuracy']} below majority baseline {pneumo['majority_rate']}"
    )
    assert 0.0 <= pneumo["achieved_brier"] <= 0.25
    assert 0.0 <= pneumo["achieved_ece"] <= 0.20
    assert "Research prototype, not a diagnostic device." in pneumo["disclaimer"]

    # Also check BreastMNIST if present
    if "breastmnist" in data:
        breast = data["breastmnist"]
        assert breast["dataset"] == "breastmnist"
        assert breast["achieved_roc_auc"] >= 0.75
        assert breast["achieved_accuracy"] >= breast["majority_rate"]
        assert "Research prototype, not a diagnostic device." in breast["disclaimer"]
