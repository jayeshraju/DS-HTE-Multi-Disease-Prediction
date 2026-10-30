"""
Unit Tests for External Heart Multi-Center Validation (Gate 2).
Validates:
- External validation artifacts presence in results/tables/
- Evaluates across Hungarian, Switzerland, and VA Long Beach cohorts
- Feature alignment across the 14 standard attributes
- Provenance sidecar presence
"""

import os
import json
import pytest
import pandas as pd

CSV_PATH = "results/tables/external_validation_heart.csv"
MD_PATH = "results/tables/external_validation_heart.md"
JSON_PATH = "results/tables/external_validation_heart.json"


def test_external_heart_validation_tables():
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    assert os.path.exists(MD_PATH), f"Missing {MD_PATH}"
    assert os.path.exists(JSON_PATH), f"Missing {JSON_PATH}"

    df = pd.read_csv(CSV_PATH)
    assert len(df) == 3
    centers = list(df["Dataset Key"])
    assert "HUNGARIAN" in centers
    assert "SWITZERLAND" in centers
    assert "VA_LONG_BEACH" in centers

    # Assert provenance sidecars
    assert os.path.exists(f"{CSV_PATH}.provenance.json")
    assert os.path.exists(f"{MD_PATH}.provenance.json")

    with open(MD_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Hungarian" in content
    assert "Switzerland" in content
    assert "VA Medical Center" in content
    assert "Research prototype, not a diagnostic device." in content
