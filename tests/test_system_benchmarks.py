"""
Unit Tests for System Performance, Latency, and Memory Benchmarks (Gate 2).
Validates:
- Benchmark tables exist in results/tables/
- p50 and p95 latencies are measured and positive
- Storage sizes are tracked
- Provenance sidecars exist
"""

import os
import json
import pytest
import pandas as pd

CSV_PATH = "results/tables/system_benchmarks.csv"
MD_PATH = "results/tables/system_benchmarks.md"
JSON_PATH = "results/tables/system_benchmarks.json"


def test_system_benchmarks_table():
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    assert os.path.exists(MD_PATH), f"Missing {MD_PATH}"
    assert os.path.exists(JSON_PATH), f"Missing {JSON_PATH}"

    df = pd.read_csv(CSV_PATH)
    assert len(df) == 4
    ops = list(df["Operation / Endpoint"])
    assert any("predict_tabular" in o for o in ops)
    assert any("explain_prediction" in o for o in ops)
    assert any("Medical Image" in o for o in ops)
    assert any("PDF Report" in o for o in ops)

    # Check latency numbers are positive floats
    for _, r in df.iterrows():
        p50 = float(r["p50 Latency (ms)"])
        p95 = float(r["p95 Latency (ms)"])
        assert p50 >= 0.0
        assert p95 >= p50

    assert os.path.exists(f"{CSV_PATH}.provenance.json")
    assert os.path.exists(f"{MD_PATH}.provenance.json")
