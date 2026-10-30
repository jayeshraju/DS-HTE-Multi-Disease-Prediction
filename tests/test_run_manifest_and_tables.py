"""
Run Manifest, 50-Fold Protocol, and Run-ID Traceability Test Suite.
Follows Sections A1 & A2:
- Write results/run_manifest.json: run_id, config hash, seeds, profile, library versions, start/end time.
- Fails if any file in results/tables/ lacks the current run_id in its header/footer or JSON sidecar.
- Asserts n_outer_folds == 50 for every model and dataset in paper profile.
"""

import os
import glob
import json
import pytest
import pandas as pd


MANIFEST_PATH = "results/run_manifest.json"
TABLES_DIR = "results/tables"
FOLD_METRICS_PATH = "results/fold_metrics/outer_fold_metrics.parquet"


@pytest.fixture(scope="module")
def manifest():
    if not os.path.exists(MANIFEST_PATH):
        pytest.skip("run_manifest.json not yet generated.")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_manifest_structure(manifest):
    required_keys = ["run_id", "config_hash", "profile", "seeds", "n_repeats", "n_outer_folds", "cohorts", "library_versions", "start_time"]
    for k in required_keys:
        assert k in manifest, f"Key '{k}' missing from run_manifest.json"

    assert len(manifest["run_id"]) > 0
    assert len(manifest["config_hash"]) > 0
    assert manifest["n_outer_folds"] * manifest["n_repeats"] == 50


def test_all_tables_have_current_run_id(manifest):
    run_id = manifest["run_id"]
    md_files = glob.glob(os.path.join(TABLES_DIR, "*.md"))
    csv_files = glob.glob(os.path.join(TABLES_DIR, "*.csv"))

    assert len(md_files) > 0, "No markdown tables found in results/tables/"

    # Check Markdown files have run_id in content
    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()
            assert run_id in content, f"Table file {os.path.basename(md_file)} lacks current run_id '{run_id}'!"

    # Check CSV files have companion JSON sidecar with current run_id
    for csv_file in csv_files:
        base_name = os.path.splitext(csv_file)[0]
        json_sidecar = f"{base_name}.json"
        assert os.path.exists(json_sidecar), f"CSV table {os.path.basename(csv_file)} lacks sidecar {os.path.basename(json_sidecar)}"
        with open(json_sidecar, "r", encoding="utf-8") as f:
            meta = json.load(f)
            assert meta.get("run_id") == run_id, f"Sidecar for {csv_file} has run_id '{meta.get('run_id')}' != '{run_id}'"


def test_50_outer_folds_count(manifest):
    if not os.path.exists(FOLD_METRICS_PATH):
        pytest.skip("outer_fold_metrics.parquet not yet generated.")

    df_folds = pd.read_parquet(FOLD_METRICS_PATH)
    models = df_folds["model"].unique()
    cohorts = df_folds["cohort"].unique()

    for cohort in cohorts:
        for model in models:
            sub = df_folds[(df_folds["cohort"] == cohort) & (df_folds["model"] == model)]
            assert len(sub) == 50, f"Expected 50 outer folds for {cohort} - {model}, found {len(sub)}"
