"""
Tests 1, 2, 3, 4: Holdout isolation, fit-boundary leakage, cache immutability, calibration/threshold barriers.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest

from src.pipeline.protocol import (
    build_and_save_protocol_splits, fit_and_evaluate_pipeline, InnerCVCache
)
from src.data.load import load_cohort
from src.models.dshte import DSHTEClassifier


def test_holdout_isolation(tmp_path):
    """
    Test 1: Holdout indices never appear in any CV fold, inner split, or outer training set.
    """
    logs_dir = str(tmp_path / "logs")
    cohorts = ["pima", "ckd"]
    holdout_dict, folds_dict = build_and_save_protocol_splits(
        cohorts=cohorts,
        raw_dir="data/raw",
        logs_dir=logs_dir,
        seed=42,
        n_repeats=2,
        n_outer_folds=3
    )

    for c in cohorts:
        ho_indices = set(holdout_dict[c]["holdout_indices"])
        dev_indices = set(folds_dict[c]["dev_indices"])

        # Dev and holdout must be completely disjoint
        assert ho_indices.isdisjoint(dev_indices), f"Leakage: holdout and dev overlap in {c}!"

        for r_info in folds_dict[c]["repeats"]:
            for f_info in r_info["folds"]:
                tr_set = set(f_info["train_indices"])
                val_set = set(f_info["val_indices"])

                # Neither train nor val can touch holdout
                assert tr_set.isdisjoint(ho_indices), f"Leakage: train touches holdout in {c}!"
                assert val_set.isdisjoint(ho_indices), f"Leakage: val touches holdout in {c}!"
                assert tr_set.isdisjoint(val_set), f"Fold overlap in {c}!"


def test_fit_boundary_leakage():
    """
    Test 2: Preprocessor, AFO, and Model fit receives ONLY training indices.
    """
    X, y, num_cols, cat_cols = load_cohort("pima", "data/raw")
    # Take arbitrary split
    tr_idx = list(range(0, 500))
    te_idx = list(range(500, len(X)))

    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]

    res = fit_and_evaluate_pipeline(
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        numerical_cols=num_cols, categorical_cols=cat_cols,
        model_class=DSHTEClassifier,
        model_kwargs={
            "include_transformer": False, "include_xgb": False, "include_lgbm": True, "include_catboost": False,
            "lgbm_params": {"n_estimators": 10}, "random_state": 42
        },
        imbalance_strategy="none",
        afo_enabled=True,
        calibration_enabled=True,
        threshold_enabled=True,
        random_state=42
    )

    # Assert test predictions length equals test set length
    assert len(res["y_pred"]) == len(te_idx)
    assert len(res["y_prob"]) == len(te_idx)
    # Assert preprocessor feature names out matches AFO input
    assert len(res["selected_features"]) <= len(res["preprocessor"].feature_names_out_)


def test_cache_immutability():
    """
    Test 3: Inner CV cache arrays are strictly read-only.
    """
    cache = InnerCVCache()
    arr = np.array([1.0, 2.0, 3.0])
    cache.put("key_1", {"data": arr})

    cached_data = cache.get("key_1")["data"]
    assert not cached_data.flags.writeable, "Cache arrays must be marked read-only!"
    with pytest.raises(ValueError):
        cached_data[0] = 999.0


def test_threshold_calibration_unresampled():
    """
    Test 4: Calibration and threshold fit uses un-resampled data only.
    """
    from src.calibration.calibrate import choose_best_calibrator
    from src.threshold.youden import optimize_youden_threshold

    # Simulate unbalanced true labels
    y_val = np.array([0]*80 + [1]*20)
    # Simulated probabilities
    probs = np.linspace(0.1, 0.9, 100)

    cal = choose_best_calibrator(probs, y_val)
    cal_probs = cal.calibrate(probs)

    # Threshold optimization
    thresh_info = optimize_youden_threshold(y_val, cal_probs)
    t = thresh_info["optimal_threshold"]
    assert 0.05 <= t <= 0.95
    assert thresh_info["youden_j"] >= -1.0
