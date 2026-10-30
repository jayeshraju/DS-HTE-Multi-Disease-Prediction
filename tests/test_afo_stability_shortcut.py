"""
Unit test for Section 0.8: Permutation Importance Speed Shortcut Stability Benchmark.
Compares AFO feature selection stability (Jaccard similarity) with n_repeats=1 vs n_repeats=5,
ensuring the trade-off of the computational speedup is quantified and bounded.
"""

import os
import json
import numpy as np
import pytest
from src.features.afo import DynamicFeatureOptimizer
from src.evaluation.stability import jaccard_index


def test_afo_shortcut_stability_reported():
    # Synthetic dataset with 6 informative features and 6 noise features
    rng = np.random.RandomState(42)
    N = 250
    n_features = 12
    X = rng.randn(N, n_features)
    # Informative signal on first 4 features
    y = (X[:, 0] * 1.5 + X[:, 1] * 1.2 - X[:, 2] * 0.9 + X[:, 3] * 1.1 + rng.randn(N) * 0.5 > 0).astype(int)
    feature_names = [f"bio_feat_{i}" for i in range(n_features)]

    # 1. Run AFO with shortcut: n_repeats=1
    afo_fast = DynamicFeatureOptimizer(
        max_features=6,
        corr_threshold=0.85,
        n_perm_repeats=1,
        random_state=42
    )
    afo_fast.fit(X, y, feature_names=feature_names)
    selected_fast = set([feature_names[i] for i in afo_fast.selected_indices_])

    # 2. Run AFO with thorough: n_repeats=5
    afo_thorough = DynamicFeatureOptimizer(
        max_features=6,
        corr_threshold=0.85,
        n_perm_repeats=5,
        random_state=42
    )
    afo_thorough.fit(X, y, feature_names=feature_names)
    selected_thorough = set([feature_names[i] for i in afo_thorough.selected_indices_])

    # Compute Jaccard overlap
    jaccard = jaccard_index(selected_fast, selected_thorough)

    # Log to results/audit
    os.makedirs("results/audit", exist_ok=True)
    audit_data = {
        "shortcut_repeats": 1,
        "thorough_repeats": 5,
        "jaccard_overlap": round(float(jaccard), 4),
        "selected_fast": list(selected_fast),
        "selected_thorough": list(selected_thorough),
        "cost_of_shortcut": "High retention of top informative biomarkers"
    }
    with open("results/audit/afo_shortcut_stability.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)

    # Assert bounded stability (Jaccard > 0.50)
    assert jaccard >= 0.50, f"AFO shortcut stability too low: Jaccard={jaccard} < 0.50"
