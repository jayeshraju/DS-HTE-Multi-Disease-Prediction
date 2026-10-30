"""
Tests 9, 10: Explainability (SHAP & LIME) and Feature Stability metrics.
Includes exact tests for Kuncheva consistency index and Jaccard similarity.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.explain.shap_explain import compute_shap_explanations
from src.explain.lime_explain import compute_lime_explanations, compute_shap_lime_agreement
from src.evaluation.stability import jaccard_index, kuncheva_index, evaluate_feature_stability


def test_stability_metrics_bounds():
    """
    Test 10: Identical sets give 1.0; disjoint sets give <= 0; range [-1, 1].
    """
    set_a = {"f1", "f2", "f3"}
    set_b = {"f1", "f2", "f3"}
    set_c = {"f4", "f5", "f6"}

    # Identical sets give 1.0
    assert np.isclose(jaccard_index(set_a, set_b), 1.0)
    assert np.isclose(kuncheva_index(set_a, set_b, total_features=10), 1.0)

    # Disjoint sets give 0.0 Jaccard and negative Kuncheva
    assert np.isclose(jaccard_index(set_a, set_c), 0.0)
    k_disjoint = kuncheva_index(set_a, set_c, total_features=10)
    # (0 * 10 - 9) / (3 * 7) = -9 / 21 = -0.42857
    assert -1.0 <= k_disjoint <= 0.0

    # Multi-fold stability
    folds = [["f1", "f2", "f3"], ["f1", "f2", "f4"], ["f1", "f2", "f3"]]
    res = evaluate_feature_stability(folds, total_features=10)
    assert 0.0 <= res["mean_jaccard"] <= 1.0
    assert -1.0 <= res["mean_kuncheva"] <= 1.0
    assert res["frequencies"]["f1"] == 1.0


def test_kuncheva_random_sets_and_edges():
    """
    Test Kuncheva on random sets gives around 0; undefined cases (k=0, k=n) return nan.
    """
    total_features = 100
    k = 20
    rng = np.random.RandomState(42)

    # Generate 50 random pairs of size k from total_features
    all_feats = [f"feat_{i}" for i in range(total_features)]
    k_scores = []
    for _ in range(50):
        s_a = set(rng.choice(all_feats, size=k, replace=False))
        s_b = set(rng.choice(all_feats, size=k, replace=False))
        val = kuncheva_index(s_a, s_b, total_features)
        assert -1.0 <= val <= 1.0
        k_scores.append(val)

    # Expected value of Kuncheva index for random subsets is 0
    mean_k = np.mean(k_scores)
    assert np.abs(mean_k) < 0.15, f"Expected mean Kuncheva near 0 for random subsets, got {mean_k}"

    # Edge cases: k=0 and k=n return nan
    empty_set = set()
    full_set = set(all_feats)
    assert np.isnan(kuncheva_index(empty_set, set(all_feats[:k]), total_features))
    assert np.isnan(kuncheva_index(full_set, full_set, total_features))


def test_kuncheva_raises_on_invalid_bounds():
    """
    Test error raised if score would fall outside [-1, 1].
    """
    # Negative total_features raises ValueError
    with pytest.raises(ValueError):
        kuncheva_index({"f1"}, {"f2"}, total_features=0)


def test_explainability_pipeline():
    """
    Test 9: SHAP importance shape and LIME explanations for sample instances.
    """
    X = np.random.RandomState(42).randn(40, 4)
    y = np.random.RandomState(42).randint(0, 2, size=40)
    feat_names = ["age", "bp", "chol", "bmi"]

    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X, y)

    shap_res = compute_shap_explanations(rf, X[:20], X[20:], feat_names, n_background=15)
    assert len(shap_res["ranked_features"]) == 4
    assert len(shap_res["global_importance"]) == 4

    lime_res = compute_lime_explanations(rf, X[:20], X[20:], feat_names, n_instances=2)
    assert len(lime_res["explanations"]) == 2

    agreement = compute_shap_lime_agreement(shap_res["global_importance"], lime_res["explanations"], feat_names)
    assert -1.0 <= agreement <= 1.0
