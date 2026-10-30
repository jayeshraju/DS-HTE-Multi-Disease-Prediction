"""
Tests 5, 6: Performance metrics verification and statistical tests suite.
"""

import numpy as np
import pytest
from sklearn.metrics import (
    matthews_corrcoef, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss
)
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests

from src.evaluation.metrics import (
    compute_all_metrics, compute_expected_calibration_error, compute_fold_scores_ci
)
from src.evaluation.stats_tests import (
    paired_wilcoxon_test, mcnemar_test, delong_paired_test,
    friedman_nemenyi_test, apply_p_adjustments, fast_delong
)


def test_metrics_against_sklearn():
    """
    Test 5: Validates all metrics against standard sklearn definitions.
    """
    rng = np.random.RandomState(42)
    y_true = rng.randint(0, 2, size=100)
    y_prob = rng.uniform(0.0, 1.0, size=100)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_all_metrics(y_true, y_pred, y_prob)

    assert np.isclose(metrics["f1"], f1_score(y_true, y_pred))
    assert np.isclose(metrics["mcc"], matthews_corrcoef(y_true, y_pred))
    assert np.isclose(metrics["balanced_accuracy"], balanced_accuracy_score(y_true, y_pred))
    assert np.isclose(metrics["roc_auc"], roc_auc_score(y_true, y_prob))
    assert np.isclose(metrics["pr_auc"], average_precision_score(y_true, y_prob))
    assert np.isclose(metrics["brier"], brier_score_loss(y_true, y_prob))
    assert 0.0 <= metrics["ece"] <= 1.0


def test_fold_scores_ci_invariants():
    """
    Test Section A4: Asserts that for any fold-level scores, reported mean lies within CI [lower, upper].
    """
    rng = np.random.RandomState(101)
    
    # Skewed / low-variance / zero-heavy scenarios (e.g. imbalanced ILPD)
    test_distributions = [
        [0.78] * 50,  # Constant
        rng.beta(2, 5, size=50).tolist(),  # Skewed
        rng.normal(0.65, 0.1, size=50).tolist(),  # Normal
        [0.0] * 10 + [0.8] * 40,  # Bimodal
        rng.uniform(0.2, 0.9, size=50).tolist()
    ]

    for dist in test_distributions:
        res = compute_fold_scores_ci(dist, n_bootstraps=500, ci=0.95, seed=42)
        mean_val = res["mean"]
        lower = res["lower"]
        upper = res["upper"]
        assert lower <= mean_val <= upper, f"Violation: {lower} <= {mean_val} <= {upper}"
        assert res["naive_lower"] <= mean_val <= res["naive_upper"]
        assert res["nb_lower"] <= mean_val <= res["nb_upper"]


def test_nadeau_bengio_ci_wider_than_naive():
    """
    Test Section 0.6: 50 outer folds share training data.
    Nadeau-Bengio corrected CI accounts for fold overlap, producing a wider CI than naive independence.
    """
    from src.evaluation.metrics import compute_nadeau_bengio_ci
    rng = np.random.RandomState(42)
    scores = rng.normal(0.75, 0.04, size=50).tolist()

    res = compute_fold_scores_ci(scores, n_bootstraps=1000, ci=0.95, seed=42)
    naive_width = res["naive_upper"] - res["naive_lower"]
    nb_width = res["nb_upper"] - res["nb_lower"]

    # Nadeau-Bengio width must exceed naive bootstrap width due to (1/J + 0.25) factor
    assert nb_width > naive_width, f"Expected NB width ({nb_width}) > naive ({naive_width})"
    assert res["nb_lower"] < res["mean"] < res["nb_upper"]



def test_ece_perfect_and_worst():
    # Perfect calibration: predicted prob matches true outcome
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)
    assert np.isclose(ece, 0.0)


def test_wilcoxon_all_zero():
    """
    Test 6a: Wilcoxon test explicitly handles all-zero differences.
    """
    scores_a = np.array([0.85] * 20)
    scores_b = np.array([0.85] * 20)
    res = paired_wilcoxon_test(scores_a, scores_b)
    assert res["p_value"] == 1.0
    assert res["n_non_zero"] == 0


def test_mcnemar_against_statsmodels():
    """
    Test 6b: McNemar test matches statsmodels implementation for both asymptotic and exact cases.
    """
    # Case 1: b + c >= 25 -> asymptotic chi2 with continuity correction
    y_true = np.array([0]*50 + [1]*50)
    preds_a = np.array([0]*35 + [1]*15 + [0]*20 + [1]*30)
    preds_b = np.array([0]*20 + [1]*30 + [0]*35 + [1]*15)

    res_our = mcnemar_test(y_true, preds_a, preds_b)
    assert res_our["b"] + res_our["c"] >= 25

    corr_a = (preds_a == y_true)
    corr_b = (preds_b == y_true)
    table = [
        [np.sum(corr_a & corr_b), np.sum(corr_a & ~corr_b)],
        [np.sum(~corr_a & corr_b), np.sum(~corr_a & ~corr_b)]
    ]
    sm_res = sm_mcnemar(table, exact=False, correction=True)
    assert np.isclose(res_our["statistic"], sm_res.statistic)
    assert np.isclose(res_our["p_value"], sm_res.pvalue)

    # Case 2: b + c < 25 -> exact binomial test
    preds_small_a = np.array([0]*45 + [1]*5 + [0]*5 + [1]*45)
    preds_small_b = np.array([0]*40 + [1]*10 + [0]*8 + [1]*42)
    res_exact = mcnemar_test(y_true, preds_small_a, preds_small_b)
    assert res_exact["b"] + res_exact["c"] < 25
    corr_sa = (preds_small_a == y_true)
    corr_sb = (preds_small_b == y_true)
    table_exact = [
        [np.sum(corr_sa & corr_sb), np.sum(corr_sa & ~corr_sb)],
        [np.sum(~corr_sa & corr_sb), np.sum(~corr_sa & ~corr_sb)]
    ]
    sm_exact_res = sm_mcnemar(table_exact, exact=True)
    assert np.isclose(res_exact["p_value"], sm_exact_res.pvalue)


def test_delong_test():
    """
    Test 6c: DeLong algorithm computes correct AUCs and p-value.
    """
    y_true = np.array([0]*50 + [1]*50)
    p_a = np.linspace(0.1, 0.9, 100)
    p_b = np.linspace(0.2, 0.8, 100)

    res = delong_paired_test(y_true, p_a, p_b)
    assert np.isclose(res["auc_a"], roc_auc_score(y_true, p_a))
    assert np.isclose(res["auc_b"], roc_auc_score(y_true, p_b))
    assert 0.0 <= res["p_value"] <= 1.0


def test_multiple_testing_adjustment():
    """
    Test 6d: Multiple-comparison corrections match statsmodels.
    """
    raw_p = [0.001, 0.012, 0.045, 0.200, 0.850]
    adj = apply_p_adjustments(raw_p)

    _, sm_holm, _, _ = multipletests(raw_p, method="holm")
    _, sm_bh, _, _ = multipletests(raw_p, method="fdr_bh")

    assert np.allclose(adj["holm"], sm_holm)
    assert np.allclose(adj["bh"], sm_bh)
