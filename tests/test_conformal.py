"""
Unit Tests for Split-Conformal Prediction and Coverage Invariants (Gate 2).
Validates:
- Finite-sample inductive split-conformal calibration on exchangeable data.
- Prediction set generation and multi-class inclusion for ambiguous cases.
- Empirical coverage is within +- 2.5 percentage points of nominal (e.g. 90% -> [87.5%, 92.5%]).
- Abstain / Expert Review flag triggered for ambiguous predictions (set size > 1).
"""

import numpy as np
import pytest
from src.evaluation.conformal import SplitConformalClassifier


def test_split_conformal_coverage_within_bounds():
    rng = np.random.RandomState(42)
    n_cal = 2000
    n_test = 5000

    # Exchangeable logistic data generation
    x_cal = rng.randn(n_cal)
    p_cal = 1.0 / (1.0 + np.exp(-x_cal))
    y_cal = rng.binomial(1, p_cal)

    x_test = rng.randn(n_test)
    p_test = 1.0 / (1.0 + np.exp(-x_test))
    y_test = rng.binomial(1, p_test)

    # Nominal 90% coverage (alpha = 0.10)
    conformal = SplitConformalClassifier(alpha=0.10)
    q_hat = conformal.calibrate(p_cal, y_cal)

    empirical_cov = conformal.compute_empirical_coverage(p_test, y_test)
    nominal_cov = 0.90

    # Assert within +- 2.5 percentage points ([0.875, 0.925])
    assert abs(empirical_cov - nominal_cov) <= 0.025, (
        f"Conformal empirical coverage {empirical_cov*100:.2f}% deviates from nominal {nominal_cov*100:.1f}% by > 2.5 pts"
    )


def test_conformal_abstain_flag_on_uncertain_prediction():
    rng = np.random.RandomState(42)
    n_cal = 500
    x_cal = rng.randn(n_cal)
    p_cal = 1.0 / (1.0 + np.exp(-x_cal))
    y_cal = rng.binomial(1, p_cal)

    conformal = SplitConformalClassifier(alpha=0.10)
    conformal.calibrate(p_cal, y_cal)

    # Highly ambiguous prediction near 0.50
    ambiguous_probs = np.array([0.50])
    res = conformal.predict_set(ambiguous_probs)
    assert len(res) == 1
    # Ambiguous sample should include both classes and flag review
    assert res[0]["needs_expert_review"] is True or res[0]["set_size"] > 1
