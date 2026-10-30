"""
Split-Conformal Prediction and Uncertainty Quantification.
Follows Gate 2:
- Inductive split-conformal prediction sets with finite-sample coverage guarantees.
- Calibration non-conformity score: s_i = 1 - P(Y = y_i | X_i).
- Conformal quantile cutoff at level (1 - alpha):
    q_hat = np.quantile(scores, ceil((n + 1) * (1 - alpha)) / n)
- Prediction sets: C(x) = { c : 1 - P(Y = c | x) <= q_hat }
- Flags "Abstain / Expert Review" when set size != 1 (uncertain or ambiguous).
- Unit tested to assert empirical coverage lands within +- 2 percentage points of nominal (e.g. 90% -> [88%, 92%]).
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class SplitConformalClassifier:
    def __init__(self, alpha: float = 0.10):
        """
        alpha: significance level (e.g. alpha=0.10 gives nominal 90% coverage).
        """
        self.alpha = alpha
        self.q_hat = 1.0

    def calibrate(self, val_probs: np.ndarray, val_y_true: np.ndarray):
        """
        Computes non-conformity quantile cutoff on calibration fold.
        """
        n = len(val_y_true)
        if val_probs.ndim == 1:
            val_probs = np.column_stack([1.0 - val_probs, val_probs])

        # Non-conformity scores on true labels
        scores = []
        for i in range(n):
            true_cls = int(val_y_true[i])
            prob_true = val_probs[i, true_cls]
            scores.append(1.0 - prob_true)

        scores = np.array(scores)
        # Finite-sample adjusted quantile
        quantile_level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, quantile_level, method="higher"))
        return self.q_hat

    def predict_set(self, test_probs: np.ndarray) -> List[Dict[str, Any]]:
        """
        Returns conformal prediction set and review flags for each test sample.
        """
        if test_probs.ndim == 1:
            test_probs = np.column_stack([1.0 - test_probs, test_probs])

        results = []
        for i in range(len(test_probs)):
            p0 = test_probs[i, 0]
            p1 = test_probs[i, 1]

            pred_set = []
            if (1.0 - p0) <= self.q_hat:
                pred_set.append(0)
            if (1.0 - p1) <= self.q_hat:
                pred_set.append(1)

            # If empty set, include top class to ensure non-empty
            if len(pred_set) == 0:
                pred_set.append(int(np.argmax([p0, p1])))

            needs_review = len(pred_set) > 1 or len(pred_set) == 0

            results.append({
                "prediction_set": pred_set,
                "set_size": len(pred_set),
                "needs_expert_review": needs_review,
                "confidence": 1.0 - self.alpha,
                "q_hat": round(self.q_hat, 4)
            })

        return results

    def compute_empirical_coverage(self,
                                   test_probs: np.ndarray,
                                   test_y_true: np.ndarray) -> float:
        """Computes empirical marginal coverage fraction on test set."""
        preds = self.predict_set(test_probs)
        covered = 0
        for i, p in enumerate(preds):
            if test_y_true[i] in p["prediction_set"]:
                covered += 1
        return covered / len(test_y_true)
