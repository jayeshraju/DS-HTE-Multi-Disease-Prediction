"""
Youden's J decision threshold optimization.
Follows Section III-F:
- J = Sensitivity + Specificity - 1
- Evaluated across validation predictions
- Aggregated across inner folds via median
- Applied unchanged to test fold and confirmatory holdout
"""

from typing import Tuple, Dict, Any, List
import numpy as np
from sklearn.metrics import roc_curve


def optimize_youden_threshold(y_true: np.ndarray, y_probs: np.ndarray,
                              threshold_grid: int = 100) -> Dict[str, Any]:
    """
    Computes optimal threshold maximizing Youden's J statistic:
    J(t) = Sensitivity(t) + Specificity(t) - 1
    """
    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]

    # Grid of candidate thresholds between 0.05 and 0.95
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    # Specificity = 1 - FPR, Sensitivity = TPR
    j_scores = tpr - fpr

    best_idx = int(np.argmax(j_scores))
    best_threshold = float(thresholds[best_idx])

    # Bound within clinical range [0.05, 0.95]
    best_threshold = float(np.clip(best_threshold, 0.05, 0.95))
    
    # Calculate sensitivity and specificity at best threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))
    j_stat = float(sens + spec - 1.0)

    return {
        "optimal_threshold": best_threshold,
        "sensitivity": sens,
        "specificity": spec,
        "youden_j": j_stat
    }


def aggregate_fold_thresholds(threshold_list: List[float]) -> float:
    """
    Aggregates optimal thresholds across inner folds using median.
    """
    if not threshold_list:
        return 0.5
    return float(np.median(threshold_list))


def optimize_accuracy_threshold(y_true: np.ndarray, y_probs: np.ndarray,
                                threshold_grid: int = 100) -> Dict[str, Any]:
    """
    Computes optimal threshold maximizing raw accuracy on validation predictions.
    Pre-declared operating point: Accuracy-optimal (for benchmark comparison against majority baseline).
    """
    if y_probs.ndim == 2:
        y_probs = y_probs[:, 1]

    candidates = np.linspace(0.05, 0.95, threshold_grid)
    best_acc = -1.0
    best_threshold = 0.5

    for t in candidates:
        preds = (y_probs >= t).astype(int)
        acc = float(np.mean(preds == y_true))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(t)

    # Compute sensitivity and specificity at accuracy-optimal threshold
    y_pred = (y_probs >= best_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))

    sens = float(tp / max(1, tp + fn))
    spec = float(tn / max(1, tn + fp))

    return {
        "optimal_threshold": best_threshold,
        "accuracy": best_acc,
        "sensitivity": sens,
        "specificity": spec
    }

