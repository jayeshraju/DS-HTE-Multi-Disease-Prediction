"""
Comprehensive Diagnostic Evaluation & Statistical Significance Module.
Computes 10 classification metrics (Sensitivity, Specificity, F1, MCC, PR-AUC, Brier)
and performs non-parametric McNemar hypothesis testing.
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, matthews_corrcoef,
    cohen_kappa_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix
)
from scipy.stats import chisquare


def evaluate_clinical_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """
    Computes 10 publication-standard diagnostic metrics for binary medical predictions.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate Precision-Recall AUC
    prec_array, rec_array, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(rec_array, prec_array)
    
    roc_auc = roc_auc_score(y_true, y_proba)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_proba)

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Sensitivity (Recall)": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc),
        "PR-AUC": float(pr_auc),
        "MCC": float(mcc),
        "Cohen's Kappa": float(kappa),
        "Balanced Accuracy": float(bal_acc),
        "Brier Score": float(brier)
    }


def mcnemar_statistical_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> Tuple[float, float]:
    """
    Performs McNemar's chi-squared test on paired binary model predictions.
    Returns (chi2_statistic, p_value).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # n01: A wrong, B right; n10: A right, B wrong
    n01 = np.sum(~correct_a & correct_b)
    n10 = np.sum(correct_a & ~correct_b)

    if (n01 + n10) == 0:
        return 0.0, 1.0

    statistic = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    # p-value from chi-square distribution with df=1
    from scipy.stats import chi2
    p_value = 1.0 - chi2.cdf(statistic, df=1)

    return float(statistic), float(p_value)
