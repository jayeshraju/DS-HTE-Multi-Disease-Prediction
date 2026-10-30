"""
Evaluation metrics module.
Implements the 11 clinical performance metrics specified in Section IV-C:
- Accuracy
- Precision
- Sensitivity (Recall)
- Specificity
- F1-score
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)
- ROC-AUC
- PR-AUC (Average Precision)
- Brier Score
- Expected Calibration Error (ECE, 10 equal-width bins)
Includes bootstrap 95% confidence intervals on pooled predictions.
"""

from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix
)


def compute_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE) with equal-width bins on [0, 1].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes the 11 required metrics for binary classification.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / max(1, tn + fp))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))

    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "balanced_accuracy": bal_acc,
        "mcc": mcc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier": brier,
        "ece": ece
    }


def compute_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                         n_bootstraps: int = 1000, ci: float = 0.95,
                         seed: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Computes 95% bootstrap confidence intervals for all 11 metrics on pooled predictions.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    alpha = (1.0 - ci) / 2.0

    boot_metrics = {k: [] for k in [
        "accuracy", "precision", "sensitivity", "specificity", "f1",
        "balanced_accuracy", "mcc", "roc_auc", "pr_auc", "brier", "ece"
    ]}

    for _ in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        # Ensure both classes present in bootstrap sample
        if len(np.unique(y_true[boot_idx])) < 2:
            continue
        m = compute_all_metrics(y_true[boot_idx], y_pred[boot_idx], y_prob[boot_idx])
        for k, v in m.items():
            boot_metrics[k].append(v)

    ci_dict = {}
    for k, v in boot_metrics.items():
        if v:
            lower = float(np.percentile(v, alpha * 100))
            upper = float(np.percentile(v, (1.0 - alpha) * 100))
        else:
            lower, upper = 0.0, 0.0
        ci_dict[k] = {"lower": lower, "upper": upper}

    return ci_dict


from scipy.stats import t as student_t


def compute_nadeau_bengio_ci(scores: List[float], n_train_test_ratio: float = 0.25, ci: float = 0.95) -> Dict[str, float]:
    """
    Computes Nadeau-Bengio (2003) corrected confidence interval for repeated CV fold means.
    Accounts for dependence across folds due to shared training samples.
    var_corrected = (1/J + n_test / n_train) * s^2, where for 5-fold CV n_test/n_train = 0.25.
    """
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0, "se_nb": 0.0, "ci_type": "nadeau_bengio"}
    mean_val = float(np.mean(arr))
    J = len(arr)
    if J <= 1:
        return {"mean": mean_val, "std": 0.0, "lower": mean_val, "upper": mean_val, "se_nb": 0.0, "ci_type": "nadeau_bengio"}

    s = float(np.std(arr, ddof=1))
    var_nb = (1.0 / J + n_train_test_ratio) * (s ** 2)
    se_nb = float(np.sqrt(max(0.0, var_nb)))
    df = J - 1
    alpha = (1.0 - ci) / 2.0
    t_crit = float(student_t.ppf(1.0 - alpha, df=df))
    margin = t_crit * se_nb

    lower = mean_val - margin
    upper = mean_val + margin
    return {
        "mean": mean_val,
        "std": s,
        "lower": lower,
        "upper": upper,
        "se_nb": se_nb,
        "ci_type": "nadeau_bengio"
    }


def compute_fold_scores_ci(scores: List[float], n_bootstraps: int = 1000, ci: float = 0.95, seed: int = 42,
                           n_train_test_ratio: float = 0.25) -> Dict[str, Any]:
    """
    Computes confidence intervals for fold scores.
    Returns:
    - 'naive_lower', 'naive_upper': naive bootstrap CI (assuming independence)
    - 'nb_lower', 'nb_upper': Nadeau-Bengio corrected CI (adjusting for fold overlap)
    - 'lower', 'upper': default naive bootstrap for backwards compatibility
    """
    arr = np.asarray(scores, dtype=float)
    if len(arr) == 0:
        return {
            "mean": 0.0, "std": 0.0, "lower": 0.0, "upper": 0.0,
            "naive_lower": 0.0, "naive_upper": 0.0,
            "nb_lower": 0.0, "nb_upper": 0.0
        }
    mean_val = float(np.mean(arr))
    J = len(arr)
    std_val = float(np.std(arr, ddof=1)) if J > 1 else 0.0

    if J == 1:
        return {
            "mean": mean_val, "std": std_val, "lower": mean_val, "upper": mean_val,
            "naive_lower": mean_val, "naive_upper": mean_val,
            "nb_lower": mean_val, "nb_upper": mean_val
        }

    # 1. Naive bootstrap CI
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstraps):
        sample = rng.choice(arr, size=J, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = (1.0 - ci) / 2.0
    naive_lower = float(np.percentile(boot_means, alpha * 100))
    naive_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    # Guard invariant: lower <= mean <= upper
    naive_lower = min(naive_lower, mean_val)
    naive_upper = max(naive_upper, mean_val)

    # 2. Nadeau-Bengio corrected CI
    nb_ci = compute_nadeau_bengio_ci(scores, n_train_test_ratio=n_train_test_ratio, ci=ci)

    return {
        "mean": mean_val,
        "std": std_val,
        "lower": naive_lower,
        "upper": naive_upper,
        "naive_lower": naive_lower,
        "naive_upper": naive_upper,
        "nb_lower": nb_ci["lower"],
        "nb_upper": nb_ci["upper"],
        "se_nb": nb_ci["se_nb"]
    }

