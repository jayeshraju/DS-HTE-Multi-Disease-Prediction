"""
Statistical significance tests module.
Follows Section IV-B:
- Wilcoxon signed-rank test (handles zero differences)
- Friedman test + Nemenyi post-hoc across 7 models
- DeLong test for paired ROC-AUC
- McNemar test for paired holdout predictions (exact binomial when b+c < 25)
- Nadeau-Bengio corrected resampled t-test for repeated CV
- Holm and Benjamini-Hochberg multiple testing correction
- Degenerate p-value logger
"""

import os
from typing import Dict, Any, List, Tuple
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp


def log_degenerate_p_value(test_name: str, p_value: float, details: Dict[str, Any],
                           audit_path: str = "results/audit/degenerate_p_values.md"):
    os.makedirs(os.path.dirname(audit_path), exist_ok=True)
    with open(audit_path, "a", encoding="utf-8") as f:
        f.write(f"### Degenerate p-value Alert: {test_name}\n")
        f.write(f"- **p-value**: {p_value}\n")
        for k, v in details.items():
            f.write(f"- **{k}**: {v}\n")
        f.write("\n")


def paired_wilcoxon_test(scores_a: np.ndarray, scores_b: np.ndarray,
                         alternative: str = "two-sided") -> Dict[str, Any]:
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    non_zero = diffs[diffs != 0]

    if len(non_zero) == 0:
        log_degenerate_p_value("Wilcoxon Signed-Rank", 1.0, {"n_diffs": len(diffs), "non_zero": 0})
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "raw_diffs": diffs.tolist(),
            "n_non_zero": 0
        }

    try:
        stat, p_val = stats.wilcoxon(non_zero, alternative=alternative)
        if p_val == 0.0 or p_val == 1.0:
            log_degenerate_p_value("Wilcoxon Signed-Rank", p_val, {"stat": stat, "n_non_zero": len(non_zero)})
    except Exception as e:
        stat, p_val = 0.0, 1.0

    diff_mean = float(np.mean(diffs))
    if diff_mean > 0:
        dir_cand = "DS-HTE better"
    elif diff_mean < 0:
        dir_cand = "Baseline better"
    else:
        dir_cand = "Tie"

    direction = dir_cand if p_val < 0.05 else "No significant difference"

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "raw_diffs": diffs.tolist(),
        "n_non_zero": len(non_zero),
        "mean_diff": diff_mean,
        "direction": direction
    }



def nadeau_bengio_corrected_ttest(scores_a: np.ndarray, scores_b: np.ndarray,
                                  n_train: int = 80, n_test: int = 20) -> Dict[str, Any]:
    """
    Nadeau-Bengio corrected resampled t-test for repeated CV.
    Corrects variance estimate for lack of independence between CV splits.
    """
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    J = len(diffs)
    if J <= 1:
        return {"t_stat": 0.0, "p_value": 1.0, "df": 0}

    d_bar = np.mean(diffs)
    s2 = np.var(diffs, ddof=1)

    if s2 <= 1e-12:
        return {"t_stat": 0.0, "p_value": 1.0, "df": J - 1}

    correction = (1.0 / J) + (n_test / n_train)
    denom = np.sqrt(correction * s2)
    t_stat = d_bar / denom
    df = J - 1
    p_val = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df))

    if d_bar > 0:
        dir_cand = "DS-HTE better"
    elif d_bar < 0:
        dir_cand = "Baseline better"
    else:
        dir_cand = "Tie"
    direction = dir_cand if p_val < 0.05 else "No significant difference"

    return {
        "t_stat": float(t_stat),
        "p_value": float(p_val),
        "df": int(df),
        "mean_diff": float(d_bar),
        "direction": direction
    }


def mcnemar_test(y_true: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray) -> Dict[str, Any]:
    """
    McNemar test on paired predictions.
    Uses exact binomial test when discordant count b + c < 25.
    """
    y_true = np.asarray(y_true)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)

    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # b: A correct, B incorrect
    b = int(np.sum(correct_a & ~correct_b))
    # c: A incorrect, B correct
    c = int(np.sum(~correct_a & correct_b))

    total_discordant = b + c
    if total_discordant == 0:
        log_degenerate_p_value("McNemar", 1.0, {"b": b, "c": c})
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "method": "exact", "direction": "No significant difference"}

    if total_discordant < 25:
        # Exact binomial test under null p = 0.5
        p_val = stats.binomtest(b, n=total_discordant, p=0.5).pvalue
        stat = float(b)
        method = "exact_binomial"
    else:
        # Continuity-corrected chi-squared
        stat = ((abs(b - c) - 1.0) ** 2) / total_discordant
        p_val = stats.chi2.sf(stat, df=1)
        method = "continuity_corrected_chi2"

    if p_val == 0.0 or p_val == 1.0:
        log_degenerate_p_value("McNemar", p_val, {"b": b, "c": c, "stat": stat})

    if b > c:
        dir_cand = "DS-HTE better"
    elif b < c:
        dir_cand = "Baseline better"
    else:
        dir_cand = "Tie"
    direction = dir_cand if p_val < 0.05 else "No significant difference"

    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "b": b,
        "c": c,
        "method": method,
        "direction": direction
    }


def compute_midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        a = i
        while i < N - 1 and Z[i] == Z[i + 1]:
            i += 1
        b = i
        mid = (a + b + 2) / 2.0
        T[a : b + 1] = mid
        i += 1
    S = np.empty(N, dtype=float)
    S[J] = T
    return S


def fast_delong(y_true: np.ndarray, probas: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fast DeLong algorithm for calculating AUCs and covariance matrix.
    probas: list of probability prediction arrays for positive class
    """
    y_true = np.asarray(y_true)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    m = len(pos_idx)
    n = len(neg_idx)
    k = len(probas)

    v10 = np.zeros((m, k))
    v01 = np.zeros((n, k))
    aucs = np.zeros(k)

    for i in range(k):
        p = probas[i]
        midranks = compute_midrank(p)
        aucs[i] = (np.sum(midranks[pos_idx]) - m * (m + 1) / 2.0) / (m * n)

        # Placement values
        v10[:, i] = (midranks[pos_idx] - compute_midrank(p[pos_idx])) / n
        v01[:, i] = 1.0 - (midranks[neg_idx] - compute_midrank(p[neg_idx])) / m

    s10 = np.cov(v10, rowvar=False) if m > 1 else np.zeros((k, k))
    s01 = np.cov(v01, rowvar=False) if n > 1 else np.zeros((k, k))

    if k == 1:
        s10 = np.array([[s10]])
        s01 = np.array([[s01]])

    sigma = s10 / m + s01 / n
    return aucs, sigma


def delong_paired_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> Dict[str, Any]:
    """
    DeLong test comparing AUC of model A vs model B on the same patients.
    """
    aucs, sigma = fast_delong(y_true, [prob_a, prob_b])
    diff = aucs[0] - aucs[1]
    var_diff = sigma[0, 0] + sigma[1, 1] - 2 * sigma[0, 1]

    if var_diff <= 1e-12:
        return {"z_stat": 0.0, "p_value": 1.0, "auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "direction": "No significant difference"}

    z_stat = diff / np.sqrt(var_diff)
    p_val = 2.0 * (1.0 - stats.norm.cdf(abs(z_stat)))

    if diff > 0:
        dir_cand = "DS-HTE better"
    elif diff < 0:
        dir_cand = "Baseline better"
    else:
        dir_cand = "Tie"
    direction = dir_cand if p_val < 0.05 else "No significant difference"

    return {
        "z_stat": float(z_stat),
        "p_value": float(p_val),
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": float(diff),
        "direction": direction
    }



def friedman_nemenyi_test(matrix_scores: np.ndarray, model_names: List[str]) -> Dict[str, Any]:
    """
    Friedman test and Nemenyi post-hoc test across models.
    matrix_scores: (n_folds, n_models)
    """
    n_folds, n_models = matrix_scores.shape
    friedman_stat, p_val = stats.friedmanchisquare(*[matrix_scores[:, j] for j in range(n_models)])

    # Average ranks
    ranks = np.zeros_like(matrix_scores)
    for i in range(n_folds):
        ranks[i] = stats.rankdata(-matrix_scores[i])  # higher score = rank 1
    avg_ranks = {model_names[j]: float(np.mean(ranks[:, j])) for j in range(n_models)}

    # Nemenyi post-hoc matrix
    try:
        nemenyi_df = sp.posthoc_nemenyi_friedman(matrix_scores)
        nemenyi_df.columns = model_names
        nemenyi_df.index = model_names
        nemenyi_pvals = nemenyi_df.to_dict()
    except Exception:
        nemenyi_pvals = {}

    return {
        "friedman_stat": float(friedman_stat),
        "friedman_p": float(p_val),
        "average_ranks": avg_ranks,
        "nemenyi_pvalues": nemenyi_pvals
    }


def apply_p_adjustments(p_values: List[float]) -> Dict[str, List[float]]:
    """
    Applies Holm and Benjamini-Hochberg corrections.
    """
    if not p_values:
        return {"holm": [], "bh": []}
    _, p_holm, _, _ = multipletests(p_values, method="holm")
    _, p_bh, _, _ = multipletests(p_values, method="fdr_bh")
    return {
        "holm": [float(p) for p in p_holm],
        "bh": [float(p) for p in p_bh]
    }
