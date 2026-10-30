"""
Error Analysis & Subgroup Profiling Module.
Follows Section A11:
- Only reports subgroups/clusters when subgroup size >= 30 patients.
- Suppresses clusters with < 30 patients with explicit message: "subgroup too small".
- Computes False Negative Rate (FNR) with Wilson 95% confidence intervals.
- Never reports FNR for clusters of size 3-14.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_wilson_ci(k: int, n: int, confidence: float = 0.95) -> Dict[str, float]:
    """
    Computes Wilson score interval for binomial proportion k/n.
    """
    if n == 0:
        return {"proportion": 0.0, "lower": 0.0, "upper": 0.0}
    p_hat = float(k / n)
    z = stats.norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    spread = z * np.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n) / denom
    lower = max(0.0, float(center - spread))
    upper = min(1.0, float(center + spread))
    return {"proportion": p_hat, "lower": lower, "upper": upper}


def profile_error_subgroups(df_features: pd.DataFrame,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            min_subgroup_size: int = 30) -> List[Dict[str, Any]]:
    """
    Analyzes false negative rate across clinical feature subgroups (e.g., age, sex, biomarker quantiles).
    Suppresses any subgroup with size < min_subgroup_size (default 30).
    """
    results = []
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # False negative indicator (true positive case misclassified as negative)
    fn_mask = (y_true == 1) & (y_pred == 0)
    pos_mask = (y_true == 1)

    for col in df_features.columns:
        series = df_features[col]
        # Discretize continuous features into bins (e.g. tertiles)
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 5:
            try:
                bins = pd.qcut(series, q=3, labels=["Low", "Medium", "High"], duplicates="drop")
            except Exception:
                continue
        else:
            bins = series.astype(str)

        for group_val, grp_idx in bins.groupby(bins, observed=False).groups.items():
            grp_indices = np.asarray(grp_idx)
            subgroup_size = len(grp_indices)
            
            # Check subgroup size threshold
            if subgroup_size < min_subgroup_size:
                results.append({
                    "feature": col,
                    "subgroup": str(group_val),
                    "size": subgroup_size,
                    "status": "subgroup too small (n < 30)",
                    "fnr": None,
                    "fnr_ci": None
                })
                continue

            # Number of positive cases in this subgroup
            subgroup_pos_indices = grp_indices[pos_mask[grp_indices]]
            n_pos = len(subgroup_pos_indices)
            if n_pos == 0:
                continue

            n_fn = int(np.sum(fn_mask[subgroup_pos_indices]))
            ci = compute_wilson_ci(n_fn, n_pos, confidence=0.95)

            results.append({
                "feature": col,
                "subgroup": str(group_val),
                "size": subgroup_size,
                "n_positives": n_pos,
                "n_false_negatives": n_fn,
                "status": "evaluated",
                "fnr": ci["proportion"],
                "fnr_lower": ci["lower"],
                "fnr_upper": ci["upper"]
            })

    return results
