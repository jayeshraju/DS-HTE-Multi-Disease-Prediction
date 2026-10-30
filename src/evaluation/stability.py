"""
Feature selection stability and frequency analysis.
Follows Section 3.2 & Section 5:
- Jaccard similarity index
- Kuncheva consistency index:
  I_C(A, B) = (r * n - k^2) / (k * (n - k))
  where r = |A \cap B|, k = |A| = |B|, n = total_features.
- Undefined cases (k=0, k=n) handled cleanly.
- Range bounded in [-1.0, 1.0].
"""

import logging
from typing import List, Dict, Any, Set, Optional
import numpy as np

logger = logging.getLogger(__name__)


def jaccard_index(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Computes pairwise Jaccard similarity between two feature sets.
    """
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union == 0:
        return 1.0
    return float(intersection / union)


def kuncheva_index(set_a: Set[str], set_b: Set[str], total_features: int) -> float:
    """
    Computes Kuncheva consistency index for two subsets A, B of size k from n features:
    (r * n - k^2) / (k * (n - k)), where r = |A \cap B|.
    Must lie in [-1, 1].
    Handles k = 0 and k = n (undefined, skips/logs and returns NaN).
    If subset sizes differ, uses the generalized formulation:
    (r * n - k_a * k_b) / (min(k_a, k_b) * n - k_a * k_b)
    """
    if total_features <= 0:
        raise ValueError("total_features must be strictly positive.")

    k_a = len(set_a)
    k_b = len(set_b)
    n = max(total_features, k_a, k_b)
    r = len(set_a.intersection(set_b))

    # Boundary cases: k = 0 or k = n return nan
    if k_a == 0 or k_b == 0:
        logger.warning("Kuncheva index undefined for empty subset (k=0).")
        return float("nan")

    if k_a == n and k_b == n:
        logger.warning("Kuncheva index undefined when all features selected (k=n).")
        return float("nan")

    if set_a == set_b and k_a == k_b:
        return 1.0

    if k_a == k_b:
        k = k_a
        denom = float(k * (n - k))
        if denom == 0:
            logger.warning("Kuncheva denominator is zero.")
            return float("nan")
        num = float(r * n - (k ** 2))
        score = num / denom
    else:
        # Generalized Kuncheva for unequal subset sizes
        denom = float(min(k_a, k_b) * n - (k_a * k_b))
        if denom == 0:
            logger.warning("Generalized Kuncheva denominator is zero.")
            return float("nan")
        num = float(r * n - (k_a * k_b))
        score = num / denom

    # Verification: must lie in [-1, 1]
    if score < -1.0001 or score > 1.0001:
        raise ValueError(f"Kuncheva index {score} is outside [-1, 1]! Check inputs: r={r}, k_a={k_a}, k_b={k_b}, n={n}")

    return float(np.clip(score, -1.0, 1.0))


def evaluate_feature_stability(feature_subsets: List[List[str]], total_features: int) -> Dict[str, Any]:
    """
    Computes mean pairwise Jaccard and Kuncheva stability across folds.
    Handles undefined pairs and ensures stability metrics lie in valid ranges.
    """
    n_subsets = len(feature_subsets)
    if n_subsets <= 1:
        return {"mean_jaccard": 1.0, "mean_kuncheva": 1.0, "frequencies": {}}

    sets = [set(s) for s in feature_subsets]
    all_features_pool = set.union(*sets) if sets else set()
    effective_n = max(int(total_features), len(all_features_pool), max(len(s) for s in sets) if sets else 1)

    jaccard_scores = []
    kuncheva_scores = []

    for i in range(n_subsets):
        for j in range(i + 1, n_subsets):
            jaccard_scores.append(jaccard_index(sets[i], sets[j]))
            k_val = kuncheva_index(sets[i], sets[j], effective_n)
            if not np.isnan(k_val):
                kuncheva_scores.append(k_val)

    # Feature selection frequencies
    all_selected = [feat for s in feature_subsets for feat in s]
    unique_feats = set(all_selected)
    frequencies = {feat: float(all_selected.count(feat) / n_subsets) for feat in unique_feats}

    mean_k = float(np.mean(kuncheva_scores)) if len(kuncheva_scores) > 0 else 0.0

    return {
        "mean_jaccard": float(np.mean(jaccard_scores)),
        "mean_kuncheva": mean_k,
        "frequencies": dict(sorted(frequencies.items(), key=lambda x: x[1], reverse=True))
    }
