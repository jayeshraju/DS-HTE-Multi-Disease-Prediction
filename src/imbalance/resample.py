"""
Class imbalance handling strategies.
Follows Section III-E:
- Strategies: none, class_weight, smote, borderline_smote
- Applied strictly to training data
- Guarantees validation sets remain un-resampled
"""

from typing import Tuple, Optional, Dict
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, BorderlineSMOTE


def apply_imbalance_strategy(strategy: str,
                             X: np.ndarray,
                             y: np.ndarray,
                             random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[int, float]]]:
    strategy = str(strategy).lower()
    
    if strategy in ["none", "no_resampling"]:
        return X, y, None

    elif strategy in ["class_weight", "class_weights"]:
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        weight_dict = {cls: float(w) for cls, w in zip(classes, weights)}
        return X, y, weight_dict

    elif strategy == "smote":
        min_class_count = int(np.min(np.bincount(y)))
        k_neighbors = max(1, min(5, min_class_count - 1))
        smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res, None

    elif strategy in ["borderline", "borderline_smote", "borderlinesmote"]:
        min_class_count = int(np.min(np.bincount(y)))
        k_neighbors = max(1, min(5, min_class_count - 1))
        try:
            bsmote = BorderlineSMOTE(k_neighbors=k_neighbors, random_state=random_state)
            X_res, y_res = bsmote.fit_resample(X, y)
            return X_res, y_res, None
        except Exception:
            # Fallback to standard SMOTE if borderline points cannot be found
            smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
            X_res, y_res = smote.fit_resample(X, y)
            return X_res, y_res, None

    else:
        raise ValueError(f"Unknown imbalance strategy '{strategy}'. Options: none, class_weight, smote, borderline_smote")
