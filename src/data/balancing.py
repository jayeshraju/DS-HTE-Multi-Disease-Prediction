"""
Adaptive Data Balancing Module.
Implements Borderline-SMOTE, SMOTE-NC, and Tomek Links boundary cleaning.
"""

import numpy as np
from typing import Tuple, List, Optional

try:
    from imblearn.over_sampling import BorderlineSMOTE, SMOTENC, SMOTE
    from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours
    from imblearn.combine import SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

class AdaptiveDataBalancer:
    """
    Leakage-free adaptive class balancer utilizing hybrid SMOTE + Tomek/ENN resampling.
    """
    def __init__(
        self,
        sampling_strategy: float = 0.85,
        method: str = "smote_tomek",
        categorical_indices: Optional[List[int]] = None,
        random_state: int = 42
    ):
        self.sampling_strategy = sampling_strategy
        self.method = method
        self.categorical_indices = categorical_indices or []
        self.random_state = random_state

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Applies class-balancing techniques strictly on training partitions.
        """
        if not IMBLEARN_AVAILABLE:
            raise ImportError("imbalanced-learn package is required for data balancing.")

        unique, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return X, y

        ir = max(counts) / min(counts)
        # Skip if dataset is already balanced (Imbalance Ratio <= 1.2)
        if ir <= 1.2:
            return X, y

        # 1. Initialize the appropriate SMOTE estimator
        if len(self.categorical_indices) > 0:
            # SMOTENC requires categorical features indices
            smote = SMOTENC(
                categorical_features=self.categorical_indices,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        else:
            if self.method == "borderline_smote":
                smote = BorderlineSMOTE(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state
                )
            else:
                smote = SMOTE(
                    sampling_strategy=self.sampling_strategy,
                    random_state=self.random_state
                )

        # 2. Apply resampling method
        if self.method == "smote_enn":
            balancer_instance = SMOTEENN(
                smote=smote,
                enn=EditedNearestNeighbours(sampling_strategy='all'),
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
            X_resampled, y_resampled = balancer_instance.fit_resample(X, y)
        elif self.method == "borderline_smote":
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            # Default: SMOTE + Tomek
            tomek = TomekLinks(sampling_strategy='all')
            X_sm, y_sm = smote.fit_resample(X, y)
            X_resampled, y_resampled = tomek.fit_resample(X_sm, y_sm)
            
        return X_resampled, y_resampled
