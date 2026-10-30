"""
Dynamic Feature Optimization (AFO) via Consensus Ranking.
Follows Section III-C:
- SHAP importance from model fit on training split
- Permutation importance
- Mutual information
- L1-logistic regression coefficients
- Borda count consensus aggregation
- Redundancy filter (|correlation| > threshold)
- Adaptive k selection
Fitted strictly on training data.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
import shap


class DynamicFeatureOptimizer(BaseEstimator, TransformerMixin):
    def __init__(self,
                 min_features: int = 4,
                 max_features: Optional[int] = None,
                 corr_threshold: float = 0.85,
                 selected_k: Optional[int] = None,
                 n_perm_repeats: int = 1,
                 random_state: int = 42):
        self.min_features = min_features
        self.max_features = max_features
        self.corr_threshold = corr_threshold
        self.selected_k = selected_k
        self.n_perm_repeats = n_perm_repeats
        self.random_state = random_state

        self.ranked_indices_ = []
        self.borda_scores_ = {}
        self.feature_rankings_ = {}
        self.selected_indices_ = []
        self.feature_names_in_ = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        n_samples, n_features = X.shape
        if feature_names is None:
            feature_names = [f"f_{i}" for i in range(n_features)]
        self.feature_names_in_ = feature_names

        # If features <= min_features, keep all
        if n_features <= self.min_features:
            self.ranked_indices_ = list(range(n_features))
            self.selected_indices_ = list(range(n_features))
            self.selected_k = n_features
            return self

        # 1. Fit tree model for SHAP & Permutation Importance
        rf = RandomForestClassifier(n_estimators=20, max_depth=5, random_state=self.random_state, n_jobs=1)
        rf.fit(X, y)

        # (a) SHAP importance
        try:
            explainer = shap.TreeExplainer(rf)
            shap_vals = explainer.shap_values(X)
            # In binary classification, shap_values may be a list [class0, class1] or 3D array
            if isinstance(shap_vals, list):
                shap_imp = np.mean(np.abs(shap_vals[1]), axis=0)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_imp = np.mean(np.abs(shap_vals[:, :, 1]), axis=0)
            else:
                shap_imp = np.mean(np.abs(shap_vals), axis=0)
        except Exception:
            shap_imp = rf.feature_importances_

        # (b) Permutation importance
        try:
            perm_res = permutation_importance(rf, X, y, n_repeats=self.n_perm_repeats, random_state=self.random_state, n_jobs=1)
            perm_imp = np.maximum(0, perm_res.importances_mean)
        except Exception:
            perm_imp = rf.feature_importances_

        # (c) Mutual information
        try:
            mi_imp = mutual_info_classif(X, y, random_state=self.random_state)
        except Exception:
            mi_imp = np.zeros(n_features)

        # (d) L1-logistic regression coefficients
        try:
            l1_clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=self.random_state, max_iter=200)
            l1_clf.fit(X, y)
            l1_imp = np.abs(l1_clf.coef_[0])
        except Exception:
            l1_imp = np.zeros(n_features)

        # Store individual importance scores
        self.feature_rankings_ = {
            "shap": shap_imp,
            "permutation": perm_imp,
            "mutual_info": mi_imp,
            "l1_logistic": l1_imp
        }

        # Compute ranks for each method (1 to n_features, higher is better)
        rank_matrix = np.zeros((4, n_features))
        for row_idx, imp in enumerate([shap_imp, perm_imp, mi_imp, l1_imp]):
            # argsort of argsort gives rank 0 to n_features-1
            rank_matrix[row_idx] = np.argsort(np.argsort(imp)) + 1

        # Borda count: sum of ranks across the 4 methods
        borda_sum = np.sum(rank_matrix, axis=0)
        self.borda_scores_ = {feature_names[i]: float(borda_sum[i]) for i in range(n_features)}

        # Sort features by descending Borda score
        sorted_by_borda = list(np.argsort(-borda_sum))

        # Redundancy filter (drop lower Borda ranked feature if |corr| > corr_threshold)
        kept_indices = []
        if self.corr_threshold < 1.0:
            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = np.corrcoef(X.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            for idx in sorted_by_borda:
                # Check if idx is strongly correlated with any already kept feature
                is_redundant = False
                for kept in kept_indices:
                    if abs(corr_matrix[idx, kept]) >= self.corr_threshold:
                        is_redundant = True
                        break
                if not is_redundant:
                    kept_indices.append(idx)
        else:
            kept_indices = sorted_by_borda

        # Ensure at least min_features are kept
        if len(kept_indices) < self.min_features:
            for idx in sorted_by_borda:
                if idx not in kept_indices:
                    kept_indices.append(idx)
                if len(kept_indices) >= self.min_features:
                    break

        self.ranked_indices_ = kept_indices

        # Default selected_k
        max_limit = self.max_features if self.max_features else len(self.ranked_indices_)
        k = self.selected_k if self.selected_k is not None else min(len(self.ranked_indices_), max(self.min_features, int(len(self.ranked_indices_) * 0.8)))
        k = max(self.min_features, min(k, max_limit, len(self.ranked_indices_)))
        self.selected_k = k
        self.selected_indices_ = self.ranked_indices_[:self.selected_k]

        return self

    def set_k(self, k: int):
        k = max(self.min_features, min(k, len(self.ranked_indices_)))
        self.selected_k = k
        self.selected_indices_ = self.ranked_indices_[:self.selected_k]

    def transform(self, X: np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        return X[:, self.selected_indices_]

    @property
    def selected_feature_names(self) -> List[str]:
        return [self.feature_names_in_[i] for i in self.selected_indices_]


class AdaptiveFeatureOptimizer(DynamicFeatureOptimizer):
    def __init__(self, k_top: Optional[int] = None, min_features: int = 4,
                 max_features: Optional[int] = None, corr_threshold: float = 0.85,
                 random_state: int = 42):
        super().__init__(
            min_features=min_features,
            max_features=max_features,
            corr_threshold=corr_threshold,
            selected_k=k_top,
            random_state=random_state
        )
        self.k_top = k_top
