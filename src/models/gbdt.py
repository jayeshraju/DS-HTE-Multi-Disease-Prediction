"""
GBDT model wrappers for XGBoost, LightGBM, and CatBoost.
Provides consistent sklearn-compatible classifiers.
"""

from typing import Dict, Any, Optional
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


class XGBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 4,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 scale_pos_weight: float = 1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.scale_pos_weight = scale_pos_weight
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model_ = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            scale_pos_weight=self.scale_pos_weight,
            eval_metric="logloss",
            verbosity=0
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


class LightGBMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = -1,
                 num_leaves: int = 31,
                 learning_rate: float = 0.05,
                 subsample: float = 0.8,
                 colsample_bytree: float = 0.8,
                 random_state: int = 42,
                 class_weight: Optional[str] = None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.class_weight = class_weight
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model_ = lgb.LGBMClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            class_weight=self.class_weight,
            verbosity=-1
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)


class CatBoostWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 iterations: int = 100,
                 depth: int = 5,
                 learning_rate: float = 0.05,
                 l2_leaf_reg: float = 3.0,
                 random_state: int = 42,
                 auto_class_weights: Optional[str] = None):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.random_state = random_state
        self.auto_class_weights = auto_class_weights
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=self.random_state,
            auto_class_weights=self.auto_class_weights,
            verbose=False
        )
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)
