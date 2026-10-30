"""
Disease-Specific Hybrid Transformer Ensemble (DS-HTE).
Follows Section III-D & Section A8:
- Integrates Tabular Transformer, XGBoost, LightGBM, CatBoost, LR, and RF
- Non-negative regularized stacking / weighted soft-vote fitted on inner OOF predictions
- Minimum-weight floor and entropy/diversity penalty to prevent collapse
- Disease-specific candidate subsets and learned weights
- Sklearn-compatible API
"""

from typing import Dict, List, Optional, Any
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.transformer import TabularTransformerClassifier
from src.models.gbdt import XGBoostWrapper, LightGBMWrapper, CatBoostWrapper


class DSHTEClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 include_transformer: bool = True,
                 include_xgb: bool = True,
                 include_lgbm: bool = True,
                 include_catboost: bool = True,
                 include_lr: bool = False,
                 include_rf: bool = False,
                 transformer_params: Optional[Dict[str, Any]] = None,
                 xgb_params: Optional[Dict[str, Any]] = None,
                 lgbm_params: Optional[Dict[str, Any]] = None,
                 catboost_params: Optional[Dict[str, Any]] = None,
                 lr_params: Optional[Dict[str, Any]] = None,
                 rf_params: Optional[Dict[str, Any]] = None,
                 min_weight_floor: float = 0.01,
                 entropy_penalty: float = 0.005,
                 n_inner_splits: int = 3,
                 random_state: int = 42):
        self.include_transformer = include_transformer
        self.include_xgb = include_xgb
        self.include_lgbm = include_lgbm
        self.include_catboost = include_catboost
        self.include_lr = include_lr
        self.include_rf = include_rf
        self.transformer_params = transformer_params or {}
        self.xgb_params = xgb_params or {}
        self.lgbm_params = lgbm_params or {}
        self.catboost_params = catboost_params or {}
        self.lr_params = lr_params or {}
        self.rf_params = rf_params or {}
        self.min_weight_floor = min_weight_floor
        self.entropy_penalty = entropy_penalty
        self.n_inner_splits = n_inner_splits
        self.random_state = random_state

        self.models_ = {}
        self.weights_ = {}
        self.model_names_ = []
        self.fitted_ = False

    def _init_models(self) -> Dict[str, BaseEstimator]:
        models = {}
        if self.include_transformer:
            params = {"random_state": self.random_state, **self.transformer_params}
            models["transformer"] = TabularTransformerClassifier(**params)
        if self.include_xgb:
            params = {"random_state": self.random_state, **self.xgb_params}
            models["xgb"] = XGBoostWrapper(**params)
        if self.include_lgbm:
            params = {"random_state": self.random_state, **self.lgbm_params}
            models["lgbm"] = LightGBMWrapper(**params)
        if self.include_catboost:
            params = {"random_state": self.random_state, **self.catboost_params}
            models["catboost"] = CatBoostWrapper(**params)
        if self.include_lr:
            params = {"random_state": self.random_state, "max_iter": 500, **self.lr_params}
            models["lr"] = LogisticRegression(**params)
        if self.include_rf:
            params = {"random_state": self.random_state, "n_estimators": 50, **self.rf_params}
            models["rf"] = RandomForestClassifier(**params)
        return models

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        base_models = self._init_models()
        self.model_names_ = list(base_models.keys())

        if not self.model_names_:
            raise ValueError("At least one constituent model must be included in DS-HTE.")

        n_samples = len(X)
        n_models = len(self.model_names_)

        # If only one model, weight is 1.0
        if n_models == 1:
            name = self.model_names_[0]
            m = base_models[name]
            m.fit(X, y)
            self.models_ = {name: m}
            self.weights_ = {name: 1.0}
            self.oof_probs_ = m.predict_proba(X)[:, 1]
            self.fitted_ = True
            return self

        # Generate Out-Of-Fold probability predictions for weight learning
        oof_probs = np.zeros((n_samples, n_models))
        cv = StratifiedKFold(n_splits=self.n_inner_splits, shuffle=True, random_state=self.random_state)

        for tr_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            fold_models = self._init_models()
            for col_idx, name in enumerate(self.model_names_):
                m = fold_models[name]
                if name == "transformer":
                    m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
                else:
                    m.fit(X_tr, y_tr)
                oof_probs[val_idx, col_idx] = m.predict_proba(X_val)[:, 1]

        # Constrained optimization with entropy regularization to prevent collapse
        floor = max(0.0, float(self.min_weight_floor))
        lambda_ent = max(0.0, float(self.entropy_penalty))

        def objective(w):
            p = np.dot(oof_probs, w)
            p = np.clip(p, 1e-6, 1 - 1e-6)
            loss = log_loss(y, p)
            # Entropy regularization: -sum(w * log(w))
            w_safe = np.clip(w, 1e-12, 1.0)
            entropy = -np.sum(w_safe * np.log(w_safe))
            return loss - lambda_ent * entropy

        init_w = np.ones(n_models) / n_models
        bounds = [(floor, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        opt_res = minimize(objective, init_w, bounds=bounds, constraints=constraints, method="SLSQP")
        if opt_res.success and np.all(opt_res.x >= floor * 0.99):
            weights = opt_res.x / np.sum(opt_res.x)
        else:
            weights = init_w

        self.weights_ = {name: float(w) for name, w in zip(self.model_names_, weights)}
        self.oof_probs_ = np.dot(oof_probs, weights)

        # Fit all constituent models on full dataset (X, y)
        self.models_ = self._init_models()
        for name in self.model_names_:
            m = self.models_[name]
            m.fit(X, y)

        self.fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("DSHTEClassifier is not fitted.")

        combined_prob_1 = np.zeros(len(X))
        for name in self.model_names_:
            w = self.weights_.get(name, 0.0)
            if w > 0:
                p1 = self.models_[name].predict_proba(X)[:, 1]
                combined_prob_1 += w * p1

        combined_prob_1 = np.clip(combined_prob_1, 0.0, 1.0)
        combined_prob_0 = 1.0 - combined_prob_1
        return np.column_stack([combined_prob_0, combined_prob_1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)
