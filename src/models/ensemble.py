"""
High-Performance Stacking Ensemble with Optimal Threshold Selection.
Combines XGBoost, LightGBM, CatBoost, ExtraTrees, and the HDA-GB transformer
with a calibrated meta-learner and per-disease Youden-J threshold optimization.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False


class HighPerformanceStackingEnsemble(BaseEstimator, ClassifierMixin):
    """
    Level-0:  XGBoost + LightGBM + CatBoost + ExtraTrees + RandomForest
    Level-1:  CalibratedClassifierCV(LogisticRegression) meta-learner
    Post-fit: Youden's J threshold optimization on OOF predictions
    """

    def __init__(
        self,
        n_folds: int = 5,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        cb_params: Optional[Dict] = None,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.cb_params = cb_params or {}
        self.random_state = random_state

        self.base_models_: List = []
        self.meta_learner_: Optional[BaseEstimator] = None
        self.optimal_threshold_: float = 0.5
        self.classes_: np.ndarray = np.array([0, 1])

    def _build_base_models(self) -> List:
        models = []

        if XGB_AVAILABLE:
            xgb_defaults = dict(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0,
                eval_metric="logloss", random_state=self.random_state,
                use_label_encoder=False
            )
            xgb_defaults.update(self.xgb_params)
            models.append(("xgb", xgb.XGBClassifier(**xgb_defaults)))

        if LGB_AVAILABLE:
            lgb_defaults = dict(
                n_estimators=800, max_depth=6, learning_rate=0.03,
                subsample=0.8, colsample_bytree=0.8,
                num_leaves=63, min_child_samples=10,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=self.random_state, verbose=-1
            )
            lgb_defaults.update(self.lgb_params)
            models.append(("lgb", lgb.LGBMClassifier(**lgb_defaults)))

        if CB_AVAILABLE:
            cb_defaults = dict(
                iterations=800, depth=6, learning_rate=0.03,
                l2_leaf_reg=3, random_seed=self.random_state,
                verbose=False
            )
            cb_defaults.update(self.cb_params)
            models.append(("cb", cb.CatBoostClassifier(**cb_defaults)))

        models.append(("et", ExtraTreesClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=self.random_state
        )))
        models.append(("rf", RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=self.random_state
        )))
        return models

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes_ = np.unique(y)
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        self.base_models_ = self._build_base_models()
        n_models = len(self.base_models_)

        # ── Out-of-fold meta-features ─────────────────────────────────
        oof_meta = np.zeros((X.shape[0], n_models))

        for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr         = y[train_idx]
            for m_i, (name, model) in enumerate(self.base_models_):
                clone_m = model.__class__(**model.get_params())
                clone_m.fit(X_tr, y_tr)
                oof_meta[val_idx, m_i] = clone_m.predict_proba(X_val)[:, 1]

        # ── Re-fit base models on all training data ───────────────────
        fitted_models = []
        for name, model in self.base_models_:
            model.fit(X, y)
            fitted_models.append((name, model))
        self.base_models_ = fitted_models

        # ── Fit calibrated meta-learner on OOF predictions ────────────
        lr = LogisticRegression(C=1.0, max_iter=500, random_state=self.random_state)
        self.meta_learner_ = CalibratedClassifierCV(lr, cv=3, method="sigmoid")
        self.meta_learner_.fit(oof_meta, y)

        # ── Optimize decision threshold via Youden's J on OOF ─────────
        oof_meta_proba = self.meta_learner_.predict_proba(oof_meta)[:, 1]
        fpr, tpr, thresholds = roc_curve(y, oof_meta_proba)
        youden_j = tpr - fpr
        self.optimal_threshold_ = float(thresholds[np.argmax(youden_j)])
        # Clamp to safe range
        self.optimal_threshold_ = np.clip(self.optimal_threshold_, 0.2, 0.8)

        print(f"   Stacking trained on {X.shape[0]} samples | "
              f"{n_models} base models | Optimal threshold: {self.optimal_threshold_:.3f}")
        return self

    def _meta_proba(self, X: np.ndarray) -> np.ndarray:
        meta_feats = np.column_stack([
            m.predict_proba(X)[:, 1] for _, m in self.base_models_
        ])
        return self.meta_learner_.predict_proba(meta_feats)[:, 1]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p1 = self._meta_proba(X)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self._meta_proba(X) >= self.optimal_threshold_).astype(int)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        proba = self._meta_proba(X)
        return 2.0 * np.abs(proba - 0.5)
