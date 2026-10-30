"""
Baseline models evaluated against DS-HTE:
- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)
- LightGBM (LGBM)
- CatBoost (CB)
- Transformer-only (without ensemble)
"""

from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from src.models.gbdt import XGBoostWrapper, LightGBMWrapper, CatBoostWrapper
from src.models.transformer import TabularTransformerClassifier


def get_baseline_model(name: str, random_state: int = 42, **kwargs):
    name = name.lower().replace("-", "_")
    if name in ["dummy_most_frequent", "dummy_majority"]:
        return DummyClassifier(strategy="most_frequent", random_state=random_state)
    elif name in ["dummy_stratified"]:
        return DummyClassifier(strategy="stratified", random_state=random_state)
    elif name in ["lr", "logistic_regression", "lr_plain"]:
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            max_iter=kwargs.get("max_iter", 500),
            random_state=random_state
        )
    elif name in ["rf", "random_forest", "rf_plain"]:
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 6),
            random_state=random_state
        )
    elif name in ["xgb", "xgboost"]:
        return XGBoostWrapper(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 4),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=random_state
        )
    elif name in ["lgb", "lgbm", "lightgbm"]:
        return LightGBMWrapper(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", -1),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=random_state
        )
    elif name in ["cb", "catboost"]:
        return CatBoostWrapper(
            iterations=kwargs.get("n_estimators", 100),
            depth=kwargs.get("max_depth", 5),
            learning_rate=kwargs.get("learning_rate", 0.05),
            random_state=random_state
        )
    elif name in ["transformer", "transformer_only"]:
        return TabularTransformerClassifier(
            epochs=kwargs.get("epochs", 10),
            d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            random_state=random_state
        )
    else:
        raise ValueError(f"Unknown baseline name '{name}'")

