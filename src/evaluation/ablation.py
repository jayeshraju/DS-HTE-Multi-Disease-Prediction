"""
Ablation study module.
Evaluates 7 variants on identical folds:
1. Full DS-HTE
2. Minus Feature Optimization (all features retained)
3. Minus Imbalance Handling (strategy = 'none')
4. Minus Probability Calibration (uncalibrated probabilities)
5. Minus Decision Threshold Optimization (fixed threshold = 0.5)
6. Transformer-only (no ensemble)
7. GBDT-only ensemble (no Transformer)
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.pipeline.protocol import fit_and_evaluate_pipeline
from src.models.dshte import DSHTEClassifier
from src.models.transformer import TabularTransformerClassifier
from src.evaluation.metrics import compute_all_metrics


def run_ablation_on_fold(X_tr: pd.DataFrame, y_tr: pd.Series,
                         X_te: pd.DataFrame, y_te: pd.Series,
                         num_cols: List[str], cat_cols: List[str],
                         selected_imbalance: str = "none",
                         transformer_epochs: int = 10,
                         gbdt_estimators: int = 50,
                         random_state: int = 42) -> Dict[str, Dict[str, float]]:
    ablation_results = {}

    configs = {
        "Full DS-HTE": {
            "model_class": DSHTEClassifier,
            "model_kwargs": {
                "include_transformer": True, "include_xgb": True, "include_lgbm": True, "include_catboost": True,
                "transformer_params": {"epochs": transformer_epochs},
                "xgb_params": {"n_estimators": gbdt_estimators},
                "lgbm_params": {"n_estimators": gbdt_estimators},
                "catboost_params": {"iterations": gbdt_estimators},
                "random_state": random_state
            },
            "afo": True, "imbalance": selected_imbalance, "calib": True, "thresh": True
        },
        "Minus Feature Optimization": {
            "model_class": DSHTEClassifier,
            "model_kwargs": {
                "include_transformer": True, "include_xgb": True, "include_lgbm": True, "include_catboost": True,
                "transformer_params": {"epochs": transformer_epochs},
                "xgb_params": {"n_estimators": gbdt_estimators},
                "lgbm_params": {"n_estimators": gbdt_estimators},
                "catboost_params": {"iterations": gbdt_estimators},
                "random_state": random_state
            },
            "afo": False, "imbalance": selected_imbalance, "calib": True, "thresh": True
        },
        "Minus Imbalance Handling": {
            "model_class": DSHTEClassifier,
            "model_kwargs": {
                "include_transformer": True, "include_xgb": True, "include_lgbm": True, "include_catboost": True,
                "transformer_params": {"epochs": transformer_epochs},
                "xgb_params": {"n_estimators": gbdt_estimators},
                "lgbm_params": {"n_estimators": gbdt_estimators},
                "catboost_params": {"iterations": gbdt_estimators},
                "random_state": random_state
            },
            "afo": True, "imbalance": "none", "calib": True, "thresh": True
        },
        "Minus Calibration": {
            "model_class": DSHTEClassifier,
            "model_kwargs": {
                "include_transformer": True, "include_xgb": True, "include_lgbm": True, "include_catboost": True,
                "transformer_params": {"epochs": transformer_epochs},
                "xgb_params": {"n_estimators": gbdt_estimators},
                "lgbm_params": {"n_estimators": gbdt_estimators},
                "catboost_params": {"iterations": gbdt_estimators},
                "random_state": random_state
            },
            "afo": True, "imbalance": selected_imbalance, "calib": False, "thresh": True
        },
        "Minus Threshold Optimization": {
            "model_class": DSHTEClassifier,
            "model_kwargs": {
                "include_transformer": True, "include_xgb": True, "include_lgbm": True, "include_catboost": True,
                "transformer_params": {"epochs": transformer_epochs},
                "xgb_params": {"n_estimators": gbdt_estimators},
                "lgbm_params": {"n_estimators": gbdt_estimators},
                "catboost_params": {"iterations": gbdt_estimators},
                "random_state": random_state
            },
            "afo": True, "imbalance": selected_imbalance, "calib": True, "thresh": False
        },
        "Transformer-Only": {
            "model_class": TabularTransformerClassifier,
            "model_kwargs": {
                "epochs": transformer_epochs,
                "random_state": random_state
            },
            "afo": True, "imbalance": selected_imbalance, "calib": True, "thresh": True
        },
        "GBDT-Only Ensemble": {
            "model_class": DSHTEClassifier,
            "model_kwargs": {
                "include_transformer": False, "include_xgb": True, "include_lgbm": True, "include_catboost": True,
                "xgb_params": {"n_estimators": gbdt_estimators},
                "lgbm_params": {"n_estimators": gbdt_estimators},
                "catboost_params": {"iterations": gbdt_estimators},
                "random_state": random_state
            },
            "afo": True, "imbalance": selected_imbalance, "calib": True, "thresh": True
        }
    }

    for variant_name, cfg in configs.items():
        res = fit_and_evaluate_pipeline(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te, y_test=y_te,
            numerical_cols=num_cols, categorical_cols=cat_cols,
            model_class=cfg["model_class"],
            model_kwargs=cfg["model_kwargs"],
            imbalance_strategy=cfg["imbalance"],
            afo_enabled=cfg["afo"],
            calibration_enabled=cfg["calib"],
            threshold_enabled=cfg["thresh"],
            random_state=random_state
        )
        metrics = compute_all_metrics(res["y_true"], res["y_pred"], res["y_prob"])
        ablation_results[variant_name] = metrics

    return ablation_results
