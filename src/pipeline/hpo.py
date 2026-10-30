"""
Optuna hyperparameter optimization module for DS-HTE and baselines.
Uses inner CV on outer-train set with cached preprocessing and feature sets.
"""

from typing import Dict, Any, List
import numpy as np
import optuna
from sklearn.metrics import f1_score, matthews_corrcoef, brier_score_loss
from sklearn.model_selection import StratifiedKFold

from src.imbalance.resample import apply_imbalance_strategy
from src.models.dshte import DSHTEClassifier


def optimize_dshte_hyperparameters(X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   n_trials: int = 10,
                                   n_inner_splits: int = 3,
                                   gbdt_n_estimators: int = 50,
                                   transformer_epochs: int = 10,
                                   random_state: int = 42) -> Dict[str, Any]:
    """
    Tuning via Optuna on inner CV splits.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Pre-generate inner CV splits
    inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)
    splits = list(inner_cv.split(X_train, y_train))

    def objective(trial: optuna.Trial) -> float:
        # Suggest imbalance strategy
        imbalance_strategy = trial.suggest_categorical(
            "imbalance_strategy", ["none", "class_weight", "smote", "borderline_smote"]
        )

        # Suggest hyperparameters
        learning_rate = trial.suggest_float("lr", 0.01, 0.08, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 6)
        d_model = trial.suggest_categorical("d_model", [32, 64])

        scores = []
        for in_tr_idx, in_val_idx in splits:
            X_tr, y_tr = X_train[in_tr_idx], y_train[in_tr_idx]
            X_val, y_val = X_train[in_val_idx], y_train[in_val_idx]

            # Apply imbalance strategy to inner train only
            X_tr_res, y_tr_res, _ = apply_imbalance_strategy(
                imbalance_strategy, X_tr, y_tr, random_state=random_state
            )

            model = DSHTEClassifier(
                transformer_params={"d_model": d_model, "epochs": transformer_epochs, "lr": learning_rate},
                xgb_params={"learning_rate": learning_rate, "max_depth": max_depth, "n_estimators": gbdt_n_estimators},
                lgbm_params={"learning_rate": learning_rate, "max_depth": max_depth, "n_estimators": gbdt_n_estimators},
                catboost_params={"learning_rate": learning_rate, "depth": max_depth, "iterations": gbdt_n_estimators},
                n_inner_splits=2,
                random_state=random_state
            )
            model.fit(X_tr_res, y_tr_res)
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)

            f1 = f1_score(y_val, preds, zero_division=0)
            mcc = matthews_corrcoef(y_val, preds)
            brier = brier_score_loss(y_val, probs)
            composite = 0.4 * f1 + 0.3 * (mcc if not np.isnan(mcc) else 0.0) + 0.3 * (1.0 - brier)
            scores.append(composite)

        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_imbalance = best_params.get("imbalance_strategy", "none")

    return {
        "imbalance_strategy": best_imbalance,
        "best_params": best_params,
        "best_value": study.best_value
    }
