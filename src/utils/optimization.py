"""
Hyperparameter Optimization Engine using Optuna (TPE).
Implements automated Bayesian hyperparameter tuning for baseline and hybrid models.
"""

from typing import Dict, Any, Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def tune_xgboost_optuna(
    X: np.ndarray,
    y: np.ndarray,
    n_trials: int = 50,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Runs Optuna TPE hyperparameter optimization for XGBoost optimizing 10-Fold CV F1-Score.
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna package is required for hyperparameter optimization.")

    import xgboost as xgb

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric": "logloss",
            "random_state": random_seed
        }

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
        f1_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            model = xgb.XGBClassifier(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_va)
            f1_scores.append(f1_score(y_va, preds, zero_division=0))

        return float(np.mean(f1_scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_seed))
    study.optimize(objective, n_trials=n_trials)

    return study.best_params
