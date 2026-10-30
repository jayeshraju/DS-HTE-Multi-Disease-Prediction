"""
Protocol engine for DS-HTE framework.
Follows Section IV-B & Non-Negotiable Rules:
- 20% Stratified Holdout isolated first and saved to results/logs/holdout_indices.json
- Repeated stratified CV on dev set (5 folds x 10 repeats = 50 folds), saved to results/logs/folds.json
- Inner 3-fold CV on outer-train set for HPO, imbalance choice, weights, calibration, threshold
- Read-only Inner-CV cache for preprocessed and feature-selected arrays
- Strict leakage barriers: holdout touched exactly once at final confirmatory evaluation
"""

import os
import json
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from src.data.load import load_cohort
from src.preprocessing.transforms import ClinicalPreprocessor
from src.features.afo import DynamicFeatureOptimizer
from src.imbalance.resample import apply_imbalance_strategy
from src.calibration.calibrate import choose_best_calibrator, ModelCalibrator
from src.threshold.youden import optimize_youden_threshold, aggregate_fold_thresholds


def get_duplicate_clusters(X: pd.DataFrame) -> np.ndarray:
    n_samples = len(X)
    clusters = np.arange(n_samples)
    exact_dup_mask = X.duplicated(keep=False)
    if exact_dup_mask.sum() > 0:
        dup_groups = X.groupby(list(X.columns)).groups
        for grp_indices in dup_groups.values():
            if len(grp_indices) > 1:
                first_idx = grp_indices[0]
                for idx in grp_indices:
                    clusters[idx] = first_idx
    return clusters


def build_and_save_protocol_splits(cohorts: List[str],
                                   raw_dir: str = "data/raw",
                                   logs_dir: str = "results/logs",
                                   seed: int = 42,
                                   n_repeats: int = 10,
                                   n_outer_folds: int = 5) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    os.makedirs(logs_dir, exist_ok=True)
    holdout_file = os.path.join(logs_dir, "holdout_indices.json")
    folds_file = os.path.join(logs_dir, "folds.json")

    holdout_dict = {}
    folds_dict = {}

    for cohort in cohorts:
        X, y, num_cols, cat_cols = load_cohort(cohort, raw_dir)
        n_samples = len(X)
        clusters = get_duplicate_clusters(X)
        has_clusters = len(np.unique(clusters)) < n_samples

        # 1. Stratified Holdout (20%)
        if has_clusters:
            splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y, groups=clusters))
        else:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            splits = list(splitter.split(X, y))

        dev_idx_rel, holdout_idx = splits[0]
        # Invert: split 0 test part is holdout (20%), dev is train part (80%)
        holdout_indices = [int(i) for i in sorted(holdout_idx)]
        dev_indices = [int(i) for i in sorted(dev_idx_rel)]

        holdout_dict[cohort] = {
            "n_total": n_samples,
            "n_holdout": len(holdout_indices),
            "n_dev": len(dev_indices),
            "holdout_indices": holdout_indices,
            "dev_indices": dev_indices
        }

        # 2. Outer Repeated CV on dev set (80%)
        dev_y = y.iloc[dev_indices].values
        dev_clusters = clusters[dev_indices]
        
        cohort_repeats = []
        for r in range(n_repeats):
            repeat_seed = seed + 100 * (r + 1)
            if has_clusters:
                cv_outer = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=repeat_seed)
                outer_splits = list(cv_outer.split(dev_indices, dev_y, groups=dev_clusters))
            else:
                cv_outer = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=repeat_seed)
                outer_splits = list(cv_outer.split(dev_indices, dev_y))

            folds_list = []
            for fold_i, (train_sub_idx, val_sub_idx) in enumerate(outer_splits):
                # Map relative indices back to original dataset row indices
                tr_indices = [dev_indices[i] for i in train_sub_idx]
                val_indices = [dev_indices[i] for i in val_sub_idx]

                # Strict isolation assertion
                assert set(tr_indices).isdisjoint(set(holdout_indices)), "Leakage! Train contains holdout!"
                assert set(val_indices).isdisjoint(set(holdout_indices)), "Leakage! Val contains holdout!"
                assert set(tr_indices).isdisjoint(set(val_indices)), "Fold overlap!"

                folds_list.append({
                    "fold": fold_i,
                    "train_indices": tr_indices,
                    "val_indices": val_indices
                })

            cohort_repeats.append({
                "repeat": r,
                "repeat_seed": repeat_seed,
                "folds": folds_list
            })

        folds_dict[cohort] = {
            "dev_indices": dev_indices,
            "repeats": cohort_repeats
        }

    with open(holdout_file, "w", encoding="utf-8") as f:
        json.dump(holdout_dict, f, indent=2)

    with open(folds_file, "w", encoding="utf-8") as f:
        json.dump(folds_dict, f, indent=2)

    return holdout_dict, folds_dict


class InnerCVCache:
    """
    Read-only cache for inner CV preprocessed and feature-selected arrays.
    Prevents redundant computation across Optuna trials.
    """
    def __init__(self):
        self._cache = {}

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        return self._cache.get(key)

    def put(self, key: str, data: Dict[str, np.ndarray]):
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                v.flags.writeable = False
        self._cache[key] = data


def fit_and_evaluate_pipeline(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             X_test: pd.DataFrame,
                             y_test: pd.Series,
                             numerical_cols: List[str],
                             categorical_cols: List[str],
                             model_class,
                             model_kwargs: Dict[str, Any],
                             imbalance_strategy: str = "none",
                             afo_enabled: bool = True,
                             calibration_enabled: bool = True,
                             threshold_enabled: bool = True,
                             n_inner_splits: int = 3,
                             preprocessor: Optional[ClinicalPreprocessor] = None,
                             afo: Optional[DynamicFeatureOptimizer] = None,
                             random_state: int = 42) -> Dict[str, Any]:
    """
    Executes complete, leakage-free pipeline fitting on (X_train, y_train)
    and evaluation on unseen (X_test, y_test).
    """
    # 1. Preprocessing fitted strictly on X_train
    if preprocessor is None:
        preprocessor = ClinicalPreprocessor(
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            random_state=random_state
        )
        preprocessor.fit(X_train, y_train)
    X_tr_proc = preprocessor.transform(X_train)
    X_te_proc = preprocessor.transform(X_test)
    feat_names = preprocessor.get_feature_names_out()

    # 2. Dynamic Feature Optimization (AFO) fitted strictly on X_train
    if afo_enabled:
        if afo is None:
            afo = DynamicFeatureOptimizer(random_state=random_state)
            afo.fit(X_tr_proc, y_train.values, feature_names=list(feat_names))
        X_tr_sel = afo.transform(X_tr_proc)
        X_te_sel = afo.transform(X_te_proc)
    else:
        afo = None
        X_tr_sel = X_tr_proc
        X_te_sel = X_te_proc

    # 3. Fit final model on outer training data with chosen imbalance handling
    X_tr_res, y_tr_res, class_weights = apply_imbalance_strategy(
        imbalance_strategy, X_tr_sel, y_train.values, random_state=random_state
    )

    final_model = model_class(**model_kwargs)
    final_model.fit(X_tr_res, y_tr_res)

    # 4. Out-of-fold validation predictions for un-resampled calibration and threshold
    inner_val_y = y_train.values.copy()
    if hasattr(final_model, "oof_probs_"):
        inner_val_probs = final_model.oof_probs_
        inner_thresholds = [optimize_youden_threshold(inner_val_y, inner_val_probs)["optimal_threshold"]]
    else:
        # Fast inner split for standalone baselines
        inner_cv = StratifiedKFold(n_splits=min(3, n_inner_splits), shuffle=True, random_state=random_state)
        inner_val_probs = np.zeros(len(X_tr_sel))
        inner_thresholds = []
        for in_tr_idx, in_val_idx in inner_cv.split(X_tr_sel, inner_val_y):
            X_in_tr, y_in_tr = X_tr_sel[in_tr_idx], inner_val_y[in_tr_idx]
            X_in_val, y_in_val = X_tr_sel[in_val_idx], inner_val_y[in_val_idx]
            X_in_tr_res, y_in_tr_res, _ = apply_imbalance_strategy(
                imbalance_strategy, X_in_tr, y_in_tr, random_state=random_state
            )
            in_m = model_class(**model_kwargs)
            in_m.fit(X_in_tr_res, y_in_tr_res)
            p_val = in_m.predict_proba(X_in_val)[:, 1]
            inner_val_probs[in_val_idx] = p_val
            if threshold_enabled:
                inner_thresholds.append(optimize_youden_threshold(y_in_val, p_val)["optimal_threshold"])

    # 5. Calibration: fit calibrator on un-resampled inner validation predictions
    calibrator = None
    if calibration_enabled:
        calibrator = choose_best_calibrator(inner_val_probs, inner_val_y)
        calibrated_inner_val = calibrator.calibrate(inner_val_probs)
    else:
        calibrated_inner_val = inner_val_probs

    # 6. Threshold selection: median Youden's J threshold across inner validation predictions
    decision_threshold = 0.5
    if threshold_enabled:
        if inner_thresholds:
            decision_threshold = aggregate_fold_thresholds(inner_thresholds)
        else:
            decision_threshold = optimize_youden_threshold(inner_val_y, calibrated_inner_val)["optimal_threshold"]

    # 7. Evaluate on outer test set (completely unseen)
    raw_test_probs = final_model.predict_proba(X_te_sel)[:, 1]
    if calibrator is not None:
        calibrated_test_probs = calibrator.calibrate(raw_test_probs)
    else:
        calibrated_test_probs = raw_test_probs

    test_preds = (calibrated_test_probs >= decision_threshold).astype(int)

    return {
        "preprocessor": preprocessor,
        "afo": afo,
        "model": final_model,
        "calibrator": calibrator,
        "threshold": decision_threshold,
        "y_true": y_test.values,
        "y_pred": test_preds,
        "y_prob": calibrated_test_probs,
        "raw_prob": raw_test_probs,
        "selected_features": afo.selected_feature_names if afo else feat_names,
        "imbalance_strategy": imbalance_strategy
    }
