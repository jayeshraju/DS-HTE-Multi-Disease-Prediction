"""
Comprehensive Model Evaluation Script.
Loads every saved pipeline, re-ingests raw data, applies the saved
preprocessor + AFO selection, and reports accuracy + full metrics for
both XGBoost Baseline and the Hybrid HDA-GB model on a sealed 20 % test split.
"""

import sys
import joblib
import warnings
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, balanced_accuracy_score
)

warnings.filterwarnings("ignore")

sys.path.insert(0, r"d:\study_files\PROJECTS\MULTI_DIS_PREDICT")

# Inject classes into __main__ namespace to avoid pickle loading errors for legacy serialized files
import __main__
from src.models.hybrid import HDAGBClassifier, CalibratedThresholdClassifierWrapper, BetaCalibrator
__main__.BetaCalibrator = BetaCalibrator
__main__.CalibratedThresholdClassifierWrapper = CalibratedThresholdClassifierWrapper
__main__.HDAGBClassifier = HDAGBClassifier

from src.config import DATASET_CONFIGS, MODELS_DIR, RANDOM_SEED
from src.data.ingest import ingest_datasets

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Map raw target column to binary labels (identical logic as main.py)
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_target(y_series: pd.Series, disease_key: str) -> pd.Series:
    y = y_series.copy()
    if disease_key == "diabetes":
        return y.astype(int)
    elif disease_key == "heart":
        if y.dtype == object:
            y = pd.to_numeric(y.astype(str).str.replace("?", "0", regex=False),
                              errors="coerce").fillna(0)
        return (y > 0).astype(int)
    elif disease_key == "kidney":
        y_str = y.astype(str).str.strip().str.lower()
        return y_str.map(lambda v: 0 if "not" in v else (1 if "ckd" in v else 0)).astype(int)
    elif disease_key == "liver":
        if y.dtype == object:
            y = pd.to_numeric(y, errors="coerce").fillna(2)
        return (y == 1).astype(int)
    elif disease_key == "cancer":
        y_str = y.astype(str).str.strip().str.upper()
        return y_str.map({"M": 1, "B": 0}).fillna(0).astype(int)
    return y.astype(int)


def evaluate_all():
    print("=" * 90)
    print("   COMPREHENSIVE MODEL ACCURACY EVALUATION - ALL 5 DISEASE PIPELINES")
    print("=" * 90)

    # ------------------------------------------------------------------
    # 1. Load the raw datasets (use cached CSVs so no rebuild is needed)
    # ------------------------------------------------------------------
    print("\n[1] Loading raw datasets (from cache) ...")
    datasets = ingest_datasets(force_rebuild=False)

    summary_rows = []

    for key, config in DATASET_CONFIGS.items():
        print(f"\n{'-'*70}")
        print(f"  DISEASE  :  {config['name'].upper()}")
        print(f"{'-'*70}")

        # ── Load pipeline ──────────────────────────────────────────────
        pipeline_path = MODELS_DIR / f"{key}_pipeline.joblib"
        if not pipeline_path.exists():
            print(f"  [!] No saved pipeline found at {pipeline_path}. Skipping.")
            continue
        pipeline = joblib.load(pipeline_path)

        preprocessor      = pipeline["preprocessor"]
        selected_indices  = pipeline["selected_feature_indices"]
        selected_names    = pipeline["selected_feature_names"]
        hybrid_model      = pipeline["model"]
        
        # ── Rebuild identical split to fit baseline if needed ──────────
        df = datasets[key]
        feature_cols = config["numerical_cols"] + config["categorical_cols"]
        X_df = df[feature_cols]
        y_bin = preprocess_target(df[config["target"]], key)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_df, y_bin, test_size=0.2, stratify=y_bin, random_state=RANDOM_SEED
        )

        baseline_model    = pipeline.get("baseline", None)
        if baseline_model is None:
            print("  [*] Baseline model not found in serialized pipeline. Fitting baseline on-the-fly...")
            try:
                import xgboost as xgb
                baseline_model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.03,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                baseline_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
            
            X_train_proc = preprocessor.transform(X_train_raw)
            X_train_afo  = X_train_proc[:, selected_indices]
            baseline_model.fit(X_train_afo, y_train)

        # ── Rebuild identical split ────────────────────────────────────
        df = datasets[key]
        feature_cols = config["numerical_cols"] + config["categorical_cols"]
        X_df = df[feature_cols]
        y_bin = preprocess_target(df[config["target"]], key)

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_df, y_bin, test_size=0.2, stratify=y_bin, random_state=RANDOM_SEED
        )

        print(f"  Total samples   : {len(X_df)}   (Train={len(X_train_raw)}, Test={len(X_test_raw)})")
        print(f"  Features (raw)  : {X_df.shape[1]}   ->  AFO selected: {len(selected_names)}")
        print(f"  Selected feats  : {selected_names}")

        # ── Apply saved preprocessor + AFO indexing ────────────────────
        X_test_proc = preprocessor.transform(X_test_raw)
        X_test_afo  = X_test_proc[:, selected_indices]

        # ── Predictions ────────────────────────────────────────────────
        base_pred    = baseline_model.predict(X_test_afo)
        base_proba   = baseline_model.predict_proba(X_test_afo)[:, 1]
        hybrid_pred  = hybrid_model.predict(X_test_afo)
        hybrid_proba = hybrid_model.predict_proba(X_test_afo)[:, 1]

        # ── Per-model metrics ──────────────────────────────────────────
        def metrics(y_true, y_pred, y_proba):
            roc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else float("nan")
            return {
                "Accuracy"          : round(accuracy_score(y_true, y_pred) * 100, 2),
                "Balanced Acc."     : round(balanced_accuracy_score(y_true, y_pred) * 100, 2),
                "F1-Score"          : round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
                "Precision"         : round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
                "Recall/Sensitivity": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
                "ROC-AUC"           : round(roc * 100, 2),
                "MCC"               : round(matthews_corrcoef(y_true, y_pred), 4),
            }

        bm = metrics(y_test, base_pred,   base_proba)
        hm = metrics(y_test, hybrid_pred, hybrid_proba)

        # ── Print per-disease table ────────────────────────────────────
        rows = [[k, f"{bm[k]} %", f"{hm[k]} %"] for k in bm if k != "MCC"]
        rows.append(["MCC", str(bm["MCC"]), str(hm["MCC"])])
        print()
        print(tabulate(
            rows,
            headers=["Metric", "XGBoost Baseline", "HDA-GB Hybrid (Proposed)"],
            tablefmt="psql"
        ))

        summary_rows.append({
            "Disease"           : config["name"],
            "Test Samples"      : len(X_test_raw),
            "Baseline Acc. (%)" : bm["Accuracy"],
            "Hybrid Acc. (%)"   : hm["Accuracy"],
            "Baseline F1 (%)"   : bm["F1-Score"],
            "Hybrid F1 (%)"     : hm["F1-Score"],
            "Baseline AUC (%)"  : bm["ROC-AUC"],
            "Hybrid AUC (%)"    : hm["ROC-AUC"],
            "Hybrid MCC"        : hm["MCC"],
        })

    # ------------------------------------------------------------------
    # 2. Final cross-disease summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("   FINAL SUMMARY - ACCURACY & KEY METRICS ACROSS ALL DISEASES")
    print("=" * 90)
    summary_df = pd.DataFrame(summary_rows)
    print(tabulate(summary_df, headers="keys", showindex=False, tablefmt="psql"))

    # Average improvement
    avg_base  = summary_df["Baseline Acc. (%)"].mean()
    avg_hybrid= summary_df["Hybrid Acc. (%)"].mean()
    avg_diff  = avg_hybrid - avg_base
    print(f"\n  Mean Accuracy - XGBoost Baseline : {avg_base:.2f} %")
    print(f"  Mean Accuracy - HDA-GB Hybrid    : {avg_hybrid:.2f} %")
    sign = "+" if avg_diff >= 0 else ""
    print(f"  Improvement (Hybrid vs Baseline) : {sign}{avg_diff:.2f} %")
    print("\n[OK] Evaluation complete.\n")


if __name__ == "__main__":
    evaluate_all()
