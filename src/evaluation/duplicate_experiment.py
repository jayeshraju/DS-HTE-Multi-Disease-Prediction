"""
Controlled Experiment for Item 0.9: ILPD Duplicate Record Leakage Audit.
Runs a 50-fold evaluation on ILPD comparing:
1. Ungrouped StratifiedKFold (standard, potential duplicate leakage across train/test)
2. Grouped StratifiedGroupKFold (duplicate records clustered and kept strictly within single fold)

Empirically measures and logs the exact delta to results/audit/ilpd_duplicate_leakage_experiment.json,
providing verified experimental data rather than untested speculation.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss

from src.data.load import load_ilpd


def run_ilpd_duplicate_experiment(n_splits: int = 5, n_repeats: int = 10, random_state: int = 42):
    X, y, num_cols, cat_cols = load_ilpd()
    
    # Identify duplicate feature rows to form cluster groups
    # Group ID: assign identical rows the same group id; unique rows get unique group id
    features_df = X.copy()
    duplicate_groups = features_df.groupby(list(features_df.columns), dropna=False).ngroup()
    n_total = len(X)
    n_unique_groups = len(duplicate_groups.unique())
    n_duplicates = n_total - n_unique_groups
    
    def evaluate_cv(splitter_generator):
        accs = []
        aucs = []
        b_accs = []
        f1s = []
        briers = []
        
        for train_idx, test_idx in splitter_generator():
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            # Simple, standard pipeline: Median Impute + StandardScale + LogisticRegression
            pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(max_iter=500, random_state=random_state))
            ])
            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)
            y_prob = pipe.predict_proba(X_te)[:, 1]
            
            accs.append(accuracy_score(y_te, y_pred))
            aucs.append(roc_auc_score(y_te, y_prob))
            b_accs.append(balanced_accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, zero_division=0))
            briers.append(brier_score_loss(y_te, y_prob))
            
        return {
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "balanced_acc_mean": float(np.mean(b_accs)),
            "balanced_acc_std": float(np.std(b_accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "brier_mean": float(np.mean(briers)),
            "brier_std": float(np.std(briers))
        }

    # 1. Ungrouped (RepeatedStratifiedKFold)
    def ungrouped_gen():
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        for tr, te in rskf.split(X, y):
            yield tr, te

    ungrouped_res = evaluate_cv(ungrouped_gen)

    # 2. Grouped (StratifiedGroupKFold across repeated random splits)
    def grouped_gen():
        for rep in range(n_repeats):
            # To vary splits across repeats, shuffle group mapping or use stratified group k-fold with shuffle
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state + rep * 100)
            for tr, te in sgkf.split(X, y, groups=duplicate_groups):
                yield tr, te

    grouped_res = evaluate_cv(grouped_gen)

    # Delta: ungrouped minus grouped (positive indicates optimistic leakage in ungrouped)
    delta = {
        "accuracy_delta": round(ungrouped_res["accuracy_mean"] - grouped_res["accuracy_mean"], 4),
        "roc_auc_delta": round(ungrouped_res["roc_auc_mean"] - grouped_res["roc_auc_mean"], 4),
        "balanced_acc_delta": round(ungrouped_res["balanced_acc_mean"] - grouped_res["balanced_acc_mean"], 4),
        "f1_delta": round(ungrouped_res["f1_mean"] - grouped_res["f1_mean"], 4)
    }

    result = {
        "dataset": "ILPD",
        "total_records": n_total,
        "unique_feature_clusters": n_unique_groups,
        "duplicate_records_count": n_duplicates,
        "n_folds_total": n_splits * n_repeats,
        "ungrouped_evaluation": ungrouped_res,
        "grouped_evaluation": grouped_res,
        "leakage_delta_ungrouped_minus_grouped": delta,
        "conclusion": "Controlled experiment completed. Empirical delta logged rather than untested speculation."
    }

    os.makedirs("results/audit", exist_ok=True)
    out_path = "results/audit/ilpd_duplicate_leakage_experiment.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


if __name__ == "__main__":
    res = run_ilpd_duplicate_experiment()
    print("ILPD Duplicate Experiment Result:")
    print(json.dumps(res, indent=2))
