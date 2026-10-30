"""
Data profiling and duplicate/near-duplicate audit module.
Generates data/processed/<cohort>_profile.json and results/audit/duplicates_<cohort>.md.
"""

import os
import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from src.data.load import load_cohort


def profile_and_audit(cohort: str, raw_dir: str = "data/raw",
                      processed_dir: str = "data/processed",
                      audit_dir: str = "results/audit") -> Dict[str, Any]:
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(audit_dir, exist_ok=True)
    
    X, y, num_cols, cat_cols = load_cohort(cohort, raw_dir)
    n_samples, n_features = X.shape
    
    # Class balance
    counts = y.value_counts().to_dict()
    proportions = (y.value_counts(normalize=True)).to_dict()
    class_balance = {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "proportions": {str(k): float(v) for k, v in proportions.items()},
        "imbalance_ratio": float(max(counts.values()) / min(counts.values())) if min(counts.values()) > 0 else 1.0
    }
    
    # Missing rates
    missing_counts = X.isnull().sum().to_dict()
    missing_rates = (X.isnull().mean()).to_dict()
    overall_missing_rate = float(X.isnull().values.mean())
    
    profile = {
        "cohort": cohort,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "class_balance": class_balance,
        "overall_missing_rate": overall_missing_rate,
        "feature_missing_rates": {k: float(v) for k, v in missing_rates.items()}
    }
    
    # Write profile JSON
    profile_path = os.path.join(processed_dir, f"{cohort}_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
        
    # Duplicate & near-duplicate scan
    # Exact duplicates in features
    df_full = X.copy()
    df_full['__target__'] = y
    exact_dup_mask = df_full.duplicated(subset=list(X.columns), keep=False)
    n_exact_dups = int(exact_dup_mask.sum())
    
    # Near duplicates: check rows with high similarity after simple median fill
    # Use normalized Manhattan distance
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    
    X_num = X.select_dtypes(include=[np.number])
    near_dup_pairs = []
    if not X_num.empty:
        imp = SimpleImputer(strategy='median')
        X_imp = imp.fit_transform(X_num)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imp)
        
        # Check pairwise distances for near matches (distance < 0.05 per feature)
        # For N < 1000, pairwise is fast
        from scipy.spatial.distance import pdist, squareform
        dists = squareform(pdist(X_scaled, metric='cityblock')) / X_scaled.shape[1]
        np.fill_diagonal(dists, np.inf)
        close_indices = np.where(dists < 0.05)
        for i, j in zip(*close_indices):
            if i < j:
                near_dup_pairs.append({
                    "row_i": int(i),
                    "row_j": int(j),
                    "dist": float(dists[i, j]),
                    "y_i": int(y.iloc[i]),
                    "y_j": int(y.iloc[j])
                })
                
    # Build cluster assignment if duplicates exist
    clusters = np.arange(n_samples)
    if n_exact_dups > 0:
        dup_groups = df_full.groupby(list(X.columns)).groups
        for grp_indices in dup_groups.values():
            if len(grp_indices) > 1:
                first_idx = grp_indices[0]
                for idx in grp_indices:
                    clusters[idx] = first_idx
                    
    profile["has_duplicates"] = bool(n_exact_dups > 0 or len(near_dup_pairs) > 0)
    profile["exact_duplicate_rows"] = n_exact_dups
    profile["near_duplicate_pairs"] = len(near_dup_pairs)
    profile["n_unique_clusters"] = int(len(np.unique(clusters)))
    
    # Write audit markdown
    audit_path = os.path.join(audit_dir, f"duplicates_{cohort}.md")
    with open(audit_path, "w", encoding="utf-8") as f:
        f.write(f"# Data & Duplicate Audit: {cohort.upper()}\n\n")
        f.write(f"- **Total Samples**: {n_samples}\n")
        f.write(f"- **Total Features**: {n_features} ({len(num_cols)} numerical, {len(cat_cols)} categorical)\n")
        f.write(f"- **Class Distribution**: {class_balance['counts']} (Ratio: {class_balance['imbalance_ratio']:.2f})\n")
        f.write(f"- **Overall Missing Rate**: {overall_missing_rate * 100:.2f}%\n")
        f.write(f"- **Exact Duplicate Rows (Feature-wise)**: {n_exact_dups}\n")
        f.write(f"- **Near-Duplicate Pairs (Normalized dist < 0.05)**: {len(near_dup_pairs)}\n\n")
        
        if n_exact_dups > 0:
            f.write("### Exact Duplicate Groups\n")
            f.write("Duplicate rows detected. Grouped stratified splitting is enforced so entire duplicate clusters stay within the same fold to prevent test leakage.\n\n")
        else:
            f.write("No exact duplicate feature rows detected.\n\n")
            
        if near_dup_pairs:
            f.write("### Near-Duplicate Pairs Sample\n")
            f.write("| Row 1 | Row 2 | Avg Norm Distance | Target 1 | Target 2 |\n")
            f.write("|---|---|---|---|---|\n")
            for p in near_dup_pairs[:10]:
                f.write(f"| {p['row_i']} | {p['row_j']} | {p['dist']:.4f} | {p['y_i']} | {p['y_j']} |\n")
            f.write("\n")
            
    return profile


def run_all_profiles(raw_dir: str = "data/raw", processed_dir: str = "data/processed",
                     audit_dir: str = "results/audit"):
    cohorts = ['pima', 'cleveland', 'ckd', 'ilpd', 'wbcd']
    results = {}
    for c in cohorts:
        results[c] = profile_and_audit(c, raw_dir, processed_dir, audit_dir)
        print(f"Profiled {c.upper()}: N={results[c]['n_samples']}, Features={results[c]['n_features']}, Missing={results[c]['overall_missing_rate']*100:.1f}%, Dups={results[c]['exact_duplicate_rows']}")
    return results


if __name__ == "__main__":
    run_all_profiles()
