"""
Ensemble Weights Audit Module for DS-HTE Framework.
Follows Section 0.1:
- Inspects per-fold ensemble weights from results/fold_metrics/ensemble_weights.parquet
- Saves audit copy to results/audit/ensemble_weights.parquet
- Flags any component with mean weight > 0.9 (indicating ensemble collapse)
- Documents whether ensemble reduces to Logistic Regression or single model
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List


def audit_ensemble_weights(weights_parquet_path: str = "results/fold_metrics/ensemble_weights.parquet",
                           audit_dir: str = "results/audit") -> Dict[str, Any]:
    if not os.path.exists(weights_parquet_path):
        raise FileNotFoundError(f"Ensemble weights parquet not found at {weights_parquet_path}")

    os.makedirs(audit_dir, exist_ok=True)
    df = pd.read_parquet(weights_parquet_path)

    # Save to audit path as specified in 0.1
    audit_parquet_path = os.path.join(audit_dir, "ensemble_weights.parquet")
    df.to_parquet(audit_parquet_path, index=False)

    cohorts = df["cohort"].unique()
    summary = {}
    collapsed_components = []

    for cohort in cohorts:
        sub = df[df["cohort"] == cohort]
        comp_means = sub.groupby("model_component")["weight"].agg(["mean", "std", "min", "max"]).reset_index()
        summary[cohort] = comp_means.to_dict(orient="records")

        for _, row in comp_means.iterrows():
            if row["mean"] > 0.90:
                collapsed_components.append({
                    "cohort": cohort,
                    "component": row["model_component"],
                    "mean_weight": float(row["mean"]),
                    "std": float(row["std"]),
                    "flag": "COLLAPSE_DETECTED"
                })

    # Render Markdown audit report
    md_path = os.path.join(audit_dir, "ensemble_collapse_audit.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Ensemble Weights & Model Diversity Audit (Section 0.1)\n\n")
        f.write("> **Audit Rule**: Any component with mean weight > 0.90 across outer folds is flagged as ensemble collapse.\n\n")

        if collapsed_components:
            f.write("### [!] Flagged Collapsed Components (Mean Weight > 0.90)\n\n")
            f.write("| Cohort | Component | Mean Weight | Std Dev | Finding |\n")
            f.write("| :--- | :--- | ---: | ---: | :--- |\n")
            for c in collapsed_components:
                f.write(f"| {c['cohort'].upper()} | `{c['component']}` | {c['mean_weight']:.4f} | {c['std']:.4f} | Reduces primarily to {c['component'].upper()} |\n")
            f.write("\n")
        else:
            f.write("### [OK] No Components Exceed 0.90 Mean Weight Floor Threshold\n\n")

        f.write("### Full Component Weight Distribution Across 50 Folds\n\n")
        f.write("| Cohort | Component | Mean Weight | Std Dev | Min Weight | Max Weight |\n")
        f.write("| :--- | :--- | ---: | ---: | ---: | ---: |\n")
        for cohort, comps in summary.items():
            for row in comps:
                f.write(f"| {cohort.upper()} | `{row['model_component']}` | {row['mean']:.4f} | {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |\n")

    return {
        "summary": summary,
        "collapsed_components": collapsed_components,
        "audit_parquet_path": audit_parquet_path,
        "audit_md_path": md_path
    }


if __name__ == "__main__":
    res = audit_ensemble_weights()
    print("Ensemble audit completed:")
    print("Collapsed components:", res["collapsed_components"])
