"""
Dataset Quality Audit Module.
Audits the five target clinical datasets for missingness, duplicates, constant features,
outliers (via Isolation Forest), and class imbalance, producing a publication-grade report.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest

from src.config import DATASET_CONFIGS, DOCS_DIR, RANDOM_SEED

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

def run_audit():
    print("=" * 80)
    print("   CLINICAL DATASET AUDIT & QUALITY METRICS GENERATION")
    print("=" * 80)

    audit_summary = []
    detailed_json = {}

    for key, config in DATASET_CONFIGS.items():
        raw_path = Path(config["raw_file"])
        if not raw_path.exists():
            print(f"[-] Raw file not found for {config['name']}: {raw_path}")
            continue

        print(f"\nAuditing: {config['name']}...")
        df = pd.read_csv(raw_path)
        
        # Target variable and prevalence
        y = preprocess_target(df[config["target"]], key)
        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        total_samples = len(df)
        pos_prevalence = float(pos_count / total_samples)

        # Missingness
        total_cells = df.size
        # Since we mapped missing strings like '?' to NaN during ingestion, null count is simple:
        null_count = int(df.isnull().sum().sum())
        null_percentage = float(null_count / total_cells * 100)

        # Duplicates
        duplicate_count = int(df.duplicated().sum())

        # Constant features
        constant_features = []
        for col in df.columns:
            if df[col].nunique(dropna=False) <= 1:
                constant_features.append(col)

        # Outlier Detection using Isolation Forest (on numerical columns)
        num_cols = config["numerical_cols"]
        df_num = df[num_cols].copy()
        
        # Temporary median imputation just to run Isolation Forest
        for col in df_num.columns:
            if df_num[col].isnull().any():
                df_num[col] = df_num[col].fillna(df_num[col].median())
        
        if len(df_num) > 10 and len(num_cols) > 0:
            iforest = IsolationForest(contamination=0.05, random_state=RANDOM_SEED)
            preds = iforest.fit_predict(df_num)
            outlier_count = int(np.sum(preds == -1))
        else:
            outlier_count = 0

        # Features details
        feature_types = {}
        for col in df.columns:
            feature_types[col] = str(df[col].dtype)

        # Save to summary CSV list
        audit_summary.append({
            "Dataset": config["name"],
            "Samples": total_samples,
            "Features": len(df.columns) - 1, # Excluding target
            "Missing Cells": null_count,
            "Missingness (%)": round(null_percentage, 2),
            "Duplicate Rows": duplicate_count,
            "Constant Features": len(constant_features),
            "Outlier Count": outlier_count,
            "Positive Count": pos_count,
            "Negative Count": neg_count,
            "Pos Prevalence (%)": round(pos_prevalence * 100, 2)
        })

        # Save to detailed JSON structure
        detailed_json[key] = {
            "dataset_name": config["name"],
            "shape": [total_samples, len(df.columns)],
            "missingness": {
                "total_missing_cells": null_count,
                "percentage_missing": null_percentage,
                "missing_by_feature": df.isnull().sum().to_dict()
            },
            "duplicates": duplicate_count,
            "constant_features": constant_features,
            "outliers": {
                "method": "IsolationForest (contamination=0.05)",
                "outlier_count": outlier_count
            },
            "class_distribution": {
                "positive_class_count": pos_count,
                "negative_class_count": neg_count,
                "positive_prevalence": pos_prevalence
            },
            "feature_metadata": feature_types
        }

    # Write outputs
    summary_df = pd.DataFrame(audit_summary)
    csv_path = DOCS_DIR / "dataset_audit_report.csv"
    json_path = DOCS_DIR / "dataset_audit_report.json"

    summary_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(detailed_json, f, indent=4)

    print("\n[+] Audit Complete!")
    print(f" -> Saved summary CSV: {csv_path}")
    print(f" -> Saved detailed JSON: {json_path}")
    print("\nSummary Table:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    run_audit()
