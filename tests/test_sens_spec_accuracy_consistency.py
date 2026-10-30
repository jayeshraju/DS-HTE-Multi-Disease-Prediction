"""
Unit test verifying mathematical consistency between sensitivity, specificity, prevalence, and accuracy.
Follows Gate 0 Check 5:
- Recomputes accuracy from sensitivity, specificity, and prevalence:
    Accuracy_calc = Sensitivity * Prevalence + Specificity * (1 - Prevalence)
- Asserts reported accuracy matches Accuracy_calc within 0.5 percentage points.
- Asserts balanced accuracy == (Sensitivity + Specificity) / 2 within 0.005.
"""

import os
import pandas as pd
import numpy as np
import pytest

from src.data.load import load_cohort

RAW_DIR = "data/raw"
TABLE_2_PATH = "results/tables/table_2_dshte_metrics.csv"
OOF_PATH = "results/predictions/oof_predictions.parquet"


def test_sens_spec_accuracy_consistency():
    assert os.path.exists(TABLE_2_PATH), f"Missing {TABLE_2_PATH}"
    df_t2 = pd.read_csv(TABLE_2_PATH)
    
    # Calculate prevalence for each cohort from OOF ground truth or raw data
    prevalences = {}
    if os.path.exists(OOF_PATH):
        df_oof = pd.read_parquet(OOF_PATH)
        for cohort in ["pima", "cleveland", "ckd", "ilpd", "wbcd"]:
            sub = df_oof[(df_oof["cohort"] == cohort) & (df_oof["model"] == "DS-HTE")]
            prevalences[cohort] = float(np.mean(sub["y_true"]))
    else:
        for cohort in ["pima", "cleveland", "ckd", "ilpd", "wbcd"]:
            _, y, _, _ = load_cohort(cohort, RAW_DIR)
            prevalences[cohort] = float(np.mean(y))
            
    for _, row in df_t2.iterrows():
        dataset_name = row["Dataset"]
        cohort_key = dataset_name.lower()
        prev = prevalences.get(cohort_key)
        assert prev is not None, f"Prevalence unknown for {dataset_name}"
        
        # Parse mean values (format: "0.xxx +- 0.yyy [0.zzz, 0.www]")
        def parse_mean(cell_text: str) -> float:
            return float(cell_text.split("+-")[0].strip())
            
        acc_rep = parse_mean(row["ACCURACY"])
        sens_rep = parse_mean(row["SENSITIVITY"])
        spec_rep = parse_mean(row["SPECIFICITY"])
        bal_rep = parse_mean(row["BALANCED_ACCURACY"])
        
        # Mathematical invariant 1: Accuracy = Sens * Prev + Spec * (1 - Prev)
        acc_calc = (sens_rep * prev) + (spec_rep * (1.0 - prev))
        delta_acc = abs(acc_rep - acc_calc) * 100.0  # percentage points
        assert delta_acc <= 0.5, (
            f"[{dataset_name}] Accuracy inconsistency: reported {acc_rep*100:.2f}%, "
            f"calculated from sens/spec/prevalence {acc_calc*100:.2f}% (delta={delta_acc:.2f} > 0.5 pt)"
        )
        
        # Mathematical invariant 2: Balanced Accuracy = (Sens + Spec) / 2
        bal_calc = (sens_rep + spec_rep) / 2.0
        delta_bal = abs(bal_rep - bal_calc)
        assert delta_bal <= 0.005, (
            f"[{dataset_name}] Balanced accuracy inconsistency: reported {bal_rep:.4f}, "
            f"calculated {bal_calc:.4f} (delta={delta_bal:.4f} > 0.005)"
        )
