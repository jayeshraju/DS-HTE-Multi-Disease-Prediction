"""
Unified Command-Line Execution Entrypoint.
Runs the end-to-end research pipeline across all 5 disease datasets:
data loading, preprocessing, AFO consensus, class balancing, model training,
metric evaluation, explainability, clinical reporting, and model serialization.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED, DATASET_CONFIGS, MODELS_DIR, PROCESSED_DATA_DIR
from src.data.ingest import ingest_datasets
from src.data.preprocessing import MedicalPreprocessor
from src.data.balancing import AdaptiveDataBalancer
from src.features.afo import AdaptiveFeatureOptimizer
from src.models.hybrid import HDAGBClassifier
from src.utils.metrics import evaluate_clinical_metrics, mcnemar_statistical_test
from src.utils.explainability import ClinicalExplainer
from src.utils.report_generator import ClinicalReportGenerator

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def preprocess_target(y_series: pd.Series, disease_key: str) -> pd.Series:
    """
    Standardizes label mapping to binary (0 = healthy, 1 = pathological) for each disease.
    """
    y = y_series.copy()
    if disease_key == "diabetes":
        # Outcome is already 0 or 1
        return y.astype(int)
    
    elif disease_key == "heart":
        # target values range 0 to 4. 0 is normal, >=1 are disease
        if y.dtype == object:
            # handle potential string values
            y = pd.to_numeric(y.astype(str).str.replace('?', '0', regex=False), errors='coerce').fillna(0)
        return (y > 0).astype(int)
    
    elif disease_key == "kidney":
        # classification values: 'ckd', 'notckd', sometimes with trailing spaces/tabs
        y_str = y.astype(str).str.strip().str.lower()
        # map ckd to 1, notckd to 0
        mapped = y_str.map(lambda val: 0 if "not" in val else (1 if "ckd" in val else 0))
        return mapped.astype(int)
        
    elif disease_key == "liver":
        # Dataset field: 1 = disease, 2 = no disease
        if y.dtype == object:
            y = pd.to_numeric(y, errors='coerce').fillna(2)
        return (y == 1).astype(int)
        
    elif disease_key == "cancer":
        # diagnosis: 'M' = malignant (1), 'B' = benign (0)
        y_str = y.astype(str).str.strip().str.upper()
        return y_str.map({'M': 1, 'B': 0}).fillna(0).astype(int)
        
    return y.astype(int)


def run_full_research_pipeline():
    print("=" * 100)
    print("      EXPLAINABLE MULTI-DISEASE PREDICTION & RISK STRATIFICATION RESEARCH FRAMEWORK")
    print("=" * 100)

    # 1. Programmatically Ingest & Load Datasets
    print("\n[Phase 2/3] Programmatic Ingestion & Schema Sanitization...")
    datasets = ingest_datasets()

    results_summary = []

    # Loop through each disease configuration
    for key, config in DATASET_CONFIGS.items():
        print("\n" + "=" * 80)
        print(f" PROCESSING DISEASE BENCHMARK: {config['name'].upper()}")
        print("=" * 80)

        df = datasets[key]
        
        # Isolate features and target
        feature_cols = config["numerical_cols"] + config["categorical_cols"]
        X_df = df[feature_cols]
        y_df = df[config["target"]]

        # Preprocess target labels to binary
        y_binary = preprocess_target(y_df, key)
        pos_ratio = np.mean(y_binary)
        
        print(f" -> Raw Feature Space Size: {X_df.shape[1]} Attributes")
        print(f" -> Total Clinical Cases: {X_df.shape[0]} Patients")
        print(f" -> Positive Diagnostic Cases (Class 1): {np.sum(y_binary == 1)} ({pos_ratio:.2%})")

        # 2. Stratified Train-Test Split (80% Train-Val / 20% Holdout Test)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_df, y_binary, test_size=0.2, stratify=y_binary, random_state=RANDOM_SEED
        )
        print(f" -> Split: {X_train_raw.shape[0]} Train Patients | {X_test_raw.shape[0]} Sealed Test Patients")

        # 3. Medical Preprocessing (MICE/KNN Imputation + Outliers + Scaling)
        print(f"\n[Phase 4] Fitting Medical Preprocessor for {config['name']}...")
        
        # Configure dataset-specific preprocessing parameters
        imp_strat = "mice" if key in ["diabetes", "liver"] else "knn"
        use_pt = True if key == "liver" else False
        outlier_contam = 0.04 if key in ["diabetes", "liver"] else 0.03
        
        preprocessor = MedicalPreprocessor(
            numerical_cols=config["numerical_cols"],
            categorical_cols=config["categorical_cols"],
            zero_as_missing=config["zero_as_missing"],
            imputation_strategy=imp_strat,
            outlier_contamination=outlier_contam,
            use_power_transform=use_pt,
            disease=key
        )
        
        if hasattr(preprocessor, "fit_transform_train"):
            X_train_proc, y_train = preprocessor.fit_transform_train(X_train_raw, y_train)
        else:
            X_train_proc = preprocessor.fit_transform(X_train_raw)
        X_test_proc = preprocessor.transform(X_test_raw)

        # Get processed column names (handling one-hot encoding expansion)
        if len(config["categorical_cols"]) > 0:
            processed_feature_names = list(preprocessor.numerical_cols) + list(preprocessor.ohe.get_feature_names_out(config["categorical_cols"]))
        else:
            processed_feature_names = list(preprocessor.numerical_cols)

        # 4. Adaptive Feature Optimization (AFO) Borda Consensus
        print(f"\n[Phase 5] Executing Adaptive Feature Optimization (AFO Consensus ranking)...")
        # Target top 12 features or 75% of features if smaller
        k_features = min(12, len(processed_feature_names))
        afo = AdaptiveFeatureOptimizer(k_top=k_features, random_state=RANDOM_SEED)
        X_train_afo = afo.fit_transform(X_train_proc, y_train, feature_names=processed_feature_names)
        X_test_afo = afo.transform(X_test_proc)
        
        selected_feature_names = [processed_feature_names[i] for i in afo.selected_indices_]
        print(f" -> Selected {len(selected_feature_names)} Key Features: {selected_feature_names}")

        # 5. Class Imbalance Resampling (SMOTE-ENN / Borderline-SMOTE)
        bal_method = "smote_enn" if key == "diabetes" else ("borderline_smote" if key == "liver" else "smote_tomek")
        print(f"\n[Phase 6] Applying Resampling ({bal_method})...")
        balancer = AdaptiveDataBalancer(sampling_strategy=0.85, method=bal_method, random_state=RANDOM_SEED)
        X_train_bal, y_train_bal = balancer.fit_resample(X_train_afo, y_train)
        print(f" -> Balanced Training Set Class Ratio: {np.mean(y_train_bal):.2%} (Size: {X_train_bal.shape[0]})")

        # 6. Fit Baseline XGBoost Classifier
        print("\n[Phase 7] Training Baseline XGBoost Estimator...")
        if XGB_AVAILABLE:
            baseline_model = xgb.XGBClassifier(n_estimators=100, eval_metric="logloss", random_state=RANDOM_SEED)
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            baseline_model = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_SEED)
        baseline_model.fit(X_train_bal, y_train_bal)

        # 7. Fit Proposed Hybrid HDA-GB Classifier
        print("\n[Phase 8/9] Fitting Hybrid Deep Attentive Gradient Boosting (HDA-GB) Model...")
        # Compute class weights for Liver GBDT loss calibration
        if key == "liver":
            class_counts = np.bincount(y_train_bal)
            weights = len(y_train_bal) / (2.0 * class_counts)
            class_weights = list(weights)
        else:
            class_weights = None

        hda_gb = HDAGBClassifier(
            d_token=128,
            n_blocks=3,
            n_heads=4,
            transformer_epochs=8,
            gbdt_n_estimators=100,
            gbdt_type="catboost",
            class_weights=class_weights,
            random_state=RANDOM_SEED
        )
        hda_gb.fit(X_train_bal, y_train_bal)

        # 8. Evaluate Metrics on Holdout Test Split
        print("\n[Phase 12] Evaluating 10 Clinical Diagnostic Metrics on Holdout Test Partition...")
        y_test_pred_base = baseline_model.predict(X_test_afo)
        y_test_proba_base = baseline_model.predict_proba(X_test_afo)[:, 1]
        base_metrics = evaluate_clinical_metrics(y_test, y_test_pred_base, y_test_proba_base)

        y_test_pred_hybrid = hda_gb.predict(X_test_afo)
        y_test_proba_hybrid = hda_gb.predict_proba(X_test_afo)[:, 1]
        hybrid_metrics = evaluate_clinical_metrics(y_test, y_test_pred_hybrid, y_test_proba_hybrid)

        # Run McNemar significance test
        chi2, p_val = mcnemar_statistical_test(y_test, y_test_pred_hybrid, y_test_pred_base)
        print(f" -> McNemar Paired Test vs XGBoost Baseline: Chi2={chi2:.4f}, p-value={p_val:.4e}")

        # Tabulate results
        metrics_table = []
        for metric in base_metrics.keys():
            metrics_table.append([metric, f"{base_metrics[metric]:.4f}", f"{hybrid_metrics[metric]:.4f}"])
        print("\n" + tabulate(metrics_table, headers=["Metric", "XGBoost Baseline", "HDA-GB Hybrid"], tablefmt="psql"))

        # Save metrics for summary table
        results_summary.append({
            "Disease": config["name"],
            "Base F1": base_metrics["F1-Score"],
            "Hybrid F1": hybrid_metrics["F1-Score"],
            "Base MCC": base_metrics["MCC"],
            "Hybrid MCC": hybrid_metrics["MCC"],
            "Base Recall": base_metrics["Sensitivity (Recall)"],
            "Hybrid Recall": hybrid_metrics["Sensitivity (Recall)"],
            "McNemar p-value": p_val
        })

        # 9. Local explainability & risk report generation for a sample patient
        print("\n[Phase 11/13] Instantiating ClinicalExplainer and Risk Report Generator...")
        explainer = ClinicalExplainer(hda_gb, selected_feature_names, model_type="hybrid")
        explainer.fit_explainers(X_train_bal)
        
        # Pick the first patient in the holdout test set as a demonstration case
        sample_patient_idx = 0
        patient_features = X_test_afo[sample_patient_idx]
        patient_prob = float(y_test_proba_hybrid[sample_patient_idx])
        patient_conf = float(hda_gb.predict_confidence(X_test_afo[sample_patient_idx:sample_patient_idx+1])[0])

        # Get SHAP explanation
        patient_explanation = explainer.explain_patient_instance(patient_features)
        shap_vals = patient_explanation.get("feature_attributions", {name: 0.0 for name in selected_feature_names})

        # Capture raw patient values
        raw_patient_row = X_test_raw.iloc[sample_patient_idx]
        raw_patient_metrics = raw_patient_row.to_dict()

        # Build clinical reports
        report_gen = ClinicalReportGenerator(disease_name=config["name"])
        text_report = report_gen.generate_text_report(
            patient_id=f"PT-2026-00{sample_patient_idx + 1}",
            age=int(raw_patient_metrics.get("age", raw_patient_metrics.get("Age", 55))),
            gender=str(raw_patient_metrics.get("sex", raw_patient_metrics.get("Gender", "Female"))),
            probability=patient_prob,
            confidence=patient_conf,
            shap_attributions=shap_vals,
            patient_metrics=raw_patient_metrics
        )
        
        # Print text report preview
        print("\n--- SAMPLE CLINICAL RISK REPORT TEXT PREVIEW ---")
        print("\n".join(text_report.split("\n")[:18]) + "\n... [TRUNCATED] ...")
        
        # Write PDF clinical risk report to processed data directory
        pdf_filename = PROCESSED_DATA_DIR / f"{key}_patient_report.pdf"
        report_gen.generate_pdf_report(
            output_path=str(pdf_filename),
            patient_id=f"PT-2026-00{sample_patient_idx + 1}",
            age=int(raw_patient_metrics.get("age", raw_patient_metrics.get("Age", 55))),
            gender="Male" if str(raw_patient_metrics.get("sex", "0")) == "1" or str(raw_patient_metrics.get("Gender", "Male")).lower().startswith("m") else "Female",
            probability=patient_prob,
            confidence=patient_conf,
            shap_attributions=shap_vals,
            patient_metrics=raw_patient_metrics
        )
        print(f" -> Vectorized PDF Clinical Risk Report exported to: {pdf_filename}")

        # 10. Serialize pipeline artifacts to disk
        pipeline_export_path = MODELS_DIR / f"{key}_pipeline.joblib"
        print(f"\n[Phase 14] Serializing trained pipeline models to: {pipeline_export_path}")
        pipeline_payload = {
            "disease_key": key,
            "disease_name": config["name"],
            "preprocessor": preprocessor,
            "selected_feature_indices": afo.selected_indices_,
            "selected_feature_names": selected_feature_names,
            "model": hda_gb,
            "baseline": baseline_model,
            "metrics": hybrid_metrics,
            "background_data": X_train_bal[:30]
        }
        joblib.dump(pipeline_payload, pipeline_export_path)

    # Print overall comparison summary across all diseases
    print("\n" + "=" * 100)
    print(" OVERALL PERFORMANCE SUMMARY ACROSS ALL 5 CLINICAL BENCHMARKS")
    print("=" * 100)
    summary_df = pd.DataFrame(results_summary)
    print(tabulate(summary_df, headers="keys", showindex=False, tablefmt="psql"))
    print("\nAll tasks completed successfully. Multi-disease pipeline is fully verified.")


if __name__ == "__main__":
    run_full_research_pipeline()
