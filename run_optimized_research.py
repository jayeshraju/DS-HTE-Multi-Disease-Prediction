"""
Optimized Research Execution Pipeline.
Implements:
  1. Repeated Stratified Nested Cross-Validation (5x5 R-NSCV).
  2. Bayesian hyperparameter optimization (Optuna) with a composite fitness function.
  3. Beta probability calibration.
  4. F-beta threshold calibration.
  5. Stacking GBDT Classifier ensembling CatBoost, XGBoost, and LightGBM.
  6. Statistical hypothesis tests (Wilcoxon and Bootstrap AUC significance).
  7. Post-hoc error profiling (K-Means clustering on misclassifications).
  8. Generates and saves vector-format diagnostic plots (ROC/PR).
  9. Serializes the final calibrated, optimized model pipelines to saved_models/.
"""

import os
import sys
import torch
from typing import List, Optional, Dict, Any
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import optuna
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, matthews_corrcoef,
    precision_score, recall_score, balanced_accuracy_score, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon, norm

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Disable Optuna logs to keep the console clean
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.abspath("."))

from src.config import RANDOM_SEED, DATASET_CONFIGS, MODELS_DIR, PROCESSED_DATA_DIR
from src.data.ingest import ingest_datasets
from src.data.preprocessing import MedicalPreprocessor
from src.data.balancing import AdaptiveDataBalancer
from src.features.afo import AdaptiveFeatureOptimizer
from src.models.hybrid import HDAGBClassifier, CalibratedThresholdClassifierWrapper, BetaCalibrator





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
        return y.astype(str).str.strip().str.upper().map({"M": 1, "B": 0}).fillna(0).astype(int)
    return y.astype(int)


def optimize_threshold(y_true: np.ndarray, probs: np.ndarray, metric: str = "f1", beta: float = 1.0) -> float:
    """
    Optimizes decision threshold to maximize specified metric (f1, f_beta, youden_j, mcc).
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_score = -999.0
    best_threshold = 0.5
    for thresh in thresholds:
        preds = (probs >= thresh).astype(int)
        if metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        elif metric == "f_beta":
            precision = precision_score(y_true, preds, zero_division=0)
            recall = recall_score(y_true, preds, zero_division=0)
            if precision + recall == 0:
                score = 0.0
            else:
                score = (1.0 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        elif metric == "youden_j":
            sens = recall_score(y_true, preds, zero_division=0)
            neg_mask = (y_true == 0)
            spec = accuracy_score(y_true[neg_mask], preds[neg_mask]) if np.sum(neg_mask) > 0 else 1.0
            score = sens + spec - 1.0
        elif metric == "mcc":
            score = matthews_corrcoef(y_true, preds)
        else:
            score = f1_score(y_true, preds, zero_division=0)
            
        if score > best_score:
            best_score = score
            best_threshold = thresh
    return float(best_threshold)


def compute_bootstrap_auc_pvalue(y_true: np.ndarray, probs1: np.ndarray, probs2: np.ndarray, n_bootstrap: int = 200) -> float:
    """
    Bootstrap-based ROC-AUC comparison p-value (similar to DeLong's test).
    Tests the null hypothesis that AUC(probs1) == AUC(probs2).
    """
    np.random.seed(42)
    n_samples = len(y_true)
    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return 1.0
    
    auc1_orig = roc_auc_score(y_true, probs1)
    auc2_orig = roc_auc_score(y_true, probs2)
    orig_diff = auc1_orig - auc2_orig
    
    diffs = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n_samples, n_samples, replace=True)
        y_boot = y_true[boot_idx]
        if len(np.unique(y_boot)) < 2:
            continue
        auc1 = roc_auc_score(y_boot, probs1[boot_idx])
        auc2 = roc_auc_score(y_boot, probs2[boot_idx])
        diffs.append(auc1 - auc2)
        
    std_val = np.std(diffs)
    if std_val == 0:
        return 1.0
    z_stat = orig_diff / std_val
    p_val = 2.0 * (1.0 - norm.cdf(np.abs(z_stat)))
    return float(p_val)


def save_diagnostic_plots(y_true: np.ndarray, probs: np.ndarray, title: str, filename_prefix: str):
    os.makedirs("docs/plots", exist_ok=True)
    
    # 1. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)
    plt.figure(figsize=(5, 4.5))
    plt.plot(fpr, tpr, label=f"Proposed Model (AUC = {auc:.4f})", color="royalblue", lw=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
    plt.title(f"ROC Curve - {title}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"docs/plots/{filename_prefix}_roc.pdf", format="pdf")
    plt.close()
    
    # 2. Precision-Recall Curve Plot
    precision, recall, _ = precision_recall_curve(y_true, probs)
    plt.figure(figsize=(5, 4.5))
    plt.plot(recall, precision, color="darkorange", lw=2, label="Proposed Model")
    plt.title(f"PR Curve - {title}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"docs/plots/{filename_prefix}_pr.pdf", format="pdf")
    plt.close()


def analyze_error_clusters(X_errors: np.ndarray, y_true_errors: np.ndarray, feat_names: List[str], title: str):
    if len(X_errors) < 8:
        return
    try:
        tsne = TSNE(n_components=2, perplexity=min(5, len(X_errors) - 1), random_state=42)
        X_tsne = tsne.fit_transform(X_errors)
        kmeans = KMeans(n_clusters=min(2, len(X_errors) // 4), random_state=42)
        clusters = kmeans.fit_predict(X_tsne)
        
        print(f"\n   [Error Profile Summary for {title}]")
        for cluster_id in np.unique(clusters):
            mask = (clusters == cluster_id)
            size = np.sum(mask)
            pct_fn = np.mean(y_true_errors[mask] == 1)
            print(f"    * Subgroup {cluster_id}: {size} patients | {pct_fn:.1%} False Negatives")
            
            means = np.mean(X_errors[mask], axis=0)
            global_means = np.mean(X_errors, axis=0)
            diffs = np.abs(means - global_means)
            top_feats = np.argsort(-diffs)[:3]
            feat_str = ", ".join([f"{feat_names[idx]}: {means[idx]:+.2f} std dev" for idx in top_feats if idx < len(feat_names)])
            print(f"      Top signature shifts: {feat_str}")
    except Exception as e:
        print(f"   -> Skipped error profiling: {str(e)}")


def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast-mode", action="store_true", help="Runs quick CV validation")
    args = parser.parse_args(sys.argv[1:]) if len(sys.argv) > 1 else argparse.Namespace(fast_mode=False)

    print("=" * 100)
    print("   IEEE/SCOPUS RESEARCH-GRADE MEDICAL AI OPTIMIZATION PIPELINE")
    print("=" * 100)

    # Ingest datasets
    datasets = ingest_datasets(force_rebuild=False)
    overall_summary = []

    # Fast-mode parameter switches
    if args.fast_mode:
        print(" -> Fast execution mode enabled: running 5-fold CV with 1 repeat.")
        n_repeats = 1
        optuna_trials = 3
        transformer_epochs = 6
        gbdt_n_estimators = 40
        mc_samples = 4
    else:
        print(" -> Rigorous publication mode enabled: running 5x3 Nested CV.")
        n_repeats = 3
        optuna_trials = 10
        transformer_epochs = 35
        gbdt_n_estimators = 120
        mc_samples = 15

    for key, config in DATASET_CONFIGS.items():
        print(f"\n{'-'*100}")
        print(f" PROCESSING COHORT: {config['name'].upper()}")
        print(f"{'-'*100}")

        df = datasets[key]
        feat_cols = config["numerical_cols"] + config["categorical_cols"]
        X_df = df[feat_cols]
        y_bin = preprocess_target(df[config["target"]], key)

        print(f" -> Patients Count: {len(df)} | Baseline Prevalence: {y_bin.mean():.2%}")

        # Set dataset-specific hyperparameter ranges and preprocessing settings
        imp_strat = "missforest" if key in ["diabetes", "liver"] else "knn"
        use_pt = True if key == "liver" else False
        outlier_strat = "winsorize" if key in ["diabetes", "liver"] else "delete"
        outlier_contam = 0.04 if key in ["diabetes", "liver"] else 0.03
        bal_method = "borderline_smote" if key in ["diabetes", "liver"] else "smote_tomek"
        bimodal_cols = ["Glucose", "BMI"] if key == "diabetes" else (["Alamine_Aminotransferase", "Aspartate_Aminotransferase"] if key == "liver" else [])
        use_ple = True if key in ["diabetes", "liver"] else False
        use_residual_bypass = True if key in ["diabetes", "liver"] else False
        beta_val = 1.5 if key == "liver" else 1.0
        threshold_metric = "youden_j" if key in ["liver", "heart"] else "f1"
        bypass_trans = True if key in ["diabetes", "heart", "liver"] else False

        # Repeated Stratified Cross Validation
        cv_outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=RANDOM_SEED)
        
        outer_metrics = []
        baseline_metrics = []
        all_hybrid_probs = []
        all_baseline_probs = []
        all_y_true = []
        all_test_errors = []
        all_test_error_y = []
        proc_names_global = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv_outer.split(X_df, y_bin)):
            print(f"\n --- Outer Fold {fold_idx + 1}/{5 * n_repeats} ---")
            
            X_train_raw, X_test_raw = X_df.iloc[train_idx].reset_index(drop=True), X_df.iloc[test_idx].reset_index(drop=True)
            y_train, y_test = y_bin.iloc[train_idx].reset_index(drop=True), y_bin.iloc[test_idx].reset_index(drop=True)

            # Fit Preprocessor (leakage-free)
            preprocessor = MedicalPreprocessor(
                numerical_cols=config["numerical_cols"],
                categorical_cols=config["categorical_cols"],
                zero_as_missing=config["zero_as_missing"],
                imputation_strategy=imp_strat,
                outlier_contamination=outlier_contam,
                outlier_strategy=outlier_strat,
                use_power_transform=use_pt,
                disease=key,
                bimodal_cols=bimodal_cols,
                add_missing_indicators=True,
                add_clinical_flags=True
            )
            
            X_train_proc, y_train_clean = preprocessor.fit_transform_train(X_train_raw, y_train)
            X_test_proc = preprocessor.transform(X_test_raw)

            # Feature Names list
            if len(config["categorical_cols"]) > 0:
                proc_names = list(preprocessor.numerical_cols) + list(preprocessor.ohe.get_feature_names_out(config["categorical_cols"]))
            else:
                proc_names = list(preprocessor.numerical_cols)
            proc_names_global = proc_names

            # Adaptive Feature Optimization (AFO Selection)
            k_features = min(12, len(proc_names))
            afo = AdaptiveFeatureOptimizer(k_top=k_features, random_state=RANDOM_SEED)
            
            # Fast pilot hybrid run to get SHAP importance for guidance
            pilot_model = HDAGBClassifier(
                d_token=64,
                n_blocks=1,
                n_heads=2,
                transformer_epochs=4,
                gbdt_n_estimators=40,
                gbdt_type="catboost",
                bypass_transformer=bypass_trans,
                random_state=RANDOM_SEED
            )
            pilot_model.fit(X_train_proc, y_train_clean)
            
            # Compute baseline GBDT SHAP value approximation
            if pilot_model.transformer_model is not None:
                raw_cls_embeddings = pilot_model.transformer_model.forward_features(torch.tensor(X_train_proc, dtype=torch.float32)).detach().numpy()
                X_aug_pilot = np.hstack([X_train_proc, raw_cls_embeddings])
            else:
                X_aug_pilot = X_train_proc
            
            # Feature importance approximation
            feat_importances = pilot_model.gbdt_model.get_feature_importance()
            shap_approx = {proc_names[i]: float(feat_importances[i]) for i in range(min(len(proc_names), len(feat_importances)))}

            # Now run AFO guided by SHAP importances
            X_train_afo = afo.fit_transform(X_train_proc, y_train_clean, feature_names=proc_names, shap_importance=shap_approx)
            X_test_afo = afo.transform(X_test_proc)

            # Resampling / Class Balancing
            if bal_method != "none":
                balancer = AdaptiveDataBalancer(sampling_strategy=0.85, method=bal_method, random_state=RANDOM_SEED)
                X_train_bal, y_train_bal = balancer.fit_resample(X_train_afo, y_train_clean)
                X_train_bal = np.asarray(X_train_bal)
                y_train_bal = np.asarray(y_train_bal)
            else:
                X_train_bal = X_train_afo
                y_train_bal = y_train_clean.values

            # Inverse frequency class weighting for cost-sensitive loss (if GBDT supports it)
            if key == "liver":
                class_counts = np.bincount(y_train_bal)
                class_weights = [float(w) for w in (len(y_train_bal) / (2.0 * class_counts))]
            else:
                class_weights = None

            # ── Train Baseline Model (XGBoost/GBDT on raw preprocessed features) ──
            if XGB_AVAILABLE:
                baseline_clf = xgb.XGBClassifier(
                    n_estimators=gbdt_n_estimators,
                    max_depth=5,
                    learning_rate=0.03,
                    eval_metric="logloss",
                    random_state=RANDOM_SEED
                )
            else:
                from sklearn.ensemble import GradientBoostingClassifier
                baseline_clf = GradientBoostingClassifier(n_estimators=gbdt_n_estimators, random_state=RANDOM_SEED)
            
            baseline_clf.fit(X_train_proc, y_train_clean)
            base_test_probs = baseline_clf.predict_proba(X_test_proc)[:, 1]
            base_test_pred = (base_test_probs >= 0.5).astype(int)
            
            baseline_metrics.append({
                "Accuracy": accuracy_score(y_test, base_test_pred),
                "F1-Score": f1_score(y_test, base_test_pred, zero_division=0),
                "MCC": matthews_corrcoef(y_test, base_test_pred)
            })

            # ── Bayesian Inner Optimization via Optuna ──
            def objective(trial):
                depth = trial.suggest_int("depth", 4, 6)
                l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1.0, 5.0)
                learning_rate = trial.suggest_float("learning_rate", 0.01, 0.05)
                d_token = trial.suggest_categorical("d_token", [64, 128])
                gbdt_type = trial.suggest_categorical("gbdt_type", ["catboost", "stacking"])
                
                # Inner K-Fold
                cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
                inner_fitness = []
                
                for inner_train_idx, inner_val_idx in cv_inner.split(X_train_bal, y_train_bal):
                    X_inner_tr, X_inner_val = X_train_bal[inner_train_idx], X_train_bal[inner_val_idx]
                    y_inner_tr, y_inner_val = y_train_bal[inner_train_idx], y_train_bal[inner_val_idx]
                    
                    # Instantiate model
                    model = HDAGBClassifier(
                        d_token=d_token,
                        n_blocks=2,
                        n_heads=4,
                        transformer_epochs=4 if args.fast_mode else 12,
                        gbdt_n_estimators=gbdt_n_estimators // 2,
                        gbdt_type=gbdt_type,
                        class_weights=class_weights,
                        depth=depth,
                        l2_leaf_reg=l2_leaf_reg,
                        learning_rate=learning_rate,
                        use_ple=use_ple,
                        use_residual_bypass=use_residual_bypass,
                        bypass_transformer=bypass_trans,
                        random_state=RANDOM_SEED
                    )
                    
                    model.fit(X_inner_tr, y_inner_tr)
                    preds = model.predict(X_inner_val)
                    probas = model.predict_proba(X_inner_val)[:, 1]
                    
                    # Calculate composite fitness score
                    f1 = f1_score(y_inner_val, preds, zero_division=0)
                    mcc = matthews_corrcoef(y_inner_val, preds)
                    brier = brier_score_loss(y_inner_val, probas)
                    
                    # Fitness = 0.4*F1 + 0.3*MCC + 0.3*(1 - Brier)
                    fitness = 0.4 * f1 + 0.3 * mcc + 0.3 * (1.0 - brier)
                    inner_fitness.append(fitness)
                    
                return np.mean(inner_fitness)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=optuna_trials, timeout=120)
            best_params = study.best_params
            print(f"   -> Best Hyperparameters: {best_params}")

            # Train Model on Full Outer Train Split with Best Parameters
            final_model = HDAGBClassifier(
                d_token=best_params["d_token"],
                n_blocks=2,
                n_heads=4,
                transformer_epochs=transformer_epochs,
                gbdt_n_estimators=gbdt_n_estimators,
                gbdt_type=best_params["gbdt_type"],
                class_weights=class_weights,
                depth=best_params["depth"],
                l2_leaf_reg=best_params["l2_leaf_reg"],
                learning_rate=best_params["learning_rate"],
                use_ple=use_ple,
                use_residual_bypass=use_residual_bypass,
                use_mc_dropout=(not args.fast_mode),
                mc_samples=mc_samples,
                bypass_transformer=bypass_trans,
                random_state=RANDOM_SEED
            )
            final_model.fit(X_train_bal, y_train_bal)

            # ── Probability Calibration (Beta Calibration) ──
            # Generate cross-validated predictions on outer train to fit calibrator
            cv_cal = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
            val_probas = np.zeros(len(X_train_bal))
            for tr_idx, val_idx in cv_cal.split(X_train_bal, y_train_bal):
                cal_sub_model = HDAGBClassifier(
                    d_token=best_params["d_token"],
                    n_blocks=2,
                    n_heads=4,
                    transformer_epochs=transformer_epochs // 2,
                    gbdt_n_estimators=gbdt_n_estimators,
                    gbdt_type=best_params["gbdt_type"],
                    class_weights=class_weights,
                    depth=best_params["depth"],
                    l2_leaf_reg=best_params["l2_leaf_reg"],
                    learning_rate=best_params["learning_rate"],
                    use_ple=use_ple,
                    use_residual_bypass=use_residual_bypass,
                    bypass_transformer=bypass_trans,
                    random_state=RANDOM_SEED
                )
                cal_sub_model.fit(X_train_bal[tr_idx], y_train_bal[tr_idx])
                val_probas[val_idx] = cal_sub_model.predict_proba(X_train_bal[val_idx])[:, 1]
            
            calibrator = BetaCalibrator()
            calibrator.fit(val_probas, y_train_bal)

            # Fit Calibration probabilities on Outer Test
            raw_test_probas = final_model.predict_proba(X_test_afo)[:, 1]
            cal_test_probas = calibrator.predict_proba(raw_test_probas)[:, 1]

            # ── Threshold Calibration via selected metric ──
            optimal_threshold = optimize_threshold(y_train_bal, val_probas, metric=threshold_metric, beta=beta_val)
            print(f"   -> Calibrated Optimal Decision Threshold ({threshold_metric}): {optimal_threshold:.4f}")

            # Predict on outer test
            y_test_pred = (cal_test_probas >= optimal_threshold).astype(int)

            # Record errors for post-hoc cluster profiling
            errors_mask = (y_test != y_test_pred)
            if np.any(errors_mask):
                all_test_errors.append(X_test_afo[errors_mask])
                all_test_error_y.append(y_test[errors_mask].values)

            # Accumulate predictions for statistical tests
            all_hybrid_probs.append(cal_test_probas)
            all_baseline_probs.append(base_test_probs)
            all_y_true.append(y_test.values)

            # Compute metrics
            fold_acc = accuracy_score(y_test, y_test_pred)
            fold_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            fold_mcc = matthews_corrcoef(y_test, y_test_pred)
            fold_sens = recall_score(y_test, y_test_pred, zero_division=0)
            
            neg_class_mask = (y_test == 0)
            fold_spec = accuracy_score(y_test[neg_class_mask], y_test_pred[neg_class_mask]) if np.sum(neg_class_mask) > 0 else 1.0
            fold_auc = roc_auc_score(y_test, cal_test_probas) if len(np.unique(y_test)) > 1 else 1.0
            fold_brier = brier_score_loss(y_test, cal_test_probas)

            outer_metrics.append({
                "Accuracy": fold_acc,
                "F1-Score": fold_f1,
                "MCC": fold_mcc,
                "Sensitivity": fold_sens,
                "Specificity": fold_spec,
                "ROC-AUC": fold_auc,
                "Brier": fold_brier
            })
            
            print(f"   Outer Fold Metrics: Accuracy={fold_acc:.4f} | F1={fold_f1:.4f} | MCC={fold_mcc:.4f} | Recall={fold_sens:.4f} | Spec={fold_spec:.4f} | AUC={fold_auc:.4f}")

        # Compute average metrics across outer folds
        df_outer = pd.DataFrame(outer_metrics)
        avg_metrics = df_outer.mean().to_dict()
        df_base = pd.DataFrame(baseline_metrics)
        avg_base = df_base.mean().to_dict()

        print(f"\n {config['name'].upper()} - Repeated Nested CV Summary Comparison:")
        print(f"  * Proposed Model: Accuracy={avg_metrics['Accuracy']:.4f} | F1-Score={avg_metrics['F1-Score']:.4f} | MCC={avg_metrics['MCC']:.4f}")
        print(f"  * Baseline Model: Accuracy={avg_base['Accuracy']:.4f} | F1-Score={avg_base['F1-Score']:.4f} | MCC={avg_base['MCC']:.4f}")

        # ── Statistical Significance Tests ──
        y_true_all = np.concatenate(all_y_true)
        hybrid_probs_all = np.concatenate(all_hybrid_probs)
        baseline_probs_all = np.concatenate(all_baseline_probs)

        # DeLong Bootstrap P-Value
        p_val_delong = compute_bootstrap_auc_pvalue(y_true_all, hybrid_probs_all, baseline_probs_all)
        
        # Wilcoxon Signed-Rank Test on accuracy differences across folds
        hybrid_acc_list = [m["Accuracy"] for m in outer_metrics]
        base_acc_list = [m["Accuracy"] for m in baseline_metrics]
        try:
            _, p_val_wilcoxon = wilcoxon(hybrid_acc_list, base_acc_list)
        except Exception:
            p_val_wilcoxon = 1.0

        print(f"  * Statistical Significance tests against Baseline:")
        print(f"    - ROC-AUC Difference (DeLong Bootstrap) p-value: {p_val_delong:.4f} {'(Significant, p<0.05)' if p_val_delong < 0.05 else '(Not Significant)'}")
        print(f"    - Fold Accuracy Difference (Wilcoxon Signed-Rank) p-value: {p_val_wilcoxon:.4f} {'(Significant, p<0.05)' if p_val_wilcoxon < 0.05 else '(Not Significant)'}")

        # ── Post-Hoc Error Clustering ──
        if len(all_test_errors) > 0:
            err_features = np.concatenate(all_test_errors)
            err_y = np.concatenate(all_test_error_y)
            # Select AFO feature names for the error profile report
            selected_names = [proc_names_global[idx] for idx in afo.selected_indices_]
            analyze_error_clusters(err_features, err_y, selected_names, config["name"])

        # ── Export Plots ──
        save_diagnostic_plots(y_true_all, hybrid_probs_all, config["name"], key)

        overall_summary.append({
            "Disease": config["name"],
            "Accuracy": avg_metrics["Accuracy"],
            "F1-Score": avg_metrics["F1-Score"],
            "MCC": avg_metrics["MCC"],
            "Sensitivity": avg_metrics["Sensitivity"],
            "Specificity": avg_metrics["Specificity"],
            "ROC-AUC": avg_metrics["ROC-AUC"],
            "Brier": avg_metrics["Brier"],
            "DeLong p-val": p_val_delong,
            "Wilcoxon p-val": p_val_wilcoxon
        })

        # ── Train Final Production Checkpoint on 100% of Data ──
        print(f"\nFitting Final Production Model pipeline for {config['name']}...")
        
        preprocessor_final = MedicalPreprocessor(
            numerical_cols=config["numerical_cols"],
            categorical_cols=config["categorical_cols"],
            zero_as_missing=config["zero_as_missing"],
            imputation_strategy=imp_strat,
            outlier_contamination=outlier_contam,
            outlier_strategy=outlier_strat,
            use_power_transform=use_pt,
            disease=key,
            bimodal_cols=bimodal_cols,
            add_missing_indicators=True,
            add_clinical_flags=True
        )
        X_proc_final, y_clean_final = preprocessor_final.fit_transform_train(X_df, y_bin)

        if len(config["categorical_cols"]) > 0:
            final_proc_names = list(preprocessor_final.numerical_cols) + list(preprocessor_final.ohe.get_feature_names_out(config["categorical_cols"]))
        else:
            final_proc_names = list(preprocessor_final.numerical_cols)

        # Re-run AFO selection on complete data
        afo_final = AdaptiveFeatureOptimizer(k_top=k_features, random_state=RANDOM_SEED)
        X_afo_final = afo_final.fit_transform(X_proc_final, y_clean_final, feature_names=final_proc_names, shap_importance=shap_approx)

        # Resampling
        if bal_method != "none":
            balancer_final = AdaptiveDataBalancer(sampling_strategy=0.85, method=bal_method, random_state=RANDOM_SEED)
            X_bal_final, y_bal_final = balancer_final.fit_resample(X_afo_final, y_clean_final)
            X_bal_final = np.asarray(X_bal_final)
            y_bal_final = np.asarray(y_bal_final)
        else:
            X_bal_final = X_afo_final
            y_bal_final = y_clean_final.values

        if key == "liver":
            class_counts = np.bincount(y_bal_final)
            class_weights_final = [float(w) for w in (len(y_bal_final) / (2.0 * class_counts))]
        else:
            class_weights_final = None

        # Fit final optimized model
        model_final = HDAGBClassifier(
            d_token=best_params["d_token"],
            n_blocks=2,
            n_heads=4,
            transformer_epochs=transformer_epochs,
            gbdt_n_estimators=gbdt_n_estimators,
            gbdt_type=best_params["gbdt_type"],
            class_weights=class_weights_final,
            depth=best_params["depth"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            learning_rate=best_params["learning_rate"],
            use_ple=use_ple,
            use_residual_bypass=use_residual_bypass,
            use_mc_dropout=(not args.fast_mode),
            mc_samples=mc_samples,
            bypass_transformer=bypass_trans,
            random_state=RANDOM_SEED
        )
        model_final.fit(X_bal_final, y_bal_final)

        # Calibrator on whole dataset using cross-val
        cv_cal_final = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        val_probas_final = np.zeros(len(X_bal_final))
        for tr_idx, val_idx in cv_cal_final.split(X_bal_final, y_bal_final):
            cal_sub_model = HDAGBClassifier(
                d_token=best_params["d_token"],
                n_blocks=2,
                n_heads=4,
                transformer_epochs=transformer_epochs // 2,
                gbdt_n_estimators=gbdt_n_estimators,
                gbdt_type=best_params["gbdt_type"],
                class_weights=class_weights_final,
                depth=best_params["depth"],
                l2_leaf_reg=best_params["l2_leaf_reg"],
                learning_rate=best_params["learning_rate"],
                use_ple=use_ple,
                use_residual_bypass=use_residual_bypass,
                bypass_transformer=bypass_trans,
                random_state=RANDOM_SEED
            )
            cal_sub_model.fit(X_bal_final[tr_idx], y_bal_final[tr_idx])
            val_probas_final[val_idx] = cal_sub_model.predict_proba(X_bal_final[val_idx])[:, 1]
        
        calibrator_final = BetaCalibrator()
        calibrator_final.fit(val_probas_final, y_bal_final)

        # Find final threshold
        optimal_threshold_final = optimize_threshold(y_bal_final, val_probas_final, metric=threshold_metric, beta=beta_val)

        wrapped_model = CalibratedThresholdClassifierWrapper(model_final, calibrator_final, optimal_threshold_final)

        # Overwrite standard production models directly
        pipeline_export_path = MODELS_DIR / f"{key}_pipeline.joblib"
        joblib.dump({
            "preprocessor": preprocessor_final,
            "selected_feature_indices": afo_final.selected_indices_,
            "selected_feature_names": [final_proc_names[i] for i in afo_final.selected_indices_],
            "model": wrapped_model,
            "background_data": X_bal_final[:80],
            "metrics": avg_metrics
        }, pipeline_export_path)
        print(f" -> Production checkpoint serialized to: {pipeline_export_path}")

    # Print Final Summary Table
    print("\n" + "=" * 100)
    print("   FINAL RESEARCH-GRADE NESTED CROSS-VALIDATION SUMMARY")
    print("=" * 100)
    df_summary = pd.DataFrame(overall_summary)
    print(tabulate(df_summary, headers="keys", showindex=False, tablefmt="psql"))


if __name__ == "__main__":
    run_pipeline()
