# DS-HTE End-to-End Clinical Research Workflow

> **Notice**: Research prototype, not a diagnostic device.

---

## Workflow Stages & Data Transformations

### Stage 1: Data Acquisition & Preprocessing
- **Inputs**: Raw CSV datasets (`data/raw/{pima,cleveland,ckd,ilpd,wbcd}.csv`).
- **Processing**:
  - Imputation of zero-coded physiological markers (glucose, diastolic BP, insulin, skin thickness, BMI).
  - Explicit missingness indicator generation for biological impossibilities.
  - Clinical ratio engineering (De Ritis ratio $\text{AST}/\text{ALT}$, Total/Direct Bilirubin index, HOMA-IR glucose-insulin proxy).
  - Categorical label encoding and group clustering for identical records.
- **Outputs**: Cleaned feature matrices and verified target vectors.

### Stage 2: Leak-Free Partitioning & Isolation
- **Inputs**: Sanitized cohort matrices.
- **Processing**:
  - Immediate stratification and carving of 20% holdout split (persisted to `results/logs/holdout_indices.json`).
  - Labeled strictly as **"previously inspected holdout (secondary)"** in compliance with Ground Rule 3.
  - 50-fold repeated cross-validation on remaining 80% development split (10 repeats of 5 folds).
- **Outputs**: `results/logs/folds.json` and cryptographic hashes of split boundaries.

### Stage 3: Dynamic Feature Optimization (AFO)
- **Inputs**: Inner-training partitions.
- **Processing**:
  - Multi-method consensus ranking using SHAP values, Permutation Importance, Mutual Information, and L1-penalized Logistic Regression.
  - Borda count aggregation combining feature ranks across methods.
  - Collinearity pruning: iteratively removes redundant features where $|r| > 0.85$.
- **Outputs**: Optimized subset of clinical biomarkers; logged to `results/logs/stability_metrics.json`.

### Stage 4: Multi-Model Training & Stacking Ensemble (DS-HTE)
- **Inputs**: Selected feature subsets on inner-training folds.
- **Processing**:
  - Fitting of 6 heterogeneous component classifiers: Tabular Transformer, LightGBM, CatBoost, XGBoost, Random Forest, regularized Logistic Regression.
  - Non-negative constrained SLSQP meta-learner optimization minimizing log-loss on out-of-fold predictions.
  - Diversity regularization penalty ($\lambda = 0.005$) and minimum weight floor ($w_i \ge 0.01$) to prevent ensemble collapse.
  - Post-hoc probability calibration (Platt scaling / temperature scaling).
  - Pre-declaration of two operating points on inner validation: Balanced (Youden's J) and Accuracy-Optimal.
- **Outputs**: Calibrated model pipelines and ensemble weight records (`results/audit/ensemble_weights.parquet`).

### Stage 5: Evaluation, Uncertainty Quantification & Explainability
- **Inputs**: Outer test splits and confirmatory holdout.
- **Processing**:
  - Prediction without label exposure; thresholding using inner-selected cutoffs.
  - Metric computation: Accuracy, Balanced Accuracy, ROC-AUC, PR-AUC, F1, MCC, Sensitivity, Specificity, Brier score, ECE.
  - Statistical significance: paired Wilcoxon tests, Nadeau-Bengio corrected t-tests, DeLong tests, and McNemar tests with Holm/Benjamini-Hochberg FDR adjustments.
  - Split-conformal prediction: calculates finite-sample non-conformity cutoffs and outputs prediction sets with "Abstain / Needs Expert Review" flags.
  - Global and local explainability: SHAP summary plots, LIME local attributions, and explainer consensus analysis.
- **Outputs**: `results/fold_metrics/outer_fold_metrics.parquet`, Tables I–IX, and publication figures.

### Stage 6: Provenance Stamping & Interactive Serving
- **Inputs**: All resulting tables, figures, models, and JSON summaries.
- **Processing**:
  - Cryptographic SHA-256 calculation of output files and generating `*.provenance.json` sidecars with script name, run_id, and input hashes.
  - Fast JSON export to `ui_data/` for SPA dashboard rendering.
  - Serving REST API endpoints and DS-HTE AI Copilot grounded state machine.
- **Outputs**: Interactive web dashboard at `http://localhost:8000/`.
