# Research Methodological Audit Report

This report presents a thorough methodological audit of the multi-disease clinical prediction pipeline, evaluating all data processing, training, validation, and evaluation components against rigorous clinical AI and IEEE/Scopus standards.

---

## 1. ARCHITECTURE MAP

```
DATA (Raw CSVs)
↓
CLEANING (NaN identification and target variable binarization)
↓
TARGET PROCESSING (Map categorical labels to binary integers)
↓
TRAIN/TEST SPLIT (80% Train / 20% Test partition via train_test_split)
↓
PREPROCESSING (Imputation, Winsorization, power transforms, OHE scaling fit on training folds)
↓
FEATURE ENGINEERING (Domain clinical ratios, e.g., HOMA-IR, De Ritis Ratio, clinical flags)
↓
FEATURE SELECTION (AFO consensus Borda ranking using MI, L1, XGBoost, and RFE)
↓
CLASS BALANCING (SMOTE + Tomek Links boundary cleaning on training folds)
↓
HYPERPARAMETER OPTIMIZATION (Optuna Bayesian HPO using inner Stratified K-Fold)
↓
MODEL TRAINING (Stacking ensemble HDAGB Classifier: FT-Transformer + GBDTs)
↓
CALIBRATION (Beta calibration fit on out-of-fold train probabilities)
↓
THRESHOLD OPTIMIZATION (Youden's J or F1 optimization on validation folds)
↓
OUTER TEST (Evaluation of saved model pipeline on unseen test fold)
↓
STATISTICAL ANALYSIS (DeLong Bootstrap, McNemar paired classification, Wilcoxon fold tests)
↓
EXPLAINABILITY (TreeSHAP global summary and local patient risk attributions)
↓
VISUALIZATION (Publication-grade 300 DPI ROC, PR, Calibration, and DCA curves)
```

---

## 2. DETAILED MODULE AUDIT

### Component 1: Evaluation Pipeline (`evaluate_models.py`)
* **Current Behavior**: Splits the dataset into 80% train / 20% test using a fixed seed, loads the saved pipeline from `saved_models/`, and evaluates the model on the 20% test partition.
* **Potential Leakage**: **CRITICAL**. The final serialized pipeline models were trained on 100% of the dataset. Therefore, the 20% test partition was already seen by the model, causing massive target and sample leakage during evaluation. This explains the artificially high accuracy values (e.g. 100% CKD, 92% ILPD, 98% WBCD) in `evaluate_models.py`.
* **Statistical Weakness**: The reported generalization metrics do not reflect actual out-of-sample performance.
* **Implementation Weakness**: Overestimates baseline and hybrid model accuracy, masking model generalization defects.
* **Reproducibility Weakness**: The evaluation script results will vary if trained on different splits, but since it is leaked, it shows misleading stability.
* **Severity**: **CRITICAL**
* **Recommended Correction**: Train the final serialized model *only* on the 80% training split. Keep the 20% test split completely sealed and untouched by any preprocessing, feature selection, or training steps until final evaluation.

### Component 2: Inner Cross-Validation Loop (`run_optimized_research.py`)
* **Current Behavior**: Optuna HPO splits the resampled and feature-selected dataset `X_train_bal` using `StratifiedKFold`.
* **Potential Leakage**: **HIGH**. Applying SMOTE and feature selection *before* the inner cross-validation split leaks information from the validation folds into the training folds. The synthetic samples in the validation set are generated from parent samples that are also present in the training set.
* **Statistical Weakness**: The inner validation score is highly optimistic, leading Optuna to select hyperparameters that overfit the SMOTE-generated distribution.
* **Implementation Weakness**: Feature selection (AFO) and resampling (SMOTE) are not encapsulated inside the inner folds.
* **Reproducibility Weakness**: Distorts the selection of optimal learning rates, depths, and ensemble stacking configurations.
* **Severity**: **HIGH**
* **Recommended Correction**: Structure the inner cross-validation loop such that the split occurs on raw training data (`X_train_raw`). For each inner fold, fit the preprocessor, select features, and apply SMOTE *only* on the inner training partition, transforming the inner validation partition without resampling.

### Component 3: Threshold Optimization
* **Current Behavior**: Decision thresholds are optimized on `X_train_bal` (which contains SMOTE-generated synthetic samples).
* **Potential Leakage**: **MEDIUM**
* **Statistical Weakness**: Optimizing the decision threshold on a balanced (50-50) synthetic class distribution yields thresholds that are poorly suited for the original imbalanced population, causing high false positive rates in clinical deployment.
* **Implementation Weakness**: Mismatch between optimization distribution and target test distribution.
* **Severity**: **MEDIUM**
* **Recommended Correction**: Optimize decision thresholds on the out-of-fold validation predictions using the original imbalanced class distribution, ensuring the threshold reflects the actual clinical prevalence.

### Component 4: Calibration Selection
* **Current Behavior**: Beta calibration is fit on `val_probas` derived from the resampled training partition.
* **Potential Leakage**: **MEDIUM**
* **Statistical Weakness**: Calibrator fits probabilities on balanced distributions, distorting confidence estimates on real-world distributions.
* **Severity**: **MEDIUM**
* **Recommended Correction**: Ensure the calibration set maintains the natural class prevalence of the cohort by fitting the Beta Calibrator on original validation folds.

---

## 3. SEVERITY SUMMARY MATRIX

| Component | Leakage Risk | Statistical Rigor | Implementation | Severity |
| :--- | :---: | :---: | :---: | :---: |
| Evaluation Split | CRITICAL | CRITICAL | HIGH | **CRITICAL** |
| Inner CV Loop | HIGH | HIGH | MEDIUM | **HIGH** |
| Threshold Optimization | LOW | MEDIUM | MEDIUM | **MEDIUM** |
| Calibration | LOW | MEDIUM | LOW | **MEDIUM** |
| Feature Selection | LOW | LOW | LOW | **LOW** |
| Dataset Duplicates | LOW | LOW | LOW | **LOW** |

---

## 4. ACTIONABLE REMEDIATION PLAN
1. **Seal 20% Test Split**: Split all raw datasets 80/20 at the beginning of the execution. Ensure the 20% test split is never modified or seen during training.
2. **Restructure Inner CV**: Run HPO inside the outer folds by performing inner splits first, then preprocessing, AFO, and SMOTE sequentially per inner fold.
3. **Calibrate on Natural Prevalence**: Perform threshold optimization and probability calibration using out-of-fold validation predictions with the original, un-resampled class labels.
