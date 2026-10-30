# Phase 4: Data Preprocessing — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Scalable Multi-Disease Early Prediction and Risk Stratification Framework using Explainable Artificial Intelligence and Adaptive Feature Optimization
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Unified Data Preprocessing Architecture

The preprocessing framework transforms raw, noisy, heterogeneous diagnostic clinical data into standardized, high-signal tabular representations ready for adaptive feature optimization and predictive modeling. Medical data preprocessing requires extreme domain caution to prevent data leakage, preserve small patient sample sizes, and accurately handle physiological anomalies.

```
+-----------------------------------------------------------------------------------+
|                            Raw Clinical Input Features                            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Step 1: Data Sanity & Validation Layer                                            |
| - Hash-based deduplication                                                        |
| - Schema & domain constraint validation (e.g., Blood Pressure > 0, Age > 0)       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Step 2: Missing Value Remediation Engine                                          |
| - Implicit Zero Replacement (PIMA Glucose/Insulin/BP -> NaN)                      |
| - Categorical Mode Imputation & Continuous KNN / MICE Model Imputation            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Step 3: Outlier Detection & Treatment                                             |
| - Multivariate Isolation Forest filtering combined with bounded Robust Trimming   |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Step 4: Categorical Feature Encoding                                              |
| - Binary Label Encoding & Nominal One-Hot Encoding                                |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Step 5: Data Partitioning & Leakage Barrier                                       |
| - Stratified Split (80% Train-Val / 20% Holdout Test)                             |
| - Fit Transformers ONLY on Training Partition; Transform Val/Test                 |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
| Step 6: Feature Rescaling & Target Imbalance Harmonization                        |
| - RobustScaler / StandardScaler fit on training folds                             |
| - Borderline-SMOTE oversampling applied strictly within training folds            |
+-----------------------------------------------------------------------------------+
```

---

## 2. Data Cleaning, Sanity Validation, & Missing Value Imputation

### 2.1 Deduplication & Range Validation Protocols
1. **Deduplication:** Hash signature check ($\text{MD5}$) across row tuples ensures zero repeated records entering the training pipeline.
2. **Physiological Range Validation:** Implicit missing values in datasets such as PIMA (where invalid physiological zero entries occur in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`) are flagged and converted to explicit `NaN` values using medical boundary rules:
   $$\text{Glucose} \le 0 \implies \text{NaN}, \quad \text{BloodPressure} \le 0 \implies \text{NaN}, \quad \text{BMI} \le 0 \implies \text{NaN}$$

### 2.2 Advanced Missing Value Handling Methodology
* **Mean / Median Imputation (Rejected for Clinical Features):** Distorts univariate variance, creates artificial point-mass spikes in distributions, and ignores inter-attribute correlations.
* **K-Nearest Neighbors (KNN) Imputation (Adopted for Continuous Features):** Imputes missing feature $x_{i,j}$ by taking the distance-weighted mean of its $K$ nearest neighbors ($K=5$) in Euclidean feature space:
  $$\hat{x}_{i,j} = \frac{\sum_{k \in N_K(i)} w_k \cdot x_{k,j}}{\sum_{k \in N_K(i)} w_k}, \quad w_k = \frac{1}{d(x_i, x_k)}$$
* **Multivariate Imputation by Chained Equations - MICE / IterativeImputer (Adopted for Mixed Features):** Models each feature with missing values as a function of all other features in an iterative round-robin sequence using Bayesian Ridge Regression.
* **Most Frequent / Mode Imputation (Adopted for Categorical Features):** Used for discrete nominal attributes in CKD (e.g., `rbc`, `pc`, `htn`).

---

## 3. Outlier Detection Paradigm Comparison

TABLE I: IEEE-FORMATTED OUTLIER DETECTION METHODOLOGY COMPARISON

| Method | Mathematical Principle | Assumptions | Strengths in Healthcare Data | Limitations | Suitability Ranking |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Interquartile Range (IQR)** | Parametric range filtering: $[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$. | Univariate distribution; symmetric data spread. | Simple, computationally fast, highly interpretable. | Evaluates features independently; drops valid extreme clinical cases (e.g., acute liver failure spikes). | Moderate (3rd) |
| **Z-Score Thresholding** | Standardized distance: $Z = \frac{x - \mu}{\sigma}$, flag if $\|Z\| > 3.0$. | Gaussian (Normal) feature distribution. | Standard parametric metric for bell-shaped attributes. | Severely biased by heavy skewness and pre-existing outliers distorting $\mu$ and $\sigma$. | Low (4th) |
| **Local Outlier Factor (LOF)** | Density-based local reachability distance relative to $k$-nearest neighbors. | Non-uniform density variations across cluster manifolds. | Effective at discovering isolated low-density spatial clusters. | High memory complexity $\mathcal{O}(N^2)$; difficult to apply inductively to unseen test points. | High (2nd) |
| **Isolation Forest (iForest)** | Unsupervised tree-based anomaly isolation via random recursive space partitioning. | Outliers are few and possess anomalous attribute values. | **Multivariate; scale-invariant; handles complex non-linear attribute interactions; fast $\mathcal{O}(N \log N)$ complexity.** | Requires hyperparameter tuning of contamination factor ($\alpha$). | **Most Suitable (1st - Selected)** |

### 3.1 Selection Justification for Isolation Forest
Isolation Forest is selected as the primary multivariate outlier detection mechanism. In medical diagnostic modeling, acute pathological conditions (e.g., severe diabetic ketoacidosis or acute renal crisis) manifest as rare, combined multivariate attribute extremes rather than isolated single-feature anomalies. Isolation Forest isolates such anomalies using random decision trees without assuming normal distributions. A conservative contamination parameter ($\alpha = 0.03$) is enforced to remove severe measurement artifacts while retaining valid pathological manifestations.

---

## 4. Categorical Feature Encoding Strategy

TABLE II: IEEE-FORMATTED CATEGORICAL ENCODING STRATEGY COMPARISON

| Encoding Technique | Mathematical Representation | Target Feature Types | Advantages | Limitations | Operational Usage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Label / Binary Encoding** | Maps categories to integer values: $c_k \to k \in \{0, 1\}$. | Binary attributes (e.g., Sex: Male=1, Female=0). | Zero dimensionality expansion; memory efficient. | Imposes implicit ordinal ordering if applied to nominal multi-class features. | **Applied to all binary clinical features.** |
| **One-Hot Encoding (OHE)** | Maps $K$ categories to $K$ (or $K-1$) binary vector dimensions: $c_k \to [0, \dots, 1, \dots, 0]^T$. | Unordered Nominal features (e.g., Chest Pain Type `cp` $1-4$, Thalassemia `thal`). | Preserves categorical independence; no false ordering assumptions. | Increases dimensionality (curse of dimensionality for high-cardinality features). | **Applied to nominal features with $K > 2$.** |
| **Ordinal Encoding** | Maps categories to rank-ordered integers: $c_k \to r_k \in \{1, 2, \dots, K\}$. | Natural Ordinal features (e.g., ST Slope: Up=1, Flat=2, Down=3; Urinalysis Albumin $0-5$). | Preserves natural clinical rank hierarchy. | Requires explicit domain-informed rank mapping. | **Applied strictly to ordinal clinical attributes.** |

---

## 5. Feature Rescaling Paradigm Comparison

TABLE III: IEEE-FORMATTED FEATURE SCALING METHODOLOGY COMPARISON

| Rescaling Algorithm | Mathematical Transformation Formula | Robustness to Outliers | Output Bounded Range | Impact on Tree Ensembles vs. Neural Models | Selected Operational Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **StandardScaler (Z-score)** | $x' = \frac{x - \mu}{\sigma}$ | Low (Outliers distort sample mean $\mu$ and standard deviation $\sigma$). | Unbounded ($\approx [-3, +3]$ for Gaussian). | Zero effect on Tree Ensembles; critical for Gradient Descent in MLPs & SVM distance metrics. | **Selected for Gaussian/Symmetric feature subsets (e.g., WBCD).** |
| **MinMaxScaler** | $x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$ | Extreme Sensitivity (Outliers collapse non-outlier feature range to tight cluster near 0 or 1). | Fixed Range $[0, 1]$ | Preserves zero entries in sparse vectors; fragile under extreme pathological biomarker spikes. | **Secondary usage for non-skewed bounded attributes.** |
| **RobustScaler (IQR-based)** | $x' = \frac{x - \text{Median}(X)}{\text{IQR}(X)}$ | **High (Uses median and Interquartile Range $Q_3 - Q_1$, resisting extreme outliers).** | Unbounded (Centered at 0, unit interquartile spread). | **Optimal for skewed biomedical data (e.g., Insulin, Bilirubin, ALT/AST, Creatinine).** | **Default Scaler for Non-Gaussian Clinical Features.** |

---

## 6. Experimental Validation, Leakage Prevention, & Workflow Architecture

### 6.1 Stratified Train-Validation-Test Partitioning Scheme
To guarantee robust empirical evaluation, datasets are partitioned into two strict operational sets:
1. **Holdout Test Set (20%):** Sealed completely during data exploration, preprocessing parameter estimation, feature selection, and model tuning. Used exclusively for final performance reporting.
2. **Training & Validation Set (80%):** Utilized for model training and hyperparameter optimization under **Stratified 10-Fold Cross-Validation**.

### 6.2 Data Leakage Prevention Protocol
Data leakage occurs when information from outside the training partition influences model training, leading to overly optimistic performance estimates that fail in real-world clinical deployment. The proposed pipeline enforces strict leakage barriers:

```
========================= LEAKAGE PREVENTION BARRIER =========================

  [ Full Dataset (100%) ] 
             |
             +---> [ Split Strategy: Stratified 80/20 Partition ]
                         |
                         +---> [ Holdout Test Set (20%) ] ---> (SEALED UNTIL FINAL TEST)
                         |
                         +---> [ Training Folds (80%) ]
                                     |
                                     +---> 1. Estimate KNN Imputer Parameters (Fit)
                                     +---> 2. Estimate RobustScaler Median/IQR (Fit)
                                     +---> 3. Compute AFO Feature Importance (Fit)
                                     +---> 4. Apply Borderline-SMOTE (Training Fold ONLY)
                                     |
                                     v
                        [ Apply Learned Transformations (Transform ONLY) ]
                         ---> Applied to Validation Folds & Sealed Test Set
```

### 6.3 Class Imbalance Resampling Strategy
To resolve class imbalance (particularly in ILPD and CKD datasets) without generating unrealistic synthetic points:
* **Borderline-SMOTE (Over-sampling):** Applied **strictly within training folds** to generate synthetic minority samples near class decision boundaries rather than deep inside cluster interiors.
* **Tomek Links (Cleaning):** Removes ambiguous overlapping majority-minority instance pairs along the decision boundary post-SMOTE to ensure clean margin separation.

### 6.4 Computational Reproducibility Standard
To guarantee 100% scientific reproducibility across all execution environments:
* A global pseudorandom seed state ($\text{SEED} = 42$) is explicitly configured across `NumPy`, `Scikit-Learn`, `PyTorch`, `XGBoost`, `LightGBM`, and `CatBoost` execution contexts.
