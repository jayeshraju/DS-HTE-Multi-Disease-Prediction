# IEEE/Scopus Optimization Roadmap: Explanable Multi-Disease Prediction Framework
**A Publication-Grade Optimization Strategy and Root-Cause Diagnosis**

---

## Abstract
This document outlines the systematic, end-to-end optimization of the proposed predictive framework: *"Explainable Multi-Disease Prediction Framework Using Hybrid Transformer-Based Deep Learning and Adaptive Feature Optimization."* To ensure compliance with IEEE Transactions on Neural Networks and Learning Systems (TNNLS), IEEE Journal of Biomedical and Health Informatics (JBHI), and Scopus-indexed clinical venues, this roadmap diagnoses current performance bottlenecks, redesigns data and model pipelines, establishes rigorous statistical validation, and outlines the transition from a rigid uniform hybrid model to a **disease-adaptive clinical framework**.

---

## Phase 1 – Root-Cause Analysis and Diagnostic Analysis

```
+------------------------------------------------------------------------------------------------------+
|                                    PHASE 1: DIAGNOSTIC LOGIC MATRIX                                  |
+------------------------------------+----------------------------------+------------------------------+
| Dataset & Symptom                  | Machine Learning Cause           | Clinical Cause               |
+------------------------------------+----------------------------------+------------------------------+
| Cleveland Heart Disease            | • Dimensionality inflation       | • Failure to represent blood |
| (Recall & F1 Drop vs Baseline)     | • Collinearity in augmented space|   pressure/cholesterol       |
|                                    | • Overfitting on sparse samples  |   interaction thresholds     |
+------------------------------------+----------------------------------+------------------------------+
| Indian Liver Patient (ILPD)        | • Dual over-balancing penalty    | • Severe outlier pruning     |
| (Recall & F1 Drop; MCC = 0.0)      | • Suboptimal threshold (Youden J)|   removes pathological crises|
|                                    | • High noise-to-signal ratio     | • Bilirubin/transaminase lag |
+------------------------------------+----------------------------------+------------------------------+
```

### 1.1 Cleveland Heart Disease Cohort Diagnosis
* **Current Behavior:** Decreased overall accuracy, $F_1$-score, Matthews Correlation Coefficient (MCC), and Specificity compared to a standard XGBoost baseline, while Recall (Sensitivity) shows marginal, unstable changes.
* **Machine Learning Root Causes:**
  1. **Dimensionality Inflation vs. Sample Sparsity:** The Cleveland Heart dataset is small ($N = 303$, with only $242$ training samples in a $5$-fold split). Projecting $13$ raw features into a high-dimensional transformer token space ($d_{\text{token}} \in [64, 128]$) and extracting a $[CLS]$ embedding results in an augmented feature space of $D \in [77, 141]$ dimensions. Training a GBDT on this inflated feature space leads to severe overfitting ("curse of dimensionality").
  2. **Collinearity in Augmented Space:** The FT-Transformer classifier head is trained using supervised Binary Cross-Entropy (BCE) loss on training targets. The extracted $[CLS]$ embedding is thus highly collinear with the target probability. The downstream GBDT over-relies on these embeddings, leading to generalization failure on unseen test partitions.
  3. **Inadequate Convergence of Attention Heads:** The FT-Transformer requires a large number of training epochs to align attention maps. When limited to few epochs, the self-attention heads output noisy, uninformative latent representations that dilute the clean signal of the raw clinical features.
* **Clinical Root Causes:**
  1. **Disruption of Clinical Risk Thresholds:** In cardiovascular medicine, risk is defined by non-linear thresholds (e.g., Systolic Blood Pressure $> 140$ mmHg paired with Cholesterol $> 240$ mg/dL). High-dimensional projections smudge these distinct diagnostic boundaries, making it difficult for the downstream tree nodes to isolate patients in acute hypertensive or hypercholesterolemic crises.

### 1.2 Indian Liver Patient Dataset (ILPD) Cohort Diagnosis
* **Current Behavior:** High sensitivity (often defaulting to $100\%$), but Specificity collapses to $0.00\%$, resulting in an MCC of $0.00$.
* **Machine Learning Root Causes:**
  1. **Double-Correction Imbalance Penalty:** ILPD exhibits a $70:30$ class imbalance (liver disease vs. control). Applying Borderline-SMOTE oversampling *and* applying inverse-frequency class weights to the downstream GBDT results in an over-correction error. The loss function heavily penalizes errors on the minority class, forcing the decision boundary away from the positive instances and classifying almost all patients as diseased.
  2. **Severe Outlier Pruning:** The consensus outlier screening (Isolation Forest + LOF) removes $4\%$ of the samples. In clinical data, extreme transaminase and bilirubin values represent the most critical pathological cases. Pruning them removes the signature of advanced liver disease, directly hurting the model's ability to learn real boundaries.
  3. **Suboptimal Thresholding:** Youden's $J$ thresholding seeks to maximize $\text{Sensitivity} + \text{Specificity} - 1$. On highly noisy test distributions, Youden's $J$ can select an overly conservative threshold, sacrificing Specificity and MCC.
* **Clinical Root Causes:**
  1. **Pathological Overlap in Early-Stage Liver Disease:** Early liver damage displays clinical signatures (e.g., mild elevations in ALT/AST) that overlap almost entirely with healthy controls or minor systemic inflammation. The standard linear scaling fails to capture the De Ritis ratio ($\text{AST}/\text{ALT}$), which is the primary marker used by hepatologists to distinguish alcoholic from non-alcoholic liver diseases.

---

## Phase 2 – Advanced Data Preprocessing

To guarantee reproducibility and eliminate data leakage, we propose a rigorous, disease-specific preprocessing framework.

```
+---------------------------------------------------------------------------------------------+
|                                  PHASE 2: PREPROCESSING FLOW                                |
+---------------------------------------------------------------------------------------------+
| [Raw Clinical Data] ---> [Shadow Missing Indicators] ---> [MissForest Imputation]           |
|                                                                     |                       |
| [Robust scaling / GMM] <--- [Clinical Winsorization] <--- [Leakage-Free Stratified Split]   |
+---------------------------------------------------------------------------------------------+
```

### 2.1 Imputation Analysis
* **Median Imputation:** Introduces severe bias by collapsing the variance of clinical distributions. Unacceptable for publication.
* **KNN Imputer:** Sensitive to scale and distance metrics; struggles with mixed-type clinical data (e.g., categorical gender alongside continuous creatinine).
* **MICE (IterativeImputer):** Captures linear relationships, but fails to model non-linear interaction paths.
* **MissForest (Recommended):** A non-parametric, random forest-based imputer. Iteratively trains forest estimators on observed values to impute missing entries. It handles mixed-type variables, requires no parametric distribution assumptions, and preserves non-linear physiological interactions.

### 2.2 Outlier Handling
* **Consensus Pruning (LOF + IForest):** Deletes clinical samples. This is highly problematic as extreme outliers (e.g., Blood Glucose $> 400$ mg/dL) represent the most critical clinical events.
* **Pathology-Preserving Winsorization (Recommended):** Clips values to the $1\text{st}$ and $99\text{th}$ percentiles:
  $$\tilde{x} = \max(P_1, \min(x, P_{99}))$$
  This bounds extreme noise (e.g., measurement or entry errors) while retaining the presence of pathological crises.

### 2.3 Scaling and Normalization
* **StandardScaler:** Distorts distributions when features are heavily skewed or multi-modal.
* **RobustScaler (Recommended for Uni-modal Skewed Features):** Centers data using the median and scales via Interquartile Range (IQR):
  $$x_{\text{robust}} = \frac{x - \text{median}(x)}{\text{IQR}(x)}$$
* **Gaussian Mixture Model (GMM) scaling (Recommended for Bimodal Features):** For features with distinct healthy and diseased peaks (e.g., Glucose in PIMA), we fit a 2-component GMM:
  $$P(x) = \pi_1 \mathcal{N}(x; \mu_1, \sigma_1^2) + \pi_2 \mathcal{N}(x; \mu_2, \sigma_2^2)$$
  The features are transformed into the probability density of belonging to the pathological sub-population component.

### 2.4 Recommended Preprocessing Strategies
```
+------------------------------------------------------------------------------------------------------+
|                              RECOMMENDED PREPROCESSING STRATEGY PER DATASET                          |
+----------------------+--------------------+--------------------+-------------------------------------+
| Dataset              | Imputation         | Outlier Strategy   | Scaling Strategy                    |
+----------------------+--------------------+--------------------+-------------------------------------+
| PIMA Diabetes        | MissForest         | Winsorization (99%)| GMM (Glucose, BMI), Robust (others) |
| Cleveland Heart      | KNN (K=5)          | None (Keep all)    | StandardScaler (uniform variance)   |
| CKD (Kidney)         | MissForest         | Winsorization (99%)| RobustScaler                        |
| ILPD (Liver)         | MissForest         | Winsorization (95%)| RobustScaler (ALT, AST, Bilirubin)  |
| WBCD (Cancer)        | None (Complete)    | Winsorization (99%)| StandardScaler                      |
+----------------------+--------------------+--------------------+-------------------------------------+
```

---

## Phase 3 – Feature Engineering

Continuous clinical variables are governed by underlying physiological processes. Introducing explicit domain-specific non-linear features reduces the reliance on self-attention layers to discover fundamental clinical laws.

### 3.1 Domain-Specific Clinical Features
* **Diabetes (PIMA):**
  * **HOMA-IR Proxy:** Homeostatic Model Assessment of Insulin Resistance:
    $$\text{HOMA-IR} = \frac{\text{Glucose (mg/dL)} \times \text{Insulin (\mu IU/mL)}}{405}$$
    *Clinical Justification:* Standard diagnostic proxy for insulin resistance, decoupling glucose levels from insulin production.
  * **Metabolic Overload Index (MOI):**
    $$\text{MOI} = \text{Glucose} \times \text{BMI}$$
    *Clinical Justification:* Models the joint risk of adiposity and circulating plasma glucose on vascular endothelium.
* **Heart Disease (Cleveland):**
  * **Rate Pressure Product (RPP):**
    $$\text{RPP} = \text{thalach (Max HR)} \times \text{trestbps (Resting BP)}$$
    *Clinical Justification:* Serves as a direct proxy for myocardial oxygen consumption and cardiac workload.
  * **Vascular Load Coefficient (VLC):**
    $$\text{VLC} = \frac{\text{age} \times \text{trestbps}}{100}$$
* **Kidney Disease (CKD):**
  * **Glomerular Filtration Rate (eGFR) Approximation:**
    $$\text{eGFR}_{\text{approx}} = \frac{141 \times \min(\text{sc}/\kappa, 1)^\alpha \times \max(\text{sc}/\kappa, 1)^{-1.209} \times 0.993^{\text{age}}}{\text{sc}}$$
    *Clinical Justification:* Captures serum creatinine excretion relative to demographic decline.
* **Liver Disease (ILPD):**
  * **De Ritis Ratio:**
    $$\text{De Ritis} = \frac{\text{Aspartate Aminotransferase (AST)}}{\text{Alamine Aminotransferase (ALT)}}$$
    *Clinical Justification:* Values $> 2.0$ indicate alcoholic liver disease or severe necrosis; values $< 1.0$ suggest non-alcoholic fatty liver disease.
  * **Bilirubin Ratio:**
    $$\text{Bilirubin Ratio} = \frac{\text{Direct Bilirubin}}{\text{Total Bilirubin}}$$
    *Clinical Justification:* Distinguishes obstructive (conjugation-completed) jaundice from hemolytic conditions.

---

## Phase 4 – Adaptive Feature Optimization (AFO)

A unified feature selection strategy fails because clinical signatures vary in sparse vs. dense spaces. We propose a localized **Borda Count Consensus Pipeline** utilizing three distinct evaluation paradigms.

```
   +-------------------------------------------------------------------+
   |                    ADAPTIVE FEATURE OPTIMIZATION                  |
   +-------------------------------------------------------------------+
                                     |
         +---------------------------+---------------------------+
         |                           |                           |
   [Information Metric]       [Regularized Linear]     [Tree-Based Wrappers]
    • Mutual Information       • LASSO (L1 penalty)     • CatBoost Importance
         |                           |                           |
         +---------------------------+---------------------------+
                                     |
                               [Borda Rank]
                                     |
                           [SHAP Validation Loop]
                                     |
                        [Optimal Feature Subspace]
```

### 4.1 Comparison of Feature Selection Paradigms
* **Mutual Information (MI):** Captures non-linear dependencies but evaluates features individually, ignoring collinearity.
* **LASSO:** Effectively eliminates redundant features via $L_1$ regularization, but forces arbitrary selection among highly correlated clinical variables.
* **Tree-Based Importances:** CatBoost and XGBoost evaluate split gains, capturing complex feature interactions but biasing toward high-cardinality continuous variables.
* **PCA / Autoencoder Embeddings:** Excellent for dimensionality reduction, but project features into uninterpretable latent spaces, which violates medical explainability requirements.

### 4.2 AFO Consensus Selection Algorithm
For each dataset, AFO runs a validation loop:
1. **Compute Ranks:** Compute importance rankings from:
   * (a) Mutual Information,
   * (b) LASSO coefficients,
   * (c) CatBoost feature importances.
2. **Borda Count Consensus:** Aggregate ranks using a weighted Borda score:
   $$\text{Score}(f) = w_1 R_{\text{MI}}(f) + w_2 R_{\text{LASSO}}(f) + w_3 R_{\text{CatBoost}}(f)$$
3. **SHAP-Guided Validation Loop:** Start with the top 5 consensus features, recursively add features from the ranked list, and evaluate the out-of-fold validation log-loss. Retain the feature set that minimizes the log-loss.

---

## Phase 5 – Class Imbalance Optimization

```
+------------------------------------------------------------------------------------------------------+
|                                    CLASS IBALANCE MITIGATION PARADIGMS                               |
+----------------------+-------------------------------------------------+-----------------------------+
| Dataset              | Strategy                                        | Clinical Justification      |
+----------------------+-------------------------------------------------+-----------------------------+
| PIMA Diabetes        | ADASYN (Applied strictly on raw features only)  | Synthesizes samples in      |
|                      |                                                 | low-density boundary zones  |
+----------------------+-------------------------------------------------+-----------------------------+
| Cleveland Heart      | None (Balanced Class Ratio ~ 54:46)             | Oversampling introduces     |
|                      |                                                 | artificial clinical noise   |
+----------------------+-------------------------------------------------+-----------------------------+
| CKD (Kidney)         | None (Classes highly separable)                 | Highly distinct clinical    |
|                      |                                                 | profiles; risk of leakage   |
+----------------------+-------------------------------------------------+-----------------------------+
| ILPD (Liver)         | Cost-Sensitive Learning (Inverse Frequency)     | Avoids corrupting highly    |
|                      | No synthetic generation                         | overlapping diagnostic spaces|
+----------------------+-------------------------------------------------+-----------------------------+
| WBCD (Cancer)        | None (Balanced Class Ratio ~ 63:37)             | High baseline separability  |
+----------------------+-------------------------------------------------+-----------------------------+
```

### 5.1 Why Uniform Synthetic Generation Fails
Applying SMOTE or SMOTE-ENN uniformly across all datasets violates clinical dataset topology:
1. **Interpolation in Latent Spaces:** Generating synthetic samples in the augmented $[CLS]$ feature space creates unrealistic patient profiles that violate clinical logic. **Oversampling must be restricted to raw features prior to tokenization.**
2. **Over-balancing Penalty:** In ILPD, combining Borderline-SMOTE with CatBoost class-weights shifts the decision threshold too far, degrading Specificity and MCC. For overlapping clinical features, cost-sensitive learning (using class weights) is superior to synthetic interpolation.

---

## Phase 6 – Model Optimization (Backbone Selection)

To build a publication-grade framework, we must address the limitations of forcing a tabular transformer onto low-dimensional, small-sample datasets.

### 6.1 Evaluation of Models on Tabular Data
* **Logistic Regression:** Stable baseline but fails to capture complex non-linear diagnostic rules.
* **XGBoost / LightGBM:** Fast, robust to outliers, and highly effective on small datasets, but struggle with smooth continuous boundary transitions.
* **CatBoost:** The best-performing tree baseline. Symmetric trees provide excellent regularization on small datasets, preventing overfitting on noisy medical records.
* **FT-Transformer:** Maps features to dense embeddings and processes them via self-attention. Excellent at capturing complex feature interactions on large datasets ($N > 5000$). However, on small datasets ($N < 1000$), it suffers from severe overfitting and slow convergence.

### 6.2 Proposed Disease-Aware Hybrid Architecture (DAHA)
Instead of a uniform architecture, we introduce a **Disease-Aware Hybrid Framework** that dynamically configures the classification backbone based on data density and sample size:

```
                            [Clinical Input Query]
                                      |
                         +------------+------------+
                         |                         |
                 (Samples < 1000)          (Samples >= 1000)
                 (Features < 15)           (Features >= 15)
                         |                         |
                         v                         v
               [Lightweight GBDT]         [Hybrid FT-Transformer]
                • Pure CatBoost            • [CLS] Embeddings Extraction
                • Target Encoding          • Downstream Stacking Ensemble
```

* **Low-Sample, Low-Dimensional Cohorts (PIMA, Cleveland, ILPD):** Configured as a direct GBDT (CatBoost/XGBoost) model. This avoids parameter inflation and target leakage, stabilizing metrics like MCC and Specificity.
* **High-Sample or High-Dimensional Cohorts (WBCD, CKD):** Configured as the Hybrid FT-Transformer + Stacking Ensemble, leveraging the transformer's ability to extract complex multi-feature interactions.

---

## Phase 7 – Hybrid Architecture Optimization

For cohorts leveraging the hybrid setup, the interface between the FT-Transformer and the downstream Gradient Boosted Trees must be regularized.

```
+---------------------------------------------------------------------------------------------+
|                               PHASE 7: ENSEMBLE METRICS ROUTING                             |
+---------------------------------------------------------------------------------------------+
| [Raw Features] ---> [FT-Transformer Encoder] ---> [CLS Token] ---> [Bottleneck Projection]  |
|                                                                             |               |
| [Logistic Regression Meta-Learner] <--- [Stacking GBDT Blend] <-------------+               |
+---------------------------------------------------------------------------------------------+
```

### 7.1 Key Architecture Enhancements
1. **Bottleneck Projection Layer:** The high-dimensional $[CLS]$ token ($d_{\text{token}} = 128$) is passed through a linear bottleneck layer to project it down to $d_{\text{proj}} = 8$. This preserves the consolidated attention representation while preventing feature space inflation.
2. **Target Leakage Mitigation:** The FT-Transformer is trained using a self-supervised reconstruction loss (reconstructing masked features) rather than supervised BCE loss. This ensures the $[CLS]$ token captures general clinical structures without target leakage.
3. **Heterogeneous GBDT Stacking:** The bottleneck $[CLS]$ token is concatenated with the raw features. A heterogeneous ensemble (CatBoost, XGBoost, and LightGBM) is trained on this joint space, with a Ridge Classifier serving as the meta-learner.

---

## Phase 8 – Hyperparameter Optimization

We implement an Optuna-based Bayesian optimization strategy using a composite fitness function designed for clinical applications.

### 8.1 Multi-Objective Composite Fitness Function
To prevent optimization from producing models with high F1-scores but poor calibration, we define the fitness function as:
$$\text{Fitness} = 0.4 \times F_1 + 0.3 \times \text{MCC} + 0.3 \times (1.0 - \text{ECE})$$
where $\text{ECE}$ is the Expected Calibration Error. This balances accuracy, performance under class imbalance, and probability reliability.

### 8.2 Optuna Search Space
```
+------------------------------------------------------------------------------------------------------+
|                                   HYPERPARAMETER SEARCH CONFIGURATION                                |
+----------------------+----------------------+----------------------+---------------------------------+
| Component            | Parameter            | Type / Range         | Optimization Purpose            |
+----------------------+----------------------+----------------------+---------------------------------+
| FT-Transformer       | d_token              | Categorical: 32, 64  | Prevents dimensionality inflation|
|                      | n_blocks             | Integer: 1 to 3      | Controls network capacity       |
|                      | learning_rate        | Float: 1e-4 to 1e-3  | Stabilizes optimization steps   |
|                      | weight_decay         | Float: 1e-4 to 1e-2  | Controls overfitting            |
+----------------------+----------------------+----------------------+---------------------------------+
| CatBoost / GBDT      | depth                | Integer: 4 to 6      | Prevents overfitting on small CV|
|                      | l2_leaf_reg          | Float: 1.0 to 10.0   | Regularizes tree structures     |
|                      | learning_rate        | Float: 0.01 to 0.05  | Ensures stable gradient steps   |
+----------------------+----------------------+----------------------+---------------------------------+
```

---

## Phase 9 – Decision Threshold Optimization

Using a default threshold of $0.5$ is clinically inappropriate. We optimize decision thresholds to align with clinical goals, balancing the risk of false negatives against false positives.

### 9.1 F-Beta Threshold Calibration
We optimize the decision threshold to maximize the $F_{\beta}$ score:
$$F_{\beta} = (1 + \beta^2) \frac{\text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}$$
* **Liver & Heart Cohorts ($\beta = 1.5$):** Biases the threshold toward Recall, reducing the risk of missing active pathology.
* **Cancer & CKD Cohorts ($\beta = 1.0$):** Optimizes the standard F1-score to balance sensitivity and precision.

### 9.2 Clinical Cost Matrix Optimization
We define a clinical utility function:
$$\text{Utility} = C_{\text{TP}} \times \text{TP} - C_{\text{FN}} \times \text{FN} - C_{\text{FP}} \times \text{FP}$$
Based on clinical consensus, the cost of a False Negative ($C_{\text{FN}}$) is set to 5 times the cost of a False Positive ($C_{\text{FP}}$). The decision threshold is selected to maximize this utility function on validation folds.

---

## Phase 10 – Explainable AI (XAI) Enhancement

Explainability must be clinically valid and expressed in terms of raw physiological features.

### 10.1 Back-Propagated Latent SHAP
Because the downstream GBDT is trained on the augmented space ($X_{\text{raw}} + [CLS]$), standard SHAP values are calculated for abstract latent dimensions. We map these latent attributions back to the raw input features using a gradient-based scaling approach:
$$\text{Attribution}(x_i) = \text{SHAP}(x_i) + \sum_{j=1}^{d_{\text{proj}}} \text{SHAP}(z_j) \times \left| \frac{\partial z_j}{\partial x_i} \right|$$
This ensures the final visualizations present the contribution of each raw clinical feature.

### 10.2 Contrastive Explainability (SHAP vs LIME)
* **SHAP (Global consistency):** Used to compute consistent global feature importances across cohorts.
* **LIME (Local fidelity):** Used to generate patient-specific risk explanations, explaining how individual risk factors contribute to a patient's risk score.

---

## Phase 11 – Experimental Validation

To meet IEEE publication standards, we implement a rigorous validation protocol to confirm that performance improvements are statistically significant.

### 11.1 Validation Design: 5x10 Repeated Nested Stratified CV
We use a **5x10 Repeated Nested Stratified Cross-Validation** protocol:
* **Outer Loop:** 5-fold stratified splits repeated 10 times with different random seeds. This produces 50 outer folds, allowing us to compute tight confidence intervals.
* **Inner Loop:** 3-fold stratified splits for hyperparameter tuning and threshold calibration, preventing data leakage.

### 11.2 Statistical Hypothesis Testing
```
+------------------------------------------------------------------------------------------------------+
|                                    STATISTICAL SIGNIFICANCE TESTS                                    |
+----------------------+---------------------------------+---------------------------------------------+
| Test                 | Mathematical Purpose            | Clinical Verification Goal                  |
+----------------------+---------------------------------+---------------------------------------------+
| Wilcoxon Signed-Rank | Non-parametric paired t-test on | Confirms that overall accuracy gains are    |
|                      | metric scores across 50 folds   | consistent and not due to random splits     |
+----------------------+---------------------------------+---------------------------------------------+
| DeLong's Test        | Compares differences in ROC-AUC | Proves the hybrid model's discriminative    |
|                      | curves                          | power is statistically superior to baseline |
+----------------------+---------------------------------+---------------------------------------------+
| McNemar's Test       | Evaluates differences in the    | Confirms that the reduction in clinical     |
|                      | classification confusion matrix | classification errors is significant        |
+----------------------+---------------------------------+---------------------------------------------+
```

---

## Phase 12 – Ablation Study

An ablation study is required to isolate and demonstrate the contribution of each module in our framework.

```
+------------------------------------------------------------------------------------------------------+
|                                    HIERARCHICAL ABLATION CONFIGURATIONS                              |
+----------------------+--------------------+--------------------+-------------------------------------+
| Configuration        | Preprocessing      | Feature Space      | Model Architecture                  |
+----------------------+--------------------+--------------------+-------------------------------------+
| M1 (Base GBDT)       | Standard           | Raw Features       | Single CatBoost (Baseline)          |
+----------------------+--------------------+--------------------+-------------------------------------+
| M2 (Engineered GBDT) | Standard           | Raw + Clinical     | Single CatBoost                     |
+----------------------+--------------------+--------------------+-------------------------------------+
| M3 (Base Hybrid)     | Standard           | Raw + [CLS]        | FT-Transformer + CatBoost           |
+----------------------+--------------------+--------------------+-------------------------------------+
| M4 (Consensus Hybrid)| Optimized          | Raw + AFO + [CLS]  | FT-Transformer + CatBoost           |
+----------------------+--------------------+--------------------+-------------------------------------+
| M5 (DAHA - Final)    | Optimized          | Optimized + AFO    | Disease-Aware Hybrid / Stacking     |
+----------------------+--------------------+--------------------+-------------------------------------+
```

---

## Phase 13 – Computational Analysis

To assess the deployment feasibility of the framework, we track its computational profile:
1. **Training Time (seconds):** Total time to execute the repeated cross-validation loop.
2. **Inference Latency (ms):** Average time to compute a single patient prediction.
3. **Memory Footprint (MB):** Peak RAM and VRAM usage.
4. **Parameter Efficiency:** The ratio of classification performance (MCC) to total model parameters.

---

## Phase 14 – Publication Readiness Review

### Remaining Weaknesses & Solutions
* **Weakness:** Overfitting on low-dimensional datasets.
  * *Solution:* Implement the Disease-Aware Hybrid Framework (DAHA) to bypass the transformer block for small datasets.
* **Weakness:** Lacking statistical significance tests.
  * *Solution:* Integrate DeLong's test and Wilcoxon signed-rank tests into the validation pipeline.
* **Weakness:** Diagnostic targets are treated as general regression targets.
  * *Solution:* Align all reporting with the **TRIPOD** (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) guidelines.

---

## Final Deliverables

### 1. Ranked List of Improvements

```
Rank 1: Disease-Aware Hybrid Framework (DAHA)
        • Impact: High (Resolves small-sample overfitting in PIMA/ILPD)
        • Cost: Low

Rank 2: Clinically Driven Feature Engineering
        • Impact: High (Introduces non-linear physiological relationships)
        • Cost: Low

Rank 3: Dynamic Threshold & Calibration Optimization
        • Impact: High (Resolves Specificity collapses and MCC errors)
        • Cost: Low

Rank 4: 5x10 Repeated Nested Stratified Cross-Validation
        • Impact: High (Establishes statistical validity for publication)
        • Cost: High (Increases computation time)

Rank 5: Back-Propagated Latent SHAP Explainability
        • Impact: Medium (Improves interpretability for reviewers)
        • Cost: Medium
```

---

### 2. Optimization Decision Tree

```
                      Is Validation MCC / F1 Acceptable?
                                      |
                     +----------------+----------------+
                     | Yes                             | No
                     v                                 v
         Is calibration error high?          Is the dataset N < 1000?
                     |                                 |
           +---------+---------+             +---------+---------+
           | Yes               | No          | Yes               | No
           v                   v             v                   v
     [Apply Beta         [Generate final  [Bypass FT-Trans    [Optimize AFO
     Calibration]         plots & paper]   Use GBDT only]      consensus ranks]
```

---

### 3. Publication Readiness Checklist
- [ ] Align reporting with the **TRIPOD** guidelines.
- [ ] Plot ROC and Precision-Recall curves with 95% confidence intervals shaded.
- [ ] Report p-values for all comparisons using DeLong's test ($p < 0.05$).
- [ ] Export all plots as vector graphics (.eps or .pdf).
- [ ] Document inference latency and GPU/CPU training times.
- [ ] Provide open-source access to code and sanitization scripts.

---

### 4. Final Recommendation
The framework should evolve into a **Disease-Aware Hybrid Architecture (DAHA)**. Forcing a tabular transformer onto small, low-dimensional datasets (like PIMA Diabetes and Cleveland Heart) leads to overfitting and target leakage. The framework should dynamically route tasks, using the hybrid transformer stacking ensemble for high-dimensional cohorts (WBCD, CKD) and optimized GBDT models for low-dimensional, sparse cohorts (Diabetes, Heart, Liver). This adaptive design improves classification performance and provides a solid foundation for academic publication.
