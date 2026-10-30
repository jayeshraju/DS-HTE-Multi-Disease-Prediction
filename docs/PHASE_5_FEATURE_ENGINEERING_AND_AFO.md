# Phase 5: Feature Engineering & Adaptive Feature Optimization (AFO) — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Scalable Multi-Disease Early Prediction and Risk Stratification Framework using Explainable Artificial Intelligence and Adaptive Feature Optimization
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Feature Engineering & Clinical Interaction Framework

Feature engineering transforms raw clinical measurement variables into enriched representation spaces. In biomedical informatics, feature construction must balance statistical expressiveness with pathophysiological validity.

### 1.1 Clinical Feature Transformation & Interaction Synthesis
1. **Domain-Driven Ratio Biomarkers:**
   * **Bilirubin Ratio (ILPD):** $\text{Bilirubin\_Ratio} = \frac{\text{Direct\_Bilirubin}}{\text{Total\_Bilirubin}}$. Measures the proportion of conjugated bilirubin; elevated ratios indicate biliary tract obstruction rather than hemolytic anemia.
   * **eGFR Proxy (CKD):** $\text{Creatinine\_Urea\_Ratio} = \frac{\text{Serum\_Creatinine}}{\text{Blood\_Urea}}$. Indicates prerenal vs. intrinsic renal failure.
   * **Glucose-BMI Risk Index (PIMA):** $\text{Metabolic\_Index} = \text{Glucose} \times \text{BMI}$. Captures synergistic metabolic pressure where obesity amplifies insulin resistance.
   * **Pulse Pressure Index (Cleveland/CKD):** $\text{Pulse\_Pressure} = \text{Systolic BP} - \text{Diastolic BP}$. Serves as a surrogate for arterial stiffness and vascular aging.

2. **Polynomial Feature Expansion:**
   * Second-order non-linear interaction terms ($x_i \cdot x_j$ and $x_i^2$) are synthesized selectively across top continuous discriminators (e.g., $\text{Glucose}^2$, $\text{BMI}^2$, $\text{thalach} \cdot \text{oldpeak}$) to provide linear models and neural networks explicit access to non-linear biological response surfaces.

3. **Autoencoder Latent Feature Embeddings:**
   * An unsupervised **Denoising Autoencoder (DAE)** compresses high-dimensional continuous features (specifically the 30 morphological features of WBCD and 24 attributes of CKD) into a compact $k$-dimensional bottleneck embedding ($k=8$).
   * **Loss Function:** Reconstructs uncorrupted input $\mathbf{x}$ from corrupted input $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\epsilon}$:
     $$\mathcal{L}_{\text{DAE}}(\mathbf{\theta}) = \frac{1}{N}\sum_{i=1}^{N} \|\mathbf{x}_i - g_{\mathbf{\theta}'}(f_{\mathbf{\theta}}(\tilde{\mathbf{x}}_i))\|^2 + \lambda \sum_{l} \|\mathbf{W}_l\|_F^2$$
   * Latent bottleneck activations ($\mathbf{z} = f_{\mathbf{\theta}}(\mathbf{x})$) are concatenated with raw engineered features as dense non-linear representation proxies.

---

## 2. Dimensionality Reduction Methods Comparison

TABLE I: IEEE-FORMATTED DIMENSIONALITY REDUCTION COMPARISON

| Method | Mathematical Foundation | Linear / Non-Linear | Global / Local Topology | Computational Complexity | Healthcare Suitability & Clinical Interpretability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Principal Component Analysis (PCA)** | Eigen-decomposition of covariance matrix $\mathbf{\Sigma} = \frac{1}{N}\mathbf{X}^T\mathbf{X}$. | Linear | Global Variance Maximization | $\mathcal{O}(D^3 + D^2 N)$ | Moderate; linear orthogonal components lose direct clinical biomarker units. |
| **Autoencoders (DAE / VAE)** | Non-linear neural bottleneck compression via non-linear activations. | Non-Linear | Global & Complex Non-Linear Manifolds | $\mathcal{O}(N \cdot D \cdot W \cdot E)$ | High for deep pipelines; preserves non-linear biological pathways; low direct interpretability. |
| **t-Distributed Stochastic Neighbor Embedding (t-SNE)** | Minimizes Kullback-Leibler (KL) divergence between high-dim and low-dim probability distributions. | Non-Linear | Local Neighborhood Distances | $\mathcal{O}(N^2)$ or $\mathcal{O}(N \log N)$ | Excellent for 2D/3D EDA visualization; **Unsuitable for predictive modeling** due to non-parametric projection fragility on unseen test points. |
| **Uniform Manifold Approximation & Projection (UMAP)** | Riemannian geometry & fuzzy simplicial set topological representation. | Non-Linear | Preserves both Local and Global Structure | $\mathcal{O}(N \log N)$ | Superior visualization vs. t-SNE; supports transform operations on unseen test data, but distorts feature interpretability. |

---

## 3. Classical Feature Selection Algorithms Comparison

TABLE II: IEEE-FORMATTED FEATURE SELECTION ALGORITHMS COMPARISON

| Selection Algorithm | Paradigm / Family | Working Principle & Objective Function | Computational Complexity | Advantages | Limitations | Healthcare Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mutual Information (MI)** | Filter (Information Theoretic) | Measures non-linear dependency: $I(X; Y) = \iint p(x,y)\log \frac{p(x,y)}{p(x)p(y)} dx dy$. | $\mathcal{O}(N \cdot D)$ | Detects non-linear associations without parametric distribution assumptions. | Evaluates features independently; ignores redundant feature pairs. | High (Initial Screening) |
| **Chi-Square ($\chi^2$) Test** | Filter (Statistical Test) | Evaluates independence: $\chi^2 = \sum \frac{(O - E)^2}{E}$ on discretized continuous features. | $\mathcal{O}(N \cdot D)$ | Fast screening for non-negative categorical features. | Requires binning continuous variables; sensitive to sample size. | Moderate (Categorical attributes) |
| **Recursive Feature Elimination (RFE)** | Wrapper (Iterative Model-Based) | Iteratively trains estimator, ranks feature weights, and prunes lowest-weighted feature $W_{\min}$. | $\mathcal{O}(D^2 \cdot M_{\text{train}})$ | Account for model-specific interactions; isolates optimal joint subsets. | High computational cost; susceptible to overfitting on small sample sizes ($N < 500$). | High (Fine Tuning) |
| **Random Forest Importance (MDI)** | Embedded (Tree Impurity) | Accumulates Gini impurity reduction across all decision tree split nodes: $\Delta I_G$. | $\mathcal{O}(N \cdot D \cdot T \log N)$ | Fast; handles non-linearities and multi-collinear feature sets automatically. | Biased toward continuous features with high cardinality. | High (Ensemble Baseline) |
| **XGBoost Gain Importance** | Embedded (Gradient Boosted Gain) | Measures average fractional gain brought by a feature to split nodes: $\text{Gain} = \frac{G_L^2}{H_L + \lambda} + \dots$. | $\mathcal{O}(N \cdot D \cdot T \cdot K)$ | High discriminative accuracy; robust under regularization ($\gamma, \lambda$). | Sensitive to hyperparameter configuration. | Very High (Benchmark standard) |
| **LASSO ($\ell_1$ Regularization)** | Embedded (Linear Penalization) | Minimizes loss with absolute coefficient penalty: $\min_{\beta} \|y - \mathbf{X}\beta\|_2^2 + \alpha \|\beta\|_1$. | $\mathcal{O}(N \cdot D^2)$ | Shrinks uninformative coefficients to exactly zero ($\beta_j = 0$), performing feature selection natively. | Arbitrarily selects 1 feature from a group of highly correlated features ($r > 0.9$). | High (Linear Baselines) |
| **Sequential Feature Selection (SFS)** | Wrapper (Forward/Backward) | Greedy search adding/subtracting 1 feature at a time based on cross-validation score. | $\mathcal{O}(D^2 \cdot M_{\text{train}})$ | Guarantees monotonic optimization for target cross-validation score. | Computationally expensive; prone to local optima traps. | Moderate |

---

## 4. Novel Adaptive Feature Optimization (AFO) Framework

To resolve the limitations of isolated feature selection heuristics (such as LASSO's arbitrary selection under collinearity or Mutual Information's neglect of redundancy), an original **Adaptive Feature Optimization (AFO)** consensus framework is designed.

```
+-----------------------------------------------------------------------------------+
|                        Preprocessed & Enriched Feature Pool                       |
|                             F_raw = {f_1, f_2, ..., f_D}                          |
+-----------------------------------------------------------------------------------+
                                         |
     +-------------------+---------------+---------------+-------------------+
     |                   |                               |                   |
     v                   v                               v                   v
+---------+         +---------+                     +---------+         +---------+
| Path A: |         | Path B: |                     | Path C: |         | Path D: |
| Mutual  |         | LASSO   |                     | XGBoost |         | RFE     |
| Info    |         | L1      |                     | Gain    |         | (RF)    |
+---------+         +---------+                     +---------+         +---------+
     |                   |                               |                   |
     v                   v                               v                   v
[Rank R_MI]         [Rank R_LASSO]                  [Rank R_XGB]        [Rank R_RFE]
     |                   |                               |                   |
     +-------------------+---------------+---------------+-------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       Phase 1: Borda Consensus Scoring Matrix                     |
|                 S_consensus(f_j) = sum_{m in M} w_m * (D - R_m(f_j) + 1)           |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       Phase 2: Collinearity & Redundancy Pruning                  |
|          If Pearson |r(f_i, f_j)| > 0.85, Drop feature with lower S_consensus       |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       Phase 3: Iterative Elbow Subset Selection                   |
|               Select top K* features maximizing Stratified 10-Fold F1             |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                           Final Optimized Feature Subset                          |
|                                  F_opt = {f*_1, ..., f*_K}                        |
+-----------------------------------------------------------------------------------+
```

### 4.1 AFO Mathematical Formulation & Consensus Mechanism

1. **Multi-Source Ranking Generation:**
   Let $M = \{\text{MI}, \text{LASSO}, \text{XGBoost}, \text{RFE}\}$ represent the ensemble of feature selectors. Each selector evaluates the feature space $\mathcal{F} = \{f_1, f_2, \dots, f_D\}$ and produces a ordinal rank vector $\mathbf{R}_m(f_j) \in \{1, 2, \dots, D\}$, where rank 1 represents the highest discriminative importance.

2. **Weighted Borda Consensus Score ($S_{\text{consensus}}$):**
   The unified consensus score for feature $f_j$ is calculated as:
   $$S_{\text{consensus}}(f_j) = \sum_{m \in M} w_m \cdot \left[ D - \mathbf{R}_m(f_j) + 1 \right]$$
   where $w_m$ represents selector reliability weights ($\sum w_m = 1.0$), initialized as:
   $$w_{\text{MI}} = 0.20, \quad w_{\text{LASSO}} = 0.20, \quad w_{\text{XGBoost}} = 0.30, \quad w_{\text{RFE}} = 0.30$$

3. **Collinearity Redundancy Pruning:**
   For feature pairs $(f_a, f_b)$ exhibiting absolute Pearson/Spearman correlation $|r(f_a, f_b)| > \tau_{\text{corr}}$ (where $\tau_{\text{corr}} = 0.85$), the algorithm computes:
   $$\text{Retain } f_a \iff S_{\text{consensus}}(f_a) \ge S_{\text{consensus}}(f_b), \quad \text{else Drop } f_b$$

4. **Iterative Elbow Subset Optimization:**
   The algorithm ranks the deduplicated features by $S_{\text{consensus}}$ and evaluates cumulative subsets $\mathcal{F}_k = \{f_{(1)}, f_{(2)}, \dots, f_{(k)}\}$ for $k = 1 \dots D'$ using LightGBM under 10-Fold Cross-Validation. The optimal cardinality $K^*$ is selected at the maximum cross-validation F1-Score:
   $$K^* = \arg\max_{k} \left( \text{F1-Score}_{\text{10-CV}}(\mathcal{F}_k) \right)$$

---

## 5. Risk Assessment & Mitigation Architecture

TABLE III: IEEE-FORMATTED RISK ASSESSMENT & MITIGATION MATRIX

| Identified Risk Factor | Risk Severity | Potential Impact on Clinical Modeling | Algorithmic Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Over-Pruning Discriminative Biomarkers** | High | Dropping subtle multi-variate features degrades model sensitivity on complex disease cases. | Enforce minimum threshold retaining top 50% candidate pool prior to collinearity pruning. |
| **Feature Rank Instability across Folds** | Moderate | Stochastic variation between CV folds produces inconsistent feature subsets. | Compute Borda consensus scores across all 10 folds independently and aggregate via mean rank. |
| **Computational Overhead of RFE Wrapper** | Moderate | High computational latency during multi-dataset hyperparameter tuning. | Parallelize RFE sub-models across multi-core CPU threads; limit RFE step size to 5% per iteration. |
| **Data Leakage in Selection** | Critical | Computing feature importances on the entire dataset leads to severe over-optimistic evaluation. | Enforce strict pipeline encapsulation: fit AFO selector matrix strictly within training folds during cross-validation. |
