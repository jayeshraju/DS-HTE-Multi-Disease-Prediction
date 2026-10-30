# Phase 7: Baseline Machine Learning Models — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction

Establishing a comprehensive baseline study using traditional supervised learning algorithms and modern gradient-boosted decision tree (GBDT) ensembles is a prerequisite for validating any proposed deep learning or hybrid machine learning architecture. In medical tabular data classification, classical algorithms often provide competitive baseline performance, rapid inference speed, and lower risk of overfitting on moderate sample sizes ($N \in [300, 1000]$).

This phase establishes the theoretical, mathematical, and practical design of a 9-model baseline evaluation suite covering diverse decision boundary paradigms: linear models, probabilistic classifiers, instance-based learners, maximum-margin classifiers, non-parametric decision trees, bagging ensembles, and advanced gradient boosting frameworks.

---

## 2. Theoretical & Mathematical Foundations of Baseline Classifiers

```
                                  +-------------------------------------------------------+
                                  |         Baseline Model Architecture Classification    |
                                  +-------------------------------------------------------+
                                                              |
          +---------------------------+-----------------------+-----------------------+---------------------------+
          |                           |                                               |                           |
          v                           v                                               v                           v
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
|   Linear & Bayes  |       |  Distance & Margin|                           |  Tree & Bagging   |       | Gradient Boosting |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| - Logistic Regr.  |       | - K-Nearest       |                           | - Decision Tree   |       | - XGBoost         |
| - Naïve Bayes     |       |   Neighbors (KNN) |                           | - Random Forest   |       | - LightGBM        |
|                   |       | - Support Vector  |                           |                   | - CatBoost        |
|                   |       |   Machine (SVM)   |                           |                   |                   |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
```

### 2.1 Linear & Probabilistic Baseline Models

#### A. Logistic Regression (LR)
* **Theory:** Parametric linear classifier that models the log-odds of the binary diagnostic target as a linear combination of input attributes.
* **Mathematical Formulation:** Given feature vector $\mathbf{x} \in \mathbb{R}^D$, posterior class probability is defined via the Sigmoidal link function:
  $$P(y=1 | \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
  Loss function minimized with $\ell_2$ (Ridge) or $\ell_1$ (LASSO) regularization:
  $$\mathcal{L}_{\text{LR}}(\mathbf{w}) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \hat{y}_i + (1 - y_i) \log (1 - \hat{y}_i) \right] + \frac{\lambda}{2}\|\mathbf{w}\|_2^2$$
* **Strengths:** Fully interpretable coefficients ($\mathbf{w}$ represents log-odds ratios); extremely fast training and real-time inference.
* **Weaknesses:** Cannot capture non-linear feature interactions without explicit manual polynomial expansion.
* **Healthcare Suitability:** Benchmark baseline for medical odds-ratio risk modeling.

#### B. Gaussian Naïve Bayes (GNB)
* **Theory:** Generative probabilistic classifier applying Bayes' Theorem under the conditional independence assumption among predictor attributes.
* **Mathematical Formulation:**
  $$P(y=k | \mathbf{x}) \propto P(y=k) \prod_{j=1}^D P(x_j | y=k), \quad P(x_j | y=k) = \frac{1}{\sqrt{2\pi \sigma_{k,j}^2}} \exp\left( -\frac{(x_j - \mu_{k,j})^2}{2\sigma_{k,j}^2} \right)$$
* **Strengths:** Zero iterative gradient steps required; highly resilient to small sample sizes.
* **Weaknesses:** Conditional independence assumption rarely holds across correlated clinical biomarkers (e.g., ALT vs. AST or Blood Urea vs. Creatinine).
* **Healthcare Suitability:** Fast probabilistic initial screening benchmark.

---

### 2.2 Instance-Based & Maximum-Margin Models

#### C. K-Nearest Neighbors (KNN)
* **Theory:** Non-parametric instance-based classifier assigning class labels based on majority voting among $k$-closest instances in metric space.
* **Mathematical Formulation:** For query point $\mathbf{x}_0$, find set $\mathcal{N}_k(\mathbf{x}_0)$ minimizing Minkowski distance ($p=2$ for Euclidean):
  $$d(\mathbf{x}_i, \mathbf{x}_j) = \left( \sum_{m=1}^D |x_{i,m} - x_{j,m}|^p \right)^{1/p}, \quad \hat{y} = \arg\max_{c} \sum_{i \in \mathcal{N}_k(\mathbf{x}_0)} \mathbb{I}(y_i = c)$$
* **Strengths:** Makes zero assumptions about underlying probability distributions.
* **Weaknesses:** Highly sensitive to unscaled feature ranges and high-dimensional noise ("curse of dimensionality").
* **Healthcare Suitability:** Effective for similarity-based patient matching cohorts.

#### D. Support Vector Machine (SVM)
* **Theory:** Maximum-margin classifier projecting input space into high-dimensional Hilbert space via kernel functions $\Phi(\mathbf{x})$ to construct an optimal separating hyperplane.
* **Mathematical Formulation:**
  $$\min_{\mathbf{w}, b, \xi} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \Phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
  Dual formulation using Radial Basis Function (RBF) Kernel:
  $$K(\mathbf{x}_i, \mathbf{x}_j) = \exp\left( -\gamma \|\mathbf{x}_i - \mathbf{x}_j\|^2 \right)$$
* **Strengths:** Superior performance in high-dimensional non-linear spaces (e.g., WBCD dataset); robust against overfitting via margin regularization $C$.
* **Weaknesses:** Sensitive to feature scaling; non-probabilistic output requires Platt scaling for probability estimates.
* **Healthcare Suitability:** Gold-standard classical classifier for continuous diagnostic parameters.

---

### 2.3 Decision Tree & Bagging Ensembles

#### E. Decision Tree (CART)
* **Theory:** Non-parametric hierarchical tree recursive space-partitioning algorithm using Gini Impurity or Information Gain split metrics.
* **Mathematical Formulation:** Gini Impurity for node $m$:
  $$I_G(m) = 1 - \sum_{k=1}^K p_{m,k}^2$$
* **Strengths:** Human-readable decision rules matching clinical diagnostic flowcharts.
* **Weaknesses:** High variance; prone to severe overfitting on noisy clinical datasets if un-pruned.
* **Healthcare Suitability:** Direct translation to decision trees used in clinical protocols.

#### F. Random Forest (RF)
* **Theory:** Ensemble bagging architecture aggregating predictions from $B$ decorrelated decision trees constructed via Bootstrap Aggregating (Bagging) and random feature subspace selection.
* **Mathematical Formulation:**
  $$\hat{f}_{\text{RF}}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x}; \Theta_b)$$
* **Strengths:** Reduces variance without increasing bias; provides Gini impurity feature importance ranking; highly robust to outliers.
* **Weaknesses:** High memory footprint for deep tree ensembles.
* **Healthcare Suitability:** Outstanding multi-purpose tabular baseline.

---

### 2.4 Gradient Boosted Decision Tree (GBDT) Ensembles

#### G. Extreme Gradient Boosting (XGBoost)
* **Theory:** Second-order gradient boosting framework optimizing regularized objective function via exact/approximate greedy tree splitting.
* **Mathematical Formulation:** Objective at step $t$:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^N \left[ g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i) \right] + \Omega(f_t), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^T w_j^2$$
  where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$.
* **Strengths:** Out-of-core computing; native handling of missing values; hardware acceleration.
* **Weaknesses:** Requires rigorous hyperparameter tuning to prevent overfitting on small cohorts.
* **Healthcare Suitability:** Benchmark industry standard for structured EHR tabular data.

#### H. LightGBM (Light Gradient Boosting Machine)
* **Theory:** Gradient boosting framework using Gradient-based One-Side Sampling (GOSS) and Exclusive Feature Bundling (EFB) with leaf-wise tree growth.
* **Mathematical Formulation:** Keeps instances with large gradients while randomly sampling instances with small gradients to reduce computation:
  $$\tilde{g}_i = g_i \text{ for top } a\%, \quad \tilde{g}_i = \frac{1-a}{b} g_i \text{ for sampled } b\%$$
* **Strengths:** Unmatched speed ($\approx 10\times$ faster than standard XGBoost); extremely low memory footprint.
* **Weaknesses:** Leaf-wise growth can overfit small sample sizes ($N < 500$) if `max_depth` is unbounded.
* **Healthcare Suitability:** Preferred for large-scale multi-disease deployment pipelines.

#### I. CatBoost (Categorical Boosting)
* **Theory:** Gradient boosting implementation utilizing Symmetric Trees and Ordered Target Statistics to eliminate target leakage in categorical features.
* **Mathematical Formulation:** Computes target encoding sequentially across random permutations:
  $$\hat{x}_{i,k} = \frac{\sum_{j \in \mathcal{P}_i} \mathbb{I}(x_{j,k} = x_{i,k}) \cdot y_j + a \cdot P}{\sum_{j \in \mathcal{P}_i} \mathbb{I}(x_{j,k} = x_{i,k}) + a}$$
* **Strengths:** Handles categorical features natively without One-Hot expansion; superior baseline accuracy out-of-the-box.
* **Weaknesses:** Slightly longer GPU training duration compared to LightGBM.
* **Healthcare Suitability:** Ideal for mixed continuous-categorical medical datasets (e.g., CKD).

---

## 3. IEEE Comparative Baseline Matrix

TABLE I: IEEE-FORMATTED COMPARATIVE ANALYSIS OF BASELINE ML ALGORITHMS

| Algorithm | Model Family | Math Foundation | Train Complexity | Inference Complexity | Primary Strengths | Critical Weaknesses | Healthcare Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Linear | Sigmoid / Log-Likelihood | $\mathcal{O}(N \cdot D)$ | $\mathcal{O}(D)$ | Interpretable odds ratios; fast execution. | Fits linear decision boundaries only. | High (Baseline) |
| **Naïve Bayes** | Probabilistic | Bayes Theorem + Gaussian PDF | $\mathcal{O}(N \cdot D)$ | $\mathcal{O}(K \cdot D)$ | Fast non-iterative training; works on small $N$. | Ignores feature correlation dynamics. | Moderate |
| **KNN** | Instance-Based | Minkowski Metric Distance | $\mathcal{O}(1)$ | $\mathcal{O}(N \cdot D)$ | Zero distributional assumptions. | Slow inference; fragile under noisy attributes. | Moderate |
| **SVM (RBF)** | Margin-Based | Dual Kernel Mapping | $\mathcal{O}(N^2 \cdot D)$ to $\mathcal{O}(N^3)$ | $\mathcal{O}(N_{\text{SV}} \cdot D)$ | Excellent high-dimensional separation. | Black-box decision boundary; slow training. | High |
| **Decision Tree** | Hierarchical | CART / Gini Impurity | $\mathcal{O}(N \cdot D \log N)$ | $\mathcal{O}(\text{depth})$ | Transparent flowchart logic. | High variance; prone to overfitting. | Moderate |
| **Random Forest** | Bagging Ensemble | Bootstrap Aggregation | $\mathcal{O}(B \cdot N \cdot D \log N)$ | $\mathcal{O}(B \cdot \text{depth})$ | Reduces variance; robust to outliers. | Memory heavy; complex internal ensemble logic. | Very High |
| **XGBoost** | Gradient Boosting | 2nd-Order Taylor Objective | $\mathcal{O}(K \cdot D \cdot N \log N)$ | $\mathcal{O}(K \cdot \text{depth})$ | High accuracy; handles missing values natively. | Requires hyperparameter tuning. | **Very High (Gold Standard)** |
| **LightGBM** | Gradient Boosting | GOSS + EFB Leaf-wise Growth | $\mathcal{O}(K \cdot D \cdot N_{\text{sub}})$ | $\mathcal{O}(K \cdot \text{depth})$ | Unmatched training speed; low memory footprint. | Prone to leaf-wise overfitting on tiny datasets. | **Very High** |
| **CatBoost** | Gradient Boosting | Ordered Target Statistics | $\mathcal{O}(K \cdot N \cdot D)$ | $\mathcal{O}(K \cdot \text{depth})$ | Superior handling of categorical EHR features. | Higher training GPU runtime. | **Very High** |

---

## 4. Fair Comparison & Experimental Protocol Design

### 4.1 Stratified Cross-Validation & Hyperparameter Tuning Strategy
To guarantee a completely unbiased baseline benchmark:
1. **Validation Protocol:** Repeated Stratified 10-Fold Cross-Validation (3 repeats = 30 evaluations per model).
2. **Search Strategy:** Bayesian Optimization via Optuna ($100$ trials per model family) optimizing cross-validated F1-Score:
   $$\text{Objective}(\mathbf{\theta}) = \frac{1}{K}\sum_{k=1}^K \text{F1-Score}^{(k)}(\mathbf{\theta})$$

```
=========================== FAIR BASELINE EVALUATION PROTOCOL ===========================

  [ Preprocessed & Balanced Dataset (Phase 4 & 6) ]
                         |
                         v
  [ Stratified 10-Fold CV Partitioning (Random Seed = 42) ]
                         |
      +------------------+------------------+
      |                                     |
      v                                     v
  [ Model Family i ]                    [ Optuna Bayesian Tuning Engine ]
  (LR, GNB, KNN, SVM, DT, RF,            (100 Trials per Estimator)
   XGB, LightGBM, CatBoost)                 |
      |                                     v
      +---------------------------------> [ Evaluate Out-of-Fold (OOF) Metrics ]
                                            - F1-Score, ROC-AUC, PR-AUC
                                            - Sensitivity, Specificity, MCC
                                            - Training Runtime & Inference Latency
```

### 4.2 Standardized Hyperparameter Grid Search Spaces

TABLE II: IEEE-FORMATTED HYPERPARAMETER SEARCH SPACE FOR BASELINE MODELS

| Model | Hyperparameter Name | Target Range / Search Domain | Distribution Type |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | `C` (Inverse regularization strength) | $[10^{-4}, 10^3]$ | Log-Uniform |
| **Logistic Regression** | `penalty` | `['l1', 'l2', 'elasticnet']` | Categorical |
| **SVM** | `C` (Margin penalty) | $[10^{-2}, 10^3]$ | Log-Uniform |
| **SVM** | `gamma` (Kernel width) | $[10^{-4}, 10^0]$ | Log-Uniform |
| **Random Forest** | `n_estimators` | $[50, 500]$ | Discrete Int |
| **Random Forest** | `max_depth` | $[3, 20]$ | Discrete Int |
| **Random Forest** | `max_features` | `['sqrt', 'log2', 0.5, 0.8]` | Categorical/Float |
| **XGBoost** | `n_estimators` | $[100, 1000]$ | Discrete Int |
| **XGBoost** | `max_depth` | $[3, 12]$ | Discrete Int |
| **XGBoost** | `learning_rate` ($\eta$) | $[0.005, 0.3]$ | Log-Uniform |
| **XGBoost** | `subsample` / `colsample_bytree` | $[0.5, 1.0]$ | Uniform Float |
| **LightGBM** | `num_leaves` | $[15, 255]$ | Discrete Int |
| **LightGBM** | `min_child_samples` | $[5, 100]$ | Discrete Int |
| **CatBoost** | `depth` | $[4, 10]$ | Discrete Int |
| **CatBoost** | `l2_leaf_reg` | $[1.0, 10.0]$ | Uniform Float |

---

## 5. Expected Outcomes & Phase Summary

* **Expected Baseline Standard:** Advanced GBDT models (XGBoost, LightGBM, CatBoost) are projected to set the top baseline performance benchmarks across all 5 disease datasets, achieving $\text{F1-Scores} \ge 0.88-0.95$ and $\text{ROC-AUC} \ge 0.92-0.97$.
* **Phase Summary:** Phase 7 provides a rigorous, publication-ready baseline evaluation suite comparing 9 classical and ensemble classifiers under standardized Bayesian hyperparameter tuning, laying the benchmark foundation against which the deep learning (Phase 8) and hybrid architectures (Phase 9) will be validated.
