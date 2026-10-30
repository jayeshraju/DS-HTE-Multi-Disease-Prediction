# Phase 10: Hyperparameter Optimization — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction

Hyperparameter optimization (HPO) is a critical phase in modern machine learning workflows. Sub-optimal hyperparameter choices (such as learning rate, tree depth, or self-attention head dimension) can severely reduce diagnostic predictive performance, induce deep network gradient explosion, or lead to decision tree overfitting.

In medical diagnostic systems, hyperparameter tuning must balance model complexity against sample size constraints while ensuring full computational reproducibility. This phase establishes a comparative evaluation across five hyperparameter optimization paradigms: **Grid Search**, **Random Search**, **Bayesian Optimization (Gaussian Processes)**, **Optuna (Tree-structured Parzen Estimator - TPE)**, and **Hyperband**, selecting the optimal framework for baseline, deep learning, and hybrid model tuning.

---

## 2. Comparative Analysis of Optimization Paradigms

```
                                  +-------------------------------------------------------+
                                  |         Hyperparameter Optimization Taxonomy          |
                                  +-------------------------------------------------------+
                                                              |
          +---------------------------+-----------------------+-----------------------+---------------------------+
          |                           |                                               |                           |
          v                           v                                               v                           v
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| Exhaustive Grid   |       | Stochastic Search |                           | Bayesian Surrogate|       | Adaptive Pruning  |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| - Grid Search     |       | - Random Search   |                           | - GP Bayesian Opt |       | - Hyperband       |
|                   |       |                   |                           | - Optuna (TPE)    |       |   (Successive     |
|                   |       |                   |                           |                   |    Halving)       |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
```

### 2.1 Theoretical & Mathematical Formulations

#### A. Exhaustive Grid Search
* **Theory:** Evaluates all Cartesian product combinations of specified discrete hyperparameter lists over a regular grid.
* **Mathematical Intuition:** Evaluates $N = \prod_{i=1}^H |P_i|$ combinations exhaustively.
* **Computational Complexity:** Exponential $\mathcal{O}(K^H)$ ("curse of dimensionality").
* **Limitations:** Extremely inefficient; wastes computational budget evaluating un-promising hyperparameter regions.

#### B. Random Search
* **Theory:** Samples hyperparameter configurations uniformly at random from specified continuous probability distributions.
* **Mathematical Intuition:** Independent sampling $\mathbf{\theta}^{(t)} \sim p(\mathbf{\theta})$.
* **Computational Complexity:** Linear $\mathcal{O}(T \cdot M_{\text{train}})$.
* **Advantages:** Significantly more efficient than Grid Search when effective hyperparameter dimensionality is low (Bergstra & Bengio principle).

#### C. Bayesian Optimization (Gaussian Process - GP)
* **Theory:** Sequential probabilistic model-based optimization using a Gaussian Process surrogate to construct a posterior probability distribution over objective function $f(\mathbf{\theta})$, guided by an Acquisition Function (e.g., Expected Improvement - EI).
* **Mathematical Intuition:**
  $$\text{EI}(\mathbf{\theta}) = \mathbb{E}\left[ \max(0, f(\mathbf{\theta}) - f(\mathbf{\theta}^*)) \right] = (\mu(\mathbf{\theta}) - f(\mathbf{\theta}^*))\Phi(Z) + \sigma(\mathbf{\theta})\phi(Z)$$
* **Limitations:** Cubic computational complexity $\mathcal{O}(T^3)$ of GP matrix inversion limits scalability for large numbers of trials ($T > 200$).

#### D. Optuna: Tree-structured Parzen Estimator (TPE)
* **Theory:** Sequential model-based optimization that models the conditional probability distributions $p(\mathbf{x}|y)$ of hyperparameters given objective values $y$, splitting trials into good and bad configurations based on quantile threshold $\gamma$.
* **Mathematical Intuition:** Uses Bayes' Rule to maximize ratio $\frac{l(\mathbf{x})}{g(\mathbf{x})}$:
  $$p(\mathbf{\theta}|y) = \begin{cases} l(\mathbf{\theta}), & \text{if } y < y^* \\ g(\mathbf{\theta}), & \text{if } y \ge y^* \end{cases}, \quad \text{EI}(\mathbf{\theta}) \propto \left( \gamma + \frac{g(\mathbf{\theta})}{l(\mathbf{\theta})}(1 - \gamma) \right)^{-1}$$
* **Advantages:** Linear complexity $\mathcal{O}(T)$ with respect to trials; native support for dynamic search space definitions and automated early pruning (Asynchronous Successive Halving - ASHA).

#### E. Hyperband (Successive Halving)
* **Theory:** Multi-fidelity optimization algorithm that allocates resources (e.g., training epochs) dynamically to a large pool of randomly sampled configurations, discarding the lowest-performing $1/\eta$ fraction at successive halving rungs.
* **Mathematical Intuition:** Balances resource allocation $R$ against configuration count $n$.
* **Limitations:** Assumes early epoch performance correlates strongly with final converged performance, which may fail for deep Transformer networks exhibiting late convergence.

---

## 3. Comprehensive IEEE-Style Optimization Comparison

TABLE I: IEEE-FORMATTED COMPARATIVE ANALYSIS OF HYPERPARAMETER OPTIMIZATION ALGORITHMS

| Algorithm | Optimization Strategy | Computational Complexity | Dynamic Search Space | Early Pruning Support | Primary Strengths | Critical Limitations | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Grid Search** | Deterministic Cartesian | $\mathcal{O}(K^H)$ (Exponential) | No | No | Simple; deterministic coverage. | High computational waste; scales poorly. | Rejected |
| **Random Search** | Uniform Stochastic | $\mathcal{O}(T \cdot M_{\text{train}})$ | No | No | Better coverage than Grid Search. | Ignores past trial evaluation history. | Baseline |
| **GP Bayesian Opt** | Gaussian Process Surrogate | $\mathcal{O}(T^3)$ (Cubic) | No | No | Sample-efficient for low trials ($T < 50$). | Cubic scaling; fails on discrete/categorical trees. | Candidate |
| **Hyperband** | Multi-Fidelity Halving | $\mathcal{O}(R \log R)$ | No | Yes (Rung Pruning) | Fast resource allocation. | Prunes late-converging deep models prematurely. | Candidate |
| **Optuna (TPE)** | **Tree-structured Parzen** | $\mathcal{O}(T)$ **(Linear)** | **Yes (Dynamic Pythonic)** | **Yes (ASHA Pruner)** | **Sample-efficient; linear scaling; handles mixed spaces.** | Requires objective function setup. | **Selected (Winner)** |

---

## 4. Proposed Unified Optimization Architecture

### 4.1 Recommended Optimization Framework: Optuna with ASHA Pruning
**Optuna** employing the **Tree-structured Parzen Estimator (TPE)** sampler paired with **Asynchronous Successive Halving (ASHA)** pruner is selected as the unified hyperparameter optimization engine across all models.

```
+-----------------------------------------------------------------------------------+
|                             Optuna Study Initialization                           |
|  - Objective: Maximize Out-of-Fold 10-Fold CV F1-Score                            |
|  - Sampler: TPESampler(seed=42, n_startup_trials=20)                              |
|  - Pruner: SuccessiveHalvingPruner(min_resource=5, reduction_factor=3)            |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                        Sample Hyperparameter Vector theta                         |
|  - GBDT: learning_rate, max_depth, n_estimators, subsample, colsample             |
|  - FT-Transformer: d_token, n_blocks, n_heads, dropout, lr, weight_decay          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                 Evaluate Stratified 10-Fold Cross-Validation                      |
|  - Apply Leakage-Free Preprocessing & Borderline-SMOTE per fold                   |
|  - Report Intermediate Metric at each CV fold/epoch                               |
|  - IF trial.should_prune() THEN Terminate Early (ASHA Pruning)                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                       Store Optimal Hyperparameters theta*                        |
+-----------------------------------------------------------------------------------+
```

---

### 4.2 Comprehensive Hyperparameter Search Space Specifications

TABLE II: IEEE-FORMATTED HYPERPARAMETER SEARCH SPACE SPECIFICATION

| Component / Model | Hyperparameter Name | Symbol | Target Domain Range | Scale / Distribution | Optimal Default |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FT-Transformer** | Embedding Dimension | $d_{\text{token}}$ | $[64, 256]$ | Step Categorical (Step 32) | $192$ |
| **FT-Transformer** | Encoder Blocks | $L$ | $[1, 6]$ | Discrete Int | $3$ |
| **FT-Transformer** | Attention Heads | $h$ | $\{2, 4, 8, 16\}$ | Categorical | $8$ |
| **FT-Transformer** | Feed-Forward Multiplier | $d_{\text{ff\_factor}}$ | $[1.0, 3.0]$ | Uniform Float | $1.33$ ($\frac{4}{3}$) |
| **FT-Transformer** | Dropout Rate | $p_{\text{drop}}$ | $[0.0, 0.5]$ | Uniform Float | $0.20$ |
| **FT-Transformer** | Learning Rate | $\eta_0$ | $[10^{-5}, 10^{-2}]$ | Log-Uniform | $10^{-4}$ |
| **FT-Transformer** | Weight Decay | $\lambda$ | $[10^{-4}, 10^{-1}]$ | Log-Uniform | $10^{-2}$ |
| **XGBoost / GBDT** | Learning Rate | $\eta$ | $[0.005, 0.3]$ | Log-Uniform | $0.05$ |
| **XGBoost / GBDT** | Tree Depth | $d_{\text{tree}}$ | $[3, 12]$ | Discrete Int | $6$ |
| **XGBoost / GBDT** | Number of Estimators | $K$ | $[100, 1000]$ | Discrete Int | $300$ |
| **XGBoost / GBDT** | Subsample Ratio | $\eta_{\text{sub}}$ | $[0.5, 1.0]$ | Uniform Float | $0.80$ |
| **XGBoost / GBDT** | Feature Subspace Ratio | $\eta_{\text{col}}$ | $[0.5, 1.0]$ | Uniform Float | $0.80$ |

---

## 5. Expected Outcomes & Phase Summary

* **Expected Optimization Performance:** Optuna TPE tuning is expected to reduce hyperparameter search latency by $4\times$ compared to Random Search and $20\times$ compared to Grid Search, producing converged hyperparameter states $\mathbf{\theta}^*$ that improve out-of-fold F1-Scores by $+2.1\%$ to $+4.5\%$ across all 5 disease datasets.
* **Phase Summary:** Phase 10 establishes a publication-ready, automated hyperparameter optimization framework utilizing **Optuna (TPE)** and ASHA early pruning under strict 10-fold cross-validation.
