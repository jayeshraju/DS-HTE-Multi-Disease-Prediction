# Phase 9: Proposed Hybrid Prediction Framework — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction & Originality Overview

While gradient boosting algorithms (XGBoost, CatBoost) achieve state-of-the-art decision tree boundary partitioning on tabular biomedical data, they lack continuous representation learning mechanisms for capturing dynamic multi-biomarker contextual embeddings. Conversely, deep self-attention Transformer architectures (FT-Transformer) extract rich non-linear feature interaction embeddings via Multi-Head Self-Attention, but can suffer from localized decision boundary over-smoothing on small-to-moderate clinical sample sizes ($N \in [300, 1000]$).

To solve this trade-off, this phase proposes an original, publication-grade **Hybrid Deep Attentive Gradient Boosting (HDA-GB) Architecture**. The framework synthesizes:
1. **Adaptive Feature Optimization (AFO):** Consensus-based information-theoretic feature selection.
2. **FT-Transformer Encoder:** Deep self-attention feature tokenization producing a high-dimensional latent representation $\mathbf{z}_{[\text{CLS}]}$.
3. **Downstream Gradient Boosting Classifier (XGBoost/CatBoost):** Gradient-boosted decision tree ensemble trained on the augmented feature space $\mathbf{X}_{\text{aug}} = [\mathbf{X}_{\text{AFO}} \mathbin{\Vert} \mathbf{z}_{[\text{CLS}]}]$.
4. **Explainable AI (XAI) Suite:** Integrated SHAP and LIME attribution layer providing local and global clinical diagnostic interpretability.

---

## 2. Comprehensive System Architecture & ASCII Diagrams

### 2.1 Complete End-to-End System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                  INPUT LAYER & DATA BALANCING                                     |
|  Raw Diagnostic Input Features -> Borderline-SMOTE / SMOTE-NC + Tomek Links Cleaning               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                            MODULE 1: ADAPTIVE FEATURE OPTIMIZATION (AFO)                          |
|  Multi-Source Rank Matrix (MI + LASSO + XGBoost + RFE) -> Borda Score -> Collinearity Pruning     |
|  Output: Optimized Feature Subset X_AFO in R^{N x K*}                                             |
+---------------------------------------------------------------------------------------------------+
                                                  |
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
+------------------------------------------+             +------------------------------------------+
|  MODULE 2: FT-TRANSFORMER FEATURE ENCODER|             |  DIRECT RESIDUAL PASS-THROUGH (X_AFO)    |
|  - Continuous Feature Tokenizer          |             |  Preserves raw, un-transformed continuous |
|  - Learnable [CLS] Token Insertion       |             |  biomarker values for interpretable tree  |
|  - L=3 Stacked Transformer Encoder Blocks|             |  splits                                  |
|  - Extract Latent Embedding z_[CLS]      |             +------------------------------------------+
+------------------------------------------+                                   |
                     |                                                         |
                     +----------------------------+----------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                        MODULE 3: AUGMENTED FEATURE REPRESENTATION MATRIX                          |
|                     X_augmented = Concatenate( X_AFO , z_[CLS] ) in R^{N x (K* + d)}               |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                       MODULE 4: DOWNSTREAM GRADIENT BOOSTING PREDICTOR                            |
|             Optimized XGBoost / CatBoost Ensemble Trained on Augmented Matrix X_augmented          |
|             Outputs: Class Probability P(y=1|x) & Prediction Confidence C_pred                    |
+---------------------------------------------------------------------------------------------------+
                                                  |
                                                  v
+---------------------------------------------------------------------------------------------------+
|                            MODULE 5: EXPLAINABLE AI (XAI) INTERPRETATION                          |
|             KernelSHAP & TreeSHAP Attribution -> Clinical Risk Stratification Report              |
+---------------------------------------------------------------------------------------------------+
```

---

### 2.2 Data Flow Diagram

```
[ Raw Patient EHR Data ] 
       │
       ▼
[ Preprocessing & Rescaling ] ──► (Imputation & RobustScaler)
       │
       ▼
[ Resampling Engine ] ─────────► (Borderline-SMOTE + Tomek Links)
       │
       ▼
[ AFO Consensus Engine ] ──────► (Borda Rank Selection -> X_AFO)
       │
       ├──────────────────────────────────────────┐
       ▼                                          ▼
[ FT-Transformer Encoder ]               [ Raw Feature Pass-through ]
       │                                          │
       ▼ (Extract z_[CLS])                        │
       └────────────────────┬─────────────────────┘
                            ▼
               [ Augmented Feature Matrix ]
                            │
                            ▼
               [ XGBoost / CatBoost Model ]
                            │
                            ▼
              [ Diagnostic Probability & XAI ]
```

---

### 2.3 Module Interaction Architecture

```
+------------------+         +----------------------+         +---------------------+
|   AFO Module     |         | FT-Transformer Module|         | Gradient Boosting   |
|                  |         |                      |         | Predictor           |
| - Computes Borda |         | - Tokenizes features |         |                     |
|   Consensus      |         | - Self-Attention     |         | - Trains GBDT on    |
| - Generates      |-------->| - Extracts z_[CLS]   |-------->|   augmented matrix  |
|   X_AFO subset   |         |   dense embedding    |         | - Computes risk &   |
+------------------+         +----------------------+         |   confidence        |
         |                                                    +---------------------+
         |                                                               |
         +---------------------------------------------------------------+
                                         |
                                         v
                             +-----------------------+
                             |   XAI Attribution     |
                             | - TreeSHAP & LIME     |
                             | - Patient Risk Report |
                             +-----------------------+
```

---

## 3. Mathematical Formulation & Latent Augmentation

### 3.1 Feature Tokenization & Attention Encoding
For an optimized input vector $\mathbf{x}_{\text{AFO}} \in \mathbb{R}^{K^*}$, the FT-Transformer Tokenizer maps each attribute $x_j$ into token space $\mathbf{T}_j \in \mathbb{R}^d$:
$$\mathbf{T}_j = x_j \cdot \mathbf{W}_j + \mathbf{b}_j, \quad \mathbf{W}_j \in \mathbb{R}^{1 \times d}, \quad \mathbf{b}_j \in \mathbb{R}^d$$

Prepend the learnable `[CLS]` classification token $\mathbf{e}_{[\text{CLS}]} \in \mathbb{R}^d$:
$$\mathbf{Z}^{(0)} = [\mathbf{e}_{[\text{CLS}]}, \mathbf{T}_1, \mathbf{T}_2, \dots, \mathbf{T}_{K^*}] \in \mathbb{R}^{(K^*+1) \times d}$$

Through $L$ Transformer layers with Multi-Head Self-Attention ($h$ heads):
$$\mathbf{Z}^{(l)} = \text{TransformerLayer}\left( \mathbf{Z}^{(l-1)} \right), \quad l = 1 \dots L$$

Extract the final `[CLS]` token activation state:
$$\mathbf{z}_{[\text{CLS}]} = \mathbf{Z}^{(L)}[0, :] \in \mathbb{R}^d$$

### 3.2 Feature Concatenation & Hybrid Representation Space
The augmented feature space $\mathbf{X}_{\text{aug}}$ couples original un-transformed clinical features with deep self-attention contextual embeddings:
$$\mathbf{X}_{\text{aug}} = \left[ \mathbf{X}_{\text{AFO}} \mathbin{\Vert} \mathbf{z}_{[\text{CLS}]} \right] \in \mathbb{R}^{N \times (K^* + d)}$$

### 3.3 Prediction & Confidence Estimation
The downstream gradient boosting model (XGBoost/CatBoost) evaluates $\mathbf{X}_{\text{aug}}$ to yield binary diagnostic probability $\hat{p} = P(y=1 | \mathbf{X}_{\text{aug}})$. Prediction confidence $C_{\text{pred}} \in [0, 1]$ is computed as:
$$C_{\text{pred}} = 2 \cdot \left| \hat{p} - 0.5 \right| = \begin{cases} 1.0, & \text{if } \hat{p} \in \{0.0, 1.0\} \text{ (Maximum Certainty)} \\ 0.0, & \text{if } \hat{p} = 0.5 \text{ (Maximum Ambiguity)} \end{cases}$$

---

## 4. Training & Inference Workflows

### 4.1 Sequential Training Protocol
1. **Phase A (AFO Fitting):** Fit consensus feature selectors on training fold $\mathbf{X}_{\text{train}}$ to derive $K^*$ optimal feature subset $\mathbf{X}_{\text{AFO}}$.
2. **Phase B (FT-Transformer Pre-training):** Train FT-Transformer using Binary Cross-Entropy loss for $E$ epochs with early stopping based on validation F1-Score.
3. **Phase C (Latent Feature Extraction):** Freeze FT-Transformer weights and pass $\mathbf{X}_{\text{AFO}}$ through the encoder to extract latent embeddings $\mathbf{Z}_{[\text{CLS}]}$.
4. **Phase D (Gradient Boosting Optimization):** Train XGBoost/CatBoost on augmented matrix $\mathbf{X}_{\text{aug}} = [\mathbf{X}_{\text{AFO}} \mathbin{\Vert} \mathbf{Z}_{[\text{CLS}]}]$.

### 4.2 Real-time Clinical Inference Protocol
1. Input patient blood panel attributes.
2. Preprocess using saved `MedicalPreprocessor` parameters.
3. Filter attributes via selected AFO feature indices.
4. Pass filtered attributes through pre-trained FT-Transformer encoder to obtain $\mathbf{z}_{[\text{CLS}]}$.
5. Construct $\mathbf{x}_{\text{aug}} = [\mathbf{x}_{\text{AFO}} \mathbin{\Vert} \mathbf{z}_{[\text{CLS}]}]$.
6. Pass $\mathbf{x}_{\text{aug}}$ through trained XGBoost model to obtain diagnostic risk probability $\hat{p}$ and confidence $C_{\text{pred}}$.
7. Pass model predictions to SHAP / LIME explainer to generate patient-level attribution scores.

---

## 5. Novelty Analysis & IEEE Comparison Matrix

TABLE I: IEEE NOVELTY MATRIX — PROPOSED HYBRID FRAMEWORK VS. EXISTING LITERATURE

| Architectural Feature | Traditional ML (RF/SVM) | Standard GBDT (XGBoost) | Standard Deep MLPs | Standalone TabTransformer | **Proposed HDA-GB Framework** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Adaptive Feature Selection** | Univariate Filter | Native Tree Splits | None | None | **Borda Consensus AFO Framework** |
| **Deep Feature Attention** | None | None | None | Categorical Only | **Full Continuous + Categorical MHSA** |
| **Latent Feature Augmentation**| None | None | Latent MLP Layer | None | **`[CLS]` Embedding + GBDT Fusion** |
| **Decision Boundary Quality** | Moderate | High (Non-linear) | Smooth | Moderate | **Optimal (Tree Partitioning on Deep Embeddings)** |
| **Built-in Interpretability** | Impurity / Coefficients | Gain Importance | Post-hoc LIME | Attention Weights | **Unified TreeSHAP + KernelSHAP Suite** |
| **Sample Efficiency ($N < 1000$)**| High | High | Low | Low | **High (Hybrid Fusion prevents deep overfitting)** |

---

## 6. Expected Outcomes & Phase Summary

* **Expected Superior Performance:** The proposed HDA-GB framework is expected to outperform standalone GBDT models and deep tabular models by $+3.5\%$ to $+6.2\%$ in F1-Score and PR-AUC across all 5 disease benchmarks.
* **Phase Summary:** Phase 9 establishes an original, publication-ready hybrid architecture coupling AFO feature consensus, FT-Transformer self-attention embeddings, and GBDT downstream classification.
