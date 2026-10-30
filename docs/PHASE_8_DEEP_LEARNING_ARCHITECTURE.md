# Phase 8: Deep Learning Architecture Design — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction

While tree-based gradient boosting algorithms excel on tabular data, deep learning architectures offer distinct representation advantages for multi-disease medical informatics. Deep neural networks enable automatic non-linear feature interaction extraction, end-to-end multi-task joint representations, and dense latent spatial embeddings without requiring explicit manual feature interaction specifications.

However, applying standard computer vision or natural language deep architectures directly to tabular medical data presents structural challenges: continuous biomedical features lack natural spatial locality (unlike pixel grids in imaging) or sequential grammar (unlike tokens in NLP). This phase conducts a comparative architectural study across five deep tabular learning paradigms: **Multi-Layer Perceptrons (MLP/ANN)**, **1D Convolutional Neural Networks (1D CNN)**, **TabNet**, **TabTransformer**, and **FT-Transformer (Feature Tokenizer Transformer)**, selecting the optimal architecture for structured diagnostic prediction.

---

## 2. Comparative Analysis of Deep Tabular Architectures

```
                                  +-------------------------------------------------------+
                                  |         Deep Tabular Architecture Taxonomy            |
                                  +-------------------------------------------------------+
                                                              |
          +---------------------------+-----------------------+-----------------------+---------------------------+
          |                           |                                               |                           |
          v                           v                                               v                           v
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
|  Dense Baseline   |       | Spatial Filtering |                           | Sequential Masking|       | Self-Attention    |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| - Artificial      |       | - 1D Convolutional|                           | - TabNet          |       | - TabTransformer  |
|   Neural Network  |       |   Neural Network  |                           |   (Sequential     |       | - FT-Transformer  |
|   (ANN / MLP)     |       |   (1D CNN)        |                           |    Attentive Mask) |       |   (Feature Tok.)  |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
```

### 2.1 Theoretical & Mathematical Formulations

#### A. Dense Artificial Neural Network (ANN / Multi-Layer Perceptron)
* **Architecture:** Fully connected feedforward network consisting of an input layer, $L$ hidden dense layers with non-linear activation functions, Batch Normalization, Dropout, and a sigmoid output neuron.
* **Feature Representation:** Projects input vector $\mathbf{x} \in \mathbb{R}^D$ into continuous latent space $\mathbf{h}^{(l)} \in \mathbb{R}^{d_l}$:
  $$\mathbf{h}^{(l)} = \text{ReLU}\left( \text{BatchNorm}\left( \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)} \right) \right)$$
* **Attention Mechanism:** None. Treats all dense connections uniformly.
* **Strengths:** Simple, low parameter count, easy to optimize on small tabular cohorts.
* **Weaknesses:** Lacks spatial or feature-wise attention; prone to overfitting on noisy features without heavy regularization.
* **Healthcare Suitability:** General continuous diagnostic baseline.

#### B. 1D Convolutional Neural Network (1D CNN)
* **Architecture:** Applies 1D sliding convolutional kernels over feature vectors reshaped into sequence tensors $\mathbf{X} \in \mathbb{R}^{1 \times D}$, followed by 1D Max-Pooling and Dense projection layers.
* **Feature Representation:** Extracts localized continuous feature gradient patterns across adjacent tabular indices:
  $$\mathbf{z}_i = f\left( \sum_{k=1}^K \mathbf{w}_k \cdot \mathbf{x}_{i+k-1} + b \right)$$
* **Attention Mechanism:** None (Static local receptive fields).
* **Strengths:** Captures local patterns among sequentially ordered blood panel measurements.
* **Weaknesses:** Tabular feature order is arbitrary; sliding 1D convolution assumes artificial local spatial dependencies that may not exist between un-ordered laboratory features (e.g., Age adjacent to Total Bilirubin).
* **Healthcare Suitability:** Moderate.

#### C. TabNet (Attentive Canonical Tabular Learning)
* **Architecture:** Uses a sequential multi-step architecture ($\text{Steps} = S$) where each step employs a sparse attention mask $\mathbf{M}[i] \in \mathbb{R}^D$ to select salient features, followed by a shared feature transformer block.
* **Feature Representation:** Uses Sparsemax / Entmax normalization to select sparse feature subsets at each decision step $i$:
  $$\mathbf{M}[i] = \text{Sparsemax}\left( \gamma \cdot h_{i-1}(\mathbf{a}[i-1]) \cdot \mathbf{P}[i-1] \right)$$
  $$\mathbf{P}[i] = \prod_{j=1}^i (\gamma - \mathbf{M}[j])$$
  where $\mathbf{P}[i]$ is the prior scale factor suppressing previously selected features, and $\gamma$ is a sparsity parameter.
* **Attention Mechanism:** Sequential instance-wise sparse feature selection masks.
* **Strengths:** Provides built-in instance-level feature selection masks (interpretable feature importances); does not require target encoding.
* **Weaknesses:** High parameter footprint; difficult optimization hyperparameter landscape ($\gamma, N_d, N_a$).
* **Healthcare Suitability:** High for clinical interpretability, but computationally intensive.

#### D. TabTransformer
* **Architecture:** Applies a Column Embedding layer to transform categorical features into dense embedding vectors, passes these embeddings through multi-head self-attention Transformer blocks, and concatenates them with raw continuous features before final MLP classification.
* **Feature Representation:** Categorical attributes $\mathbf{x}_{\text{cat}}$ undergo parametric lookup embeddings $\mathbf{E}_i(x_{i}) \in \mathbb{R}^d$, processed via Multi-Head Self-Attention:
  $$\mathbf{Z} = \text{MultiHeadAttn}(\mathbf{E}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\mathbf{W}^O$$
* **Attention Mechanism:** Multi-Head Self-Attention over categorical feature embeddings.
* **Strengths:** Superior contextual interaction modeling across categorical diagnostic attributes.
* **Weaknesses:** Continuous features bypass the Transformer attention mechanism and are simply concatenated at the final layer.
* **Healthcare Suitability:** High for categorical-heavy datasets (e.g., CKD), sub-optimal for purely continuous datasets (e.g., WBCD).

#### E. FT-Transformer (Feature Tokenizer Transformer)
* **Architecture:** Converts **both** continuous and categorical features into dense feature tokens using a Feature Tokenizer module, then processes all tokens through a stack of standard Transformer Multi-Head Self-Attention layers and residual blocks.
* **Feature Representation (Feature Tokenizer):** For feature $j \in \{1 \dots D\}$, token $\mathbf{T}_j \in \mathbb{R}^d$ is generated as:
  $$\mathbf{T}_j = \begin{cases} \mathbf{x}_j \cdot \mathbf{W}_j^{(num)} + \mathbf{b}_j^{(num)}, & \text{if } j \text{ is continuous} \\ \text{Embedding}_j(\mathbf{x}_j), & \text{if } j \text{ is categorical} \end{cases}$$
  A learnable `[CLS]` token is prepended: $\mathbf{Z}^{(0)} = [\mathbf{e}_{[\text{CLS}]}, \mathbf{T}_1, \mathbf{T}_2, \dots, \mathbf{T}_D] \in \mathbb{R}^{(D+1) \times d}$.
* **Attention Mechanism:** Full Multi-Head Self-Attention ($\mathbf{Q}, \mathbf{K}, \mathbf{V}$) across all feature tokens:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V}$$
* **Strengths:** **Models all pairwise non-linear feature interactions (continuous-continuous, continuous-categorical, categorical-categorical) uniformly; state-of-the-art deep performance on tabular data.**
* **Weaknesses:** Higher computational complexity ($\mathcal{O}(D^2 \cdot d)$ per attention layer).
* **Healthcare Suitability:** **Highest (Selected Architecture).**

---

## 3. Comprehensive IEEE-Style Deep Architecture Comparison

TABLE I: IEEE-FORMATTED COMPARATIVE ANALYSIS OF DEEP TABULAR ARCHITECTURES

| Architecture | Representation Strategy | Attention Mechanism | Continuous Feature Integration | Categorical Feature Integration | Computational Complexity | Primary Strengths | Critical Limitations | Selection Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ANN / MLP** | Dense Projections | None | Direct numerical input | One-Hot / Label Encoded | $\mathcal{O}(N \cdot D \cdot d_h)$ | Simple; low memory footprint. | No feature attention; overfits noisy data. | Baseline |
| **1D CNN** | Sliding Kernels | None (Local receptive fields) | Reshaped 1D Sequence | Numeric mapping | $\mathcal{O}(N \cdot D \cdot K \cdot C)$ | Extracts localized trend signals. | Assumes artificial spatial ordering of features. | Rejected |
| **TabNet** | Sequential Masks | Sparsemax Attentive Masks | Direct numerical input | Direct categorical input | $\mathcal{O}(N \cdot S \cdot D \cdot d_a)$ | Built-in instance feature masks. | High optimization sensitivity; slow convergence. | Candidate |
| **TabTransformer** | Hybrid Embedding | Multi-Head Self-Attention | Bypasses Attention (MLP concatenated) | Dense Lookup Embeddings | $\mathcal{O}(N \cdot D_{\text{cat}}^2 \cdot d)$ | Superior for categorical-heavy data. | Continuous features do not interact via attention. | Candidate |
| **FT-Transformer** | Feature Tokenizer | **Full Multi-Head Self-Attention** | **Linear Tokenizer Embedding** | **Dense Lookup Tokenizer Embedding** | $\mathcal{O}(N \cdot D^2 \cdot d \cdot L)$ | **Captures all non-linear continuous and categorical interactions.** | Higher computational overhead. | **Selected (Winner)** |

---

## 4. Proposed Deep Architecture & Optimization Design

### 4.1 Recommended Architecture: FT-Transformer for Structured Medical Data
The **FT-Transformer (Feature Tokenizer Transformer)** is selected as the primary deep learning architecture for the multi-disease prediction framework. By tokenizing continuous numerical blood biomarkers (e.g., Glucose, Serum Creatinine, Bilirubin) alongside categorical clinical attributes (e.g., ECG results, Chest Pain Type), FT-Transformer allows self-attention heads to dynamically weight clinical feature-pair interactions without requiring manual polynomial feature engineering.

```
+-----------------------------------------------------------------------------------+
|                        Preprocessed Clinical Features (X)                         |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            Feature Tokenizer Module                               |
| - Continuous: T_j = x_j * W_j + b_j  (Linear Projection to d=192)                  |
| - Categorical: T_j = LookupEmbedding(x_j)                                         |
| - Prepend Learnable [CLS] Token -> Z^(0) in R^{(D+1) x d}                          |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                  Stacked Transformer Blocks (L = 3 Layers)                       |
|  +-----------------------------------------------------------------------------+  |
|  | LayerNorm -> Multi-Head Self-Attention (h=8 heads) -> Residual Add          |  |
|  | LayerNorm -> ReGLU / GELU Feed-Forward Network (d_ff = 4/3 * d) -> Residual  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                            Classification Head                                    |
| - Extract [CLS] Token Representation z_[CLS]^(L)                                   |
| - LayerNorm -> Linear(d -> 1) -> Sigmoid Output                                   |
+-----------------------------------------------------------------------------------+
```

### 4.2 Optimization & Regularization Design
To prevent overfitting on small-to-moderate clinical cohorts ($N \in [300, 1000]$), a robust deep learning optimization stack is specified:

1. **Activation Function (GELU / ReGLU):**
   GELU (Gaussian Error Linear Unit) provides smooth, non-monotonic gradient flow superior to ReLU:
   $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[ 1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right) \right]$$
2. **Layer Normalization & Pre-Norm Architecture:** Pre-Layer Normalization is applied prior to self-attention and feed-forward sub-layers to stabilize gradient propagation.
3. **Optimizer (AdamW):** Decouples weight decay regularization from gradient updates, preventing coefficient shrinkage distortion:
   $$\mathbf{\theta}_{t+1} = \mathbf{\theta}_t - \eta_t \left( \frac{\mathbf{m}_t}{\sqrt{\mathbf{v}_t} + \epsilon} + \lambda \mathbf{\theta}_t \right)$$
   Parameters: Initial learning rate $\eta_0 = 10^{-4}$, weight decay $\lambda = 10^{-2}$, $\beta_1 = 0.9, \beta_2 = 0.999$.
4. **Learning Rate Scheduler (Cosine Annealing with Warm Restarts):**
   $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left( 1 + \cos\left( \frac{T_{cur}}{T_i}\pi \right) \right)$$
5. **Regularization Suite:**
   * **Attention Dropout ($p_a = 0.2$):** Drops attention matrix connections randomly during training.
   * **Residual Dropout ($p_r = 0.1$):** Applied to feedforward sub-layer outputs.
   * **Label Smoothing ($\epsilon = 0.05$):** Prevents overconfident binary diagnostic predictions.

---

## 5. Expected Outcomes & Phase Summary

* **Expected Outcome:** The FT-Transformer is expected to achieve superior deep feature representation capabilities over standard MLPs, capturing subtle multi-biomarker interaction surfaces and providing clean `[CLS]` latent embeddings for downstream hybrid model coupling.
* **Phase Summary:** Phase 8 formulates a publication-ready deep learning architecture strategy, establishing **FT-Transformer** with GELU activations, Pre-Layer Normalization, and AdamW optimization as the gold-standard deep tabular model for structured multi-disease diagnostics.
