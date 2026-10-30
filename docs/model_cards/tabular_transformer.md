# Model Card: Tabular Transformer Classifier

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Architecture Overview
- **Architecture**: Self-attention multi-head tabular encoder with feature embedding projections and MLP classification head.
- **Layers**: 1–2 Transformer encoder layers (lightweight design tuned for small $N \le 1000$ clinical cohorts).
- **Hidden Dimensions**: $d_{model} = 32$, 2 attention heads, feed-forward dimension 64, dropout rate 0.10.
- **Optimizer**: AdamW ($\text{lr}=10^{-3}$, weight decay $10^{-4}$), binary cross-entropy loss with early stopping.

---

## 2. Empirical Findings: When it Helps vs When it Overfits
- **Small Sample Overfitting**: On compact clinical cohorts ($N < 500$, such as Cleveland and WBCD), deeper transformers ($\ge 3$ layers) overfit severely, trailing gradient boosted trees by 3–4 points in ROC-AUC.
- **Stabilization**: Restricting depth to 1 layer with heavy dropout and embedding normalization stabilized convergence, enabling competitive representation inside the DS-HTE ensemble.
