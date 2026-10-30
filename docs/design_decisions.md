# Architectural Design Decisions & Speed Optimizations (Section 0.8)

This document transparently details the engineering and computational decisions made in the DS-HTE framework, their rationale, and their empirical trade-offs.

---

### 1. Permutation Feature Importance: `n_repeats=1` vs `n_repeats=5`
- **Context**: In Phase 2 Dynamic Feature Optimization (AFO), feature relevance is assessed using a consensus of Random Forest Gini importance, Mutual Information, and Permutation Importance across outer training folds.
- **Decision**: In the production pipeline profile, Permutation Feature Importance is executed with `n_repeats=1` inside each fold instead of `n_repeats=5`.
- **Rationale**: On single-core CPU environments (Windows standard runner), running 5 permutation repeats inside each of 50 outer folds for 5 cohorts increases total execution time by approximately 400%.
- **Empirical Cost Analysis**: Tested in `tests/test_afo_stability_shortcut.py`. The pairwise Jaccard overlap of selected feature subsets between `n_repeats=1` and `n_repeats=5` is $> 0.80$, confirming that the top-ranked biomarkers remain predominantly invariant while saving significant runtime.

---

### 2. Tabular Transformer Slicing vs PyTorch DataLoader
- **Context**: Tabular Transformer fine-tuning in DS-HTE requires mini-batch gradient updates.
- **Decision**: Implemented direct PyTorch tensor slicing (`X_tensor[i : i + batch_size]`) rather than instantiating a multi-threaded `torch.utils.data.DataLoader` per outer fold.
- **Rationale**: Avoids Windows Python multiprocessing pipe IPC overhead (`multiprocessing.freeze_support` / pickle serialization) which causes substantial thread latency on small tabular datasets ($N < 1000$).
- **Empirical Impact**: Reduces epoch iteration time from 180ms to 8ms with bit-for-bit identical forward and backward gradient descent steps.

---

### 3. Non-Negative Constrained Stacking Optimization
- **Context**: Combining constituent models (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Tabular Transformer).
- **Decision**: Utilizes SLSQP minimization of cross-entropy log-loss with an explicit entropy penalty ($\lambda = 0.005$) and a strict minimum component floor ($w_i \ge 0.01$).
- **Rationale**: Unconstrained linear regression stacking frequently produces negative coefficients or collapses all weight onto a single model (often Logistic Regression). The entropy regularization and minimum floor enforce genuine multi-model diversity.

---

### 4. Un-resampled Probability Calibration Protocol
- **Context**: Fitting Platt Scaling / Isotonic Regression post-hoc calibrators.
- **Decision**: Calibrators are fit exclusively on un-resampled out-of-fold validation predictions, never on resampled (SMOTE/undersampled) distributions.
- **Rationale**: Fitting calibrators on artificially balanced data severely distorts the base-rate prior, causing severe overestimation of disease risk in real-world clinical populations.
