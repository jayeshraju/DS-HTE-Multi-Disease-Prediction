# Model Card: DS-HTE Stacking Meta-Learner

> **Notice**: Research prototype, not a certified clinical diagnostic device.

---

## 1. Model Details
- **Architecture**: Disease-Specific Hybrid Transformer Ensemble (DS-HTE).
- **Component Classifiers**:
  - Tabular Transformer (Self-attention encoder + linear head)
  - LightGBM (Gradient boosted decision trees)
  - CatBoost (Ordered boosting with categorical support)
  - XGBoost (Extreme gradient boosting)
  - Random Forest (Bagged decision trees)
  - Regularized Logistic Regression (L2 penalty)
- **Meta-Learner**: Constrained Sequential Least Squares Programming (SLSQP) optimizing log-loss with non-negativity constraint ($w_i \ge 0.01$) and entropy diversity penalty ($\lambda = 0.005$).
- **Calibration**: Temperature scaling / Platt logistic calibration.
- **Decision Rule**: Two pre-declared thresholds: Balanced (Youden's J) and Accuracy-Optimal.

---

## 2. Intended Use & Clinical Scope
- **Intended Use**: Multi-cohort risk stratification research, clinical decision support benchmarking, and comparative model auditing.
- **Out of Scope**: Primary autonomous diagnosis, triage in acute emergencies, treatment prescription, or pediatric patients.

---

## 3. Empirical Performance Profile (50-Fold Nested CV)

| Cohort | Balanced Acc (%) | Accuracy-Optimal (%) | ROC-AUC | Brier Score | ECE | Majority Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **PIMA** | 73.7% | 76.8% | 0.836 | 0.156 | 0.062 | 65.1% |
| **Cleveland** | 82.0% | 82.7% | 0.902 | 0.127 | 0.048 | 54.1% |
| **CKD** | 98.7% | 98.7% | 0.996 | 0.010 | 0.008 | 62.5% |
| **ILPD** | 62.4% | 71.5% | 0.745 | 0.178 | 0.051 | 71.5% |
| **WBCD** | 96.2% | 96.5% | 0.991 | 0.026 | 0.014 | 62.7% |

---

## 4. Known Limitations & Failure Modes
- **Linear Dominance**: On linearly separable cohorts (WBCD), meta-learner weights collapse to Logistic Regression ($w_{LR} = 0.922$).
- **Imbalance Trade-offs**: On high-prevalence majority cohorts (ILPD 71.5%), Youden-J thresholding depresses raw accuracy to 62.4% while preserving clinical sensitivity.
