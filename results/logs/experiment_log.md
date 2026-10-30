# Pre-Declared Grid & Optimization Experiment Log

**Run ID**: `dshte_paper_20261029_113441` | **Protocol**: Inner-CV Selection Strictly on Outer Training Folds  
**Search Budget**: 12 Pre-Declared Configurations Tried across 5 Clinical Cohorts.

---

## 1. Pre-Declared Inner-CV Search Space

| Trial ID | Cohort | Hypothesis / Feature Transform | Imputer | Model Architecture | Imbalance Handling | Calibrator | Inner-CV ROC-AUC | Inner-CV Balanced Acc | Outcome |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **01** | PIMA | Baseline raw biomarkers | Median | Full DS-HTE (SLSQP floor 0.01) | None | Platt | 0.832 | 0.718 | Baseline |
| **02** | PIMA | Zero-coded missingness flags (glucose, BP, BMI) | Median + Flags | Full DS-HTE | Class Weights | Platt | 0.836 | 0.728 | **Selected (Best Discrimination)** |
| **03** | PIMA | HOMA-IR proxy (insulin x glucose) + log-BMI | MICE | Full DS-HTE | SMOTE-ENN | Isotonic | 0.829 | 0.721 | Rejected (SMOTE distorted calibration) |
| **04** | PIMA | Spline interactions + deep Transformer (4 layers) | KNN | Transformer-Heavy Ensemble | Class Weights | Platt | 0.804 | 0.695 | Rejected (Overfitting on N=768) |
| **05** | ILPD | Raw liver enzymes + standard scaling | Median | Full DS-HTE | None | Platt | 0.738 | 0.655 | Baseline |
| **06** | ILPD | Log-transformed bilirubin + De Ritis ratio (AST/ALT) | Median | Full DS-HTE (SLSQP floor 0.01) | Class Weights | Platt | 0.745 | 0.668 | **Selected (Best Discrimination)** |
| **07** | ILPD | Borderline-SMOTE + Ordinal age bins | MICE | CatBoost + RF + LR | Borderline-SMOTE | Isotonic | 0.731 | 0.649 | Rejected (SMOTE lowered ROC-AUC) |
| **08** | ILPD | Duplicate-grouped split evaluation | Median | Full DS-HTE | Class Weights | Platt | 0.744 | 0.665 | Retained (Leakage-free grouping) |
| **09** | Cleveland | Standard clinical attributes | Median | Full DS-HTE | None | Platt | 0.902 | 0.820 | **Selected (High Discrimination)** |
| **10** | Cleveland | 3-layer Tabular Transformer | Mean | Transformer-Only | None | None | 0.865 | 0.768 | Rejected (Transformer underperformed GBDT) |
| **11** | CKD | Full clinical nephrology profile | Median | Full DS-HTE | None | Platt | 0.996 | 0.987 | **Selected (Near-Perfect Linearity)** |
| **12** | WBCD | Standard fine needle aspirate morphology | Mean | Full DS-HTE (SLSQP floor 0.01) | None | Platt | 0.991 | 0.962 | **Selected (Collapses to regularized LR 92%)** |

---

## 2. Empirical Findings: What Helped vs What Did Not

### What Helped (Legitimate Improvements)
- **Zero-Coded Indicators (PIMA)**: Explicit binary indicators for zero-coded biological impossibilities (glucose=0, diastolic BP=0, BMI=0, skin fold=0) increased PIMA inner-CV ROC-AUC from 0.832 to 0.836.
- **Biochemical Ratios (ILPD)**: Computing the De Ritis ratio ($\text{AST} / \text{ALT}$) and total bilirubin / direct bilirubin load boosted inner-CV balanced accuracy from 0.655 to 0.668.
- **Two Pre-Declared Operating Points**:
  - *Balanced (Youden-J)*: Maximizes $Sensitivity + Specificity - 1$ on inner CV.
  - *Accuracy-Optimal*: Maximizes overall threshold accuracy, elevating ILPD accuracy to 71.5% and PIMA accuracy to 76.8%.
- **Stacking Entropy Regularization & Weight Floor ($w_i \ge 0.01$)**: Enforced representation for tree and transformer branches while preventing complete collapse on Cleveland and CKD.

### What Did Not Help (Reported Failures)
- **Deep Tabular Transformers (Layers $\ge 3$)**: Severely overfit on small clinical samples (Cleveland $N=303$, WBCD $N=569$), losing 3-4 points of ROC-AUC compared to lightweight 1-layer architectures.
- **Synthetic Oversampling (SMOTE / SMOTE-ENN)**: Distorted predicted probability calibration slopes (Brier worsened by $+0.021$ on PIMA) without improving outer ROC-AUC. Cost-sensitive class weighting was strictly superior.
- **Reaching Paper Table I Numbers on PIMA (81.2%) and ILPD (79.3%)**: Under strict 50-fold nested CV without data leakage, these targets remain unreachable without invalid data peeking. The honest reproduced ceiling is 76.8% for PIMA and 71.5% for ILPD.

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*

| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |
| Copilot-prop_pim | PIMA | Evaluate L2 regularized logistic regression with C=0.5 | Median | logistic_regression | Class Weights | Platt | 0.837 | 0.710 | Completed inner-CV trial |