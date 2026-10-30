# Phase 12: Model Evaluation — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction & Evaluation Rigor

Evaluating medical prediction algorithms requires an exhaustive, statistically rigorous validation protocol. Relying solely on standard accuracy or ROC-AUC is insufficient and potentially dangerous in diagnostic settings: high accuracy can mask complete minority class failure, while ROC-AUC can present an overly optimistic assessment under severe class imbalance.

This phase establishes a publication-grade evaluation methodology combining **10 statistical classification metrics**, **Repeated Stratified 10-Fold Cross-Validation**, **Non-Parametric Hypothesis Testing (McNemar, Wilcoxon, Friedman)**, and **Probability Calibration (Brier Score)** to validate the proposed Hybrid HDA-GB model against all baseline estimators across five disease benchmarks.

---

## 2. Mathematical Formulations of Diagnostic Evaluation Metrics

```
                                  +-------------------------------------------------------+
                                  |       Diagnostic Evaluation Metrics Taxonomy          |
                                  +-------------------------------------------------------+
                                                              |
          +---------------------------+-----------------------+-----------------------+---------------------------+
          |                           |                                               |                           |
          v                           v                                               v                           v
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| Threshold-Based   |       | Probabilistic     |                           | Correlation &     |       | Non-Parametric    |
| Confusion Matrix  |       | Rank Curves       |                           | Agreement         |       | Significance Tests|
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| - Sensitivity/Rec |       | - ROC-AUC         |                           | - MCC (Matthews)  |       | - McNemar's Test  |
| - Specificity     |       | - PR-AUC          |                           | - Cohen's Kappa   |       | - Wilcoxon Signed |
| - Precision / F1  |       | - Brier Score     |                           | - Balanced Acc.   |       | - Friedman Test   |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
```

### 2.1 Threshold-Based & Discriminative Metrics

1. **Sensitivity / Recall (True Positive Rate - TPR):**
   $$\text{Recall} = \frac{TP}{TP + FN}$$
   *Clinical Significance:* Measures the proportion of actual diseased patients correctly identified. High recall minimizes false negatives ($FN$).

2. **Specificity (True Negative Rate - TNR):**
   $$\text{Specificity} = \frac{TN}{TN + FP}$$
   *Clinical Significance:* Measures the proportion of healthy individuals correctly classified as non-diseased, minimizing unnecessary secondary invasive testing.

3. **Precision (Positive Predictive Value - PPV):**
   $$\text{Precision} = \frac{TP}{TP + FP}$$
   *Clinical Significance:* Probability that a patient flagged as high risk actually harbors the condition.

4. **Balanced Accuracy:**
   $$\text{Balanced Accuracy} = \frac{\text{Sensitivity} + \text{Specificity}}{2}$$
   *Clinical Significance:* Evaluates raw performance un-skewed by majority class proportions.

5. **Harmonic Mean F1-Score:**
   $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 TP}{2 TP + FP + FN}$$

6. **Matthews Correlation Coefficient (MCC):**
   $$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$
   *Clinical Significance:* The single most informative binary metric; returns $+1$ for perfect prediction, $0$ for random chance, and $-1$ for inverse prediction. Unlike F1-Score, MCC is symmetric and invariant to class label swapping.

7. **Cohen’s Kappa ($\kappa$):**
   $$\kappa = \frac{p_o - p_e}{1 - p_e}, \quad p_o = \frac{TP+TN}{N}, \quad p_e = \frac{(TP+FP)(TP+FN) + (TN+FP)(TN+FN)}{N^2}$$
   *Clinical Significance:* Quantifies inter-annotator agreement above random chance between AI predictions and ground-truth clinical diagnoses.

---

### 2.2 Probabilistic & Calibration Metrics

8. **Area Under Receiver Operating Characteristic Curve (ROC-AUC):**
   $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) dt$$

9. **Area Under Precision-Recall Curve (PR-AUC):**
   $$\text{PR-AUC} = \int_{0}^{1} \text{Precision}(\text{Recall}^{-1}(t)) dt$$
   *Clinical Significance:* Primary metric for imbalanced disease datasets ($IR > 3:1$).

10. **Brier Score (Probability Calibration Error):**
    $$\text{Brier Score} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$$
    *Clinical Significance:* Measures the mean squared error of predicted probabilities. A lower Brier score ($\approx 0$) guarantees reliable clinical risk probability calibration.

---

## 3. Comprehensive IEEE-Style Metrics Comparison

TABLE I: IEEE-FORMATTED METRIC EVALUATION COMPARISON

| Metric Name | Mathematical Range | Sensitivity to Imbalance | Clinical Primary Objective | Target Benchmark Threshold |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | $[0.0, 1.0]$ | Extremely High (Misleading) | Overall raw correct fraction | $\ge 0.90$ (Reference only) |
| **Sensitivity (Recall)** | $[0.0, 1.0]$ | Low | **Minimizing False Negatives ($FN$)** | **$\ge 0.92$ (Primary)** |
| **Specificity** | $[0.0, 1.0]$ | Low | Minimizing False Positives ($FP$) | $\ge 0.88$ |
| **Precision (PPV)** | $[0.0, 1.0]$ | High | Diagnostic predictive value | $\ge 0.85$ |
| **F1-Score** | $[0.0, 1.0]$ | Moderate | Harmonic balance of Precision & Recall | $\ge 0.90$ |
| **ROC-AUC** | $[0.0, 1.0]$ | High (Optimistic) | Global ranking capability | $\ge 0.95$ |
| **PR-AUC** | $[0.0, 1.0]$ | **Invariant (Robust)** | **Minority class precision-recall trade-off** | **$\ge 0.92$ (Primary)** |
| **Matthews (MCC)** | $[-1.0, +1.0]$ | **Zero (Symmetric)** | **Complete binary correlation measure** | **$\ge 0.82$ (Gold Standard)** |
| **Cohen’s Kappa ($\kappa$)** | $[-1.0, +1.0]$ | Low | Agreement beyond random chance | $\ge 0.80$ |
| **Brier Score** | $[0.0, 1.0]$ | Low | Probability calibration accuracy | $\le 0.08$ |

---

## 4. Non-Parametric Statistical Hypothesis Testing

To prove that the proposed Hybrid HDA-GB architecture achieves statistically significant performance gains over baseline models rather than stochastic variation:

1. **McNemar’s Test (Paired Contingency Test):** Evaluates paired classification errors on the sealed test set between Model A (HDA-GB) and Model B (XGBoost Baseline):
   $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} \sim \chi^2(df=1)$$
   where $n_{01}$ is the number of instances misclassified by A but correct by B, and $n_{10}$ is correct by A but misclassified by B. Rejects null hypothesis $H_0$ if $p < 0.05$.
2. **Wilcoxon Signed-Rank Test:** Evaluates paired 10-fold cross-validation F1-Score distributions across folds.
3. **Friedman Test & Nemenyi Post-Hoc Test:** Performs non-parametric multi-model ranking across all 5 disease datasets. Rejects $H_0$ if critical difference $\text{CD} = q_{\alpha} \sqrt{\frac{K(K+1)}{6N}}$ is exceeded.

```
========================= STATISTICAL SIGNIFICANCE TESTING FLOW =========================

  [ 10-Fold Out-of-Fold (OOF) Metric Distributions across Models ]
                               |
                               v
  [ Step 1: Omnibus Friedman Test across Models (p < 0.05) ]
                               |
            +------------------+------------------+
            | (Reject H0)                         | (Fail to Reject)
            v                                     v
  [ Step 2: Nemenyi Post-Hoc Test ]    [ Models Statistically Equivalent ]
  Compute Critical Difference (CD)
  Compare Pairwise Model Ranks
                               |
                               v
  [ Step 3: Paired McNemar Test ] -> Report p-value & Cohen's d Effect Size
```

---

## 5. Expected Outcomes & Phase Summary

* **Expected Empirical Results:** The proposed Hybrid HDA-GB model is projected to achieve statistically significant superior performance ($p < 0.01$ under McNemar & Wilcoxon tests) over standard XGBoost and FT-Transformer baselines, achieving $\text{MCC} \ge 0.85$, $\text{F1-Score} \ge 0.92$, and $\text{PR-AUC} \ge 0.94$.
* **Phase Summary:** Phase 12 formulates a rigorous, IEEE-standard model evaluation framework combining 10 complementary diagnostic metrics, calibration analysis, and non-parametric hypothesis testing.
