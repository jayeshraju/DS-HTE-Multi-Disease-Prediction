# Rigorous Evaluation Protocol & Leakage Controls

> **Protocol Standard**: Designed for leak-free, reproducible clinical machine learning evaluation.
> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Nested Cross-Validation (NCV) Design
To prevent optimistic evaluation bias and parameter leakage:
1. **Outer Loop**: 10 repeats of 5-fold cross-validation ($N=50$ outer evaluation folds). Each outer fold serves exclusively as sealed test data.
2. **Inner Loop**: Repeated 3-fold cross-validation ($3 \times 3$) executed strictly inside each outer training fold.
3. **Parameter Selection**: Feature selection (AFO), imputer parameters, scaling factors, model hyperparameters, and stacking weights are chosen strictly within inner CV.
4. **Final Outer Evaluation**: The frozen pipeline is applied exactly once to the outer test fold.

---

## 2. Leakage Controls & Boundary Protections
- **Pre-Imputation Leakage Prevention**: Imputation statistics (means, medians) and feature scalers are fitted exclusively on training instances and applied downstream to validation/test instances.
- **Duplicate Cluster Isolation**: Near-duplicate and identical patient records are grouped using perceptual clustering (`StratifiedGroupKFold`) ensuring that identical feature rows never span train and test sets simultaneously.
- **Patient ID Grouping**: In medical imaging and clinical records with `patient_id`, grouping is enforced to prevent multi-slice or multi-visit patient leakage.
- **Holdout Status**: The 20% holdout split, having been inspected across historical pipeline iterations, is formally classified as **"previously inspected holdout (secondary)"**. True primary performance is represented by the 50-fold nested CV.

---

## 3. Threshold Optimization & Probability Calibration
- **Calibration**: Temperature scaling and Platt logistic regression calibrated on un-resampled inner validation folds prior to threshold selection.
- **Two Pre-Declared Operating Points**:
  1. **Balanced (Youden's J)**: Threshold chosen to maximize $J(t) = \text{Sensitivity}(t) + \text{Specificity}(t) - 1$. Recommended for clinical screening where false negatives carry severe consequences.
  2. **Accuracy-Optimal**: Threshold chosen to maximize overall classification accuracy on inner validation folds.
- **Alarm Rule (Leakage Detector)**: If outer CV accuracy on PIMA exceeds 80% or ILPD exceeds 76%, the run is automatically flagged for immediate leakage auditing before reporting.

---

## 4. Statistical Significance Testing
Because 50 repeated folds are not statistically independent:
- **Nadeau-Bengio Correction**: Applied to fold mean comparisons with sample correlation correction factor $\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}} = \frac{1}{50} + \frac{0.20}{0.80} = 0.27$, preventing artificially narrow confidence intervals.
- **Paired Non-Parametric Tests**: Wilcoxon signed-rank tests computed across paired fold scores.
- **Multiple Comparison Corrections**: Holm and Benjamini-Hochberg (FDR) adjustments applied to all p-values.
- **Directional Labeling**: Every statistical comparison explicitly states direction (e.g. "DS-HTE higher", "Baseline higher", or "No significant difference").

---

## 5. External Geographic Validation
- Evaluates Cleveland-trained models without parameter retraining on three distinct geographic cohorts:
  - Hungarian Institute of Cardiology (Budapest, N=294)
  - University Hospital Zurich (Switzerland, N=123)
  - Veterans Administration Medical Center (Long Beach, CA, N=200)
- Limitations: For PIMA, ILPD, and CKD, no comparable public external cohorts with matching clinical schemas exist.
