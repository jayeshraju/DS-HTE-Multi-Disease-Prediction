# Scientific Limitations, Governance, and Ethical Disclosures

> **Mandatory Notice**: Research prototype, not a diagnostic device.

---

## 1. Scientific Limitations

1. **Scope of Medical Imaging (v1)**: The medical vision module is strictly designed for **2D single-slice image classification** (binary or multiclass). Volumetric 3D reconstructions (CT/MRI), object detection, and spatial lesion segmentation are out of scope.
2. **Dependence on Labelled Ground Truth & Patient IDs**: Unlabelled images cannot be classified without prior supervised training. When patient IDs are omitted, cross-validation grouping relies on perceptual hashing, which cannot eliminate potential patient leakage if distinct slices vary in pixel representation.
3. **Absence of Public External Validation for PIMA, ILPD, and CKD**: While Cleveland cardiac models were validated across Hungarian, Swiss, and Long Beach cohorts, no standardized public external cohorts with identical clinical schemas exist for outpatient liver, diabetes, or nephrology records. Using historical holdouts as "confirmatory" is mathematically invalid once inspected.
4. **Ensemble Collapse on Linearly Separable Data**: On datasets with pronounced linear boundaries (WBCD), non-negative SLSQP optimization assigns 92.2% of meta-learner weight to regularized Logistic Regression, providing zero marginal benefit from non-linear or Transformer branches.
5. **Pre-Declared Configuration Trials Budget**: In total, 12 configurations were explored in inner-CV search space. In accordance with honest ML reporting, higher exploration budgets correspond to higher statistical optimism; all attempts (failures and successes) are permanently logged in `results/logs/experiment_log.md`.

---

## 2. Ethical Principles & Clinical Safety

- **Not a Diagnostic Device**: The software has not received clinical clearance (e.g. FDA 510(k), CE mark) and must not be used for medical decisions, diagnosis, or treatment prescription without certified clinician oversight.
- **Uncertainty & Abstention**: Split-conformal prediction sets and Mahalanobis distance OOD detectors enforce an "Abstain / Needs Expert Review" policy on ambiguous predictions.
- **Privacy & PHI Protection**: Local offline processing with automated DICOM header de-identification prevents inadvertent exposure of Protected Health Information.
