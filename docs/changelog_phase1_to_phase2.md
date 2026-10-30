# Changelog: Phase 1 to Phase 2 (with Phase 1 Errata Statement)

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Phase 1 Errata Statement & Corrected Benchmark Values

During rigorous replication and audit of the published Phase 1 paper ("Adaptive Explainable Multi-Disease Prediction Framework..."), several published Table I values could not be reproduced under leak-free 50-fold nested cross-validation. 

Below is the complete errata table documenting the published numbers, the verified reproduced values, and the empirical causes:

| Dataset | Published Acc (%) | Reproduced 50-Fold CV Acc (%) | Delta CV | Previously Inspected Holdout Acc (%) | Delta HO | Published F1 | Repr. CV F1 | Published MCC | Repr. CV MCC | Primary Cause of Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **PIMA** | 81.2 | **73.7** (Bal) / **76.8** (Acc-Opt) | -7.5 / -4.4 | 78.6 | -2.6 | 0.784 | 0.670 | 0.601 | 0.477 | Unrecorded data peeking / threshold tuning in published run |
| **Cleveland** | 88.4 | **82.0** (Bal) / **82.7** (Acc-Opt) | -6.4 / -5.7 | 93.4 | +5.0 | 0.861 | 0.797 | 0.742 | 0.645 | High variance across outer folds; published value matches lucky split |
| **CKD** | 95.6 | **98.7** | +3.1 | 98.8 | +3.2 | 0.943 | 0.990 | 0.887 | 0.972 | Strong linear separability ($hemo$, $sc$); reproduced performance exceeds paper |
| **ILPD** | 79.3 | **62.4** (Bal) / **71.5** (Acc-Opt) | -16.9 / -7.8 | 67.5 | -11.8 | 0.771 | 0.670 | 0.518 | 0.337 | Imbalance penalty under Youden-J screening threshold (~0.38) vs 71.4% majority rate |
| **WBCD** | 97.1 | **96.2** | -0.9 | 95.6 | -1.5 | 0.958 | 0.950 | 0.928 | 0.921 | Meta-learner collapses to Logistic Regression ($w_{LR} = 0.922$) |

---

## 2. Key Phase 2 Architectural & Scientific Upgrades

### A. Protocol Integrity & Provenance (Gate 0)
- **Sidecar Provenance**: Every output file in `results/` carries a cryptographic `*.provenance.json` sidecar linking file hashes to producing scripts and run IDs.
- **Two Pre-Declared Operating Points**: Both **Balanced (Youden's J)** and **Accuracy-Optimal** thresholds are explicitly declared and reported alongside trivial majority-rate baselines.
- **Leakage Controls**: Enforced `StratifiedGroupKFold` clustering for ILPD duplicates and verified that outer-test labels never reach threshold selectors.

### B. DS-HTE AI Copilot (Gate 1)
- Built a deterministic, 16-tool workflow agent featuring a zero-hallucination **Grounding Verifier** that cross-examines draft text against computed artifact arrays.
- Enforced prompt-injection neutralization and a sandboxed development improver with a strict 40-trial budget cap.

### C. Multi-Modal Vision & Geographic Validation (Gate 2)
- Added the **Image Lab** for single-slice radiograph classification (ResNet-18, Grad-CAM saliency, Mahalanobis OOD detection).
- Evaluated on official MedMNIST test splits (PneumoniaMNIST: 82.21% acc vs 62.5% test majority; BreastMNIST: 80.13% acc vs 73.08% test majority).
- Conducted external multi-center validation on Hungarian, Swiss, and VA Long Beach cardiac cohorts.
- Integrated **Split-Conformal Prediction Sets** with empirical marginal coverage verified within $\pm 2.5$ percentage points of nominal 90% coverage.
