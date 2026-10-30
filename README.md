# DS-HTE: Adaptive Explainable Multi-Disease Risk Prediction Framework

[![Python](https://img.shields.io/badge/Python-3.13.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0.0-teal.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://reactjs.org)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(73%2F73)-success.svg)]()
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

> **Clinical Disclaimer**: Research prototype, not a certified medical diagnostic device. Not authorized for independent clinical diagnosis, triage, or prescription.

---

## Overview

The **DS-HTE Framework** is a clinical risk prediction system that combines:
1. **Dynamic Feature Optimization (AFO)**: Multi-method consensus ranking (SHAP, Permutation, Mutual Information, L1-logistic) with redundancy pruning.
2. **Disease-Specific Hybrid Transformer Ensembles**: Non-negative constrained SLSQP stacking meta-learner combining Tabular Transformers, LightGBM, CatBoost, XGBoost, Random Forest, and regularized Logistic Regression.
3. **Rigorous Nested Evaluation**: 50-fold nested repeated cross-validation with Nadeau-Bengio corrected confidence intervals, two pre-declared operating points (Balanced vs Accuracy-Optimal), and trivial majority-rate baselines.
4. **Uncertainty & Safety**: Split-conformal prediction sets ($1 - \alpha = 0.90$), Mahalanobis distance out-of-distribution (OOD) detection, and automated "Needs Expert Review" abstention.
5. **Medical Image Lab**: Single-slice 2D radiograph classification (ResNet-18, DenseNet-121) with Layer-4 Grad-CAM saliency heatmaps, DICOM PHI de-identification, and MedMNIST v2 benchmarks.
6. **DS-HTE AI Copilot**: A tool-using assistant with 16 allow-listed tools, deterministic state machine, zero-hallucination **Grounding Verifier**, and sandboxed inner-CV optimization.

---

## Quick Start (Windows / PowerShell)

```powershell
# 1. Setup virtual environment and dependencies
make setup

# 2. Gate 0: Independent Phase 1 verification
make verify-phase1

# 3. Run full 50-fold cross-validation pipeline
make run PROFILE=paper

# 4. Run external multi-center cardiac validation
make external-validate

# 5. Run medical image benchmarks (MedMNIST)
make image-bench

# 6. Execute complete test suite
make test

# 7. Build UI and launch dashboard
make ui-build
make ui
```
Open **`http://localhost:8000/`** in your browser.

---

## Project Documentation Index

- **System Architecture**: [`docs/architecture.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/architecture.md)
- **Pipeline Workflow**: [`docs/workflow.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/workflow.md)
- **Evaluation Protocol**: [`docs/protocol.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/protocol.md)
- **Dataset Datasheets**: [`docs/datasheets/`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/datasheets/)
  - [PIMA Diabetes](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/datasheets/pima.md)
  - [Cleveland Heart Disease](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/datasheets/cleveland.md)
  - [Chronic Kidney Disease](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/datasheets/ckd.md)
  - [Indian Liver Patient](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/datasheets/ilpd.md)
  - [Wisconsin Breast Cancer](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/datasheets/wbcd.md)
- **Model Cards**: [`docs/model_cards/`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/model_cards/)
- **AI Copilot User Guide**: [`docs/copilot_guide.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/copilot_guide.md)
- **UI User Guide**: [`docs/user_guide.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/user_guide.md)
- **API Reference**: [`docs/api.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/api.md)
- **Phase 1 Errata & Changelog**: [`docs/changelog_phase1_to_phase2.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/changelog_phase1_to_phase2.md)
- **Claims Registry**: [`docs/claims.yaml`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/claims.yaml)
- **Test Inventory**: [`docs/test_inventory.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/test_inventory.md)
- **Reproducibility Manifest**: [`docs/reproducibility.md`](file:///d:/study_files/PROJECTS/MULTI_DIS_PREDICT/docs/reproducibility.md)
