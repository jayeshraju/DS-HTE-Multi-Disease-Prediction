# Proposed Scientific Revisions for Paper Text & Discussion

> **Protocol Integrity Rule**: All revised sentences use exclusively reproduced empirical numbers from the 50-fold nested repeated cross-validation protocol. The 20% holdout is designated as **previously inspected holdout (secondary)**.

## 1. Discrepancy Flag in Original Manuscript (Fig. 2-4)
> **Critical Forensic Finding**: In the original manuscript, the caption and discussion for Fig. 2-4 describe a representative validation fold with $n = 10$ and an ROC-AUC of 0.700. Under 5-fold CV on 80% development data, the minimum fold size across cohorts is $N=48$. The proposed text replaces these sections with empirically verified 50-fold out-of-fold statistics.

## 2. Revised Abstract Excerpt
The proposed framework was evaluated across five clinical cohorts under a rigorous, leakage-free 50-fold repeated cross-validation protocol and secondary holdout evaluation. Under the pre-declared balanced operating point (Youden's J for screening), the framework achieved 50-fold cross-validation accuracies of:
- **WBCD Breast Cancer**: 96.2% (95% Nadeau-Bengio CI: [94.8%, 97.6%], F1: 0.950, MCC: 0.921, ROC-AUC: 0.991)
- **CKD Kidney Disease**: 98.7% (95% Nadeau-Bengio CI: [97.6%, 99.8%], F1: 0.990, MCC: 0.972, ROC-AUC: 0.996)
- **Cleveland Heart Disease**: 82.0% (95% Nadeau-Bengio CI: [78.1%, 85.9%], F1: 0.797, MCC: 0.645, ROC-AUC: 0.902)
- **PIMA Diabetes**: 73.7% balanced / 76.8% accuracy-optimal (95% Nadeau-Bengio CI: [71.5%, 75.9%], F1: 0.670, MCC: 0.477, ROC-AUC: 0.836)
- **ILPD Liver Disease**: 62.4% balanced / 71.5% accuracy-optimal (95% Nadeau-Bengio CI: [59.1%, 65.7%], F1: 0.670, MCC: 0.337, ROC-AUC: 0.736)

Under an accuracy-optimal threshold matching majority-class baseline evaluation, PIMA reaches 76.8% and ILPD reaches 71.5%. The paper's originally reported values of PIMA 81.2% and ILPD 79.3% could not be replicated under leak-free nested cross-validation and should be revised.

## 3. Revised Model Comparison Discussion
Statistical significance tests with Holm and Benjamini-Hochberg multiple-testing corrections indicate that DS-HTE performs competitively with standard tuned gradient-boosted trees and penalized linear models. On clinical cohorts with pronounced tabular structure and moderate sample sizes, hybrid ensembling provides calibrated probability estimates and robust decision support without overclaiming superior accuracy over simpler clinical baselines.

---
*Run ID: `dshte_paper_20261029_113441`*
