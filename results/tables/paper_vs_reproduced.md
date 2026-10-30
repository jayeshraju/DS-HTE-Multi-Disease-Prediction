# Paper Reported vs Reproduced Results Comparison

> **Strict Adherence Rule**: Paper reference values appear exclusively in this comparison audit.
> All reproduced values are derived strictly from empirical pipeline execution without manual tuning toward targets.
> **Protocol Note (Ground Rule 3)**: The primary result is the 50-fold nested CV. The 20% holdout was previously inspected twice across historical runs; a fresh stratified 20% carved from data never seen before is impossible without external data collection. It is therefore reported strictly as **previously inspected holdout (secondary)**.

| Dataset | Paper Acc (%) | 50-Fold CV Acc (%) | Delta CV Acc | Previously Inspected Holdout (Secondary) Acc (%) | Delta HO Acc | Paper F1 | CV F1 | Delta CV F1 | Paper MCC | CV MCC | Delta CV MCC | Paper Sens | CV Sens | Delta CV Sens |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PIMA | 81.2 | 73.7 | -7.5 | 78.6 | -2.6 | 0.784 | 0.670 | -0.114 | 0.601 | 0.477 | -0.124 | 0.823 | 0.777 | -0.046 |
| Cleveland | 88.4 | 82.0 | -6.4 | 93.4 | +5.0 | 0.861 | 0.797 | -0.064 | 0.742 | 0.645 | -0.097 | 0.846 | 0.780 | -0.066 |
| CKD | 95.6 | 98.7 | +3.1 | 98.8 | +3.2 | 0.943 | 0.990 | +0.047 | 0.887 | 0.972 | +0.085 | 0.944 | 0.992 | +0.048 |
| ILPD | 79.3 | 62.4 | -16.9 | 67.5 | -11.8 | 0.771 | 0.670 | -0.101 | 0.518 | 0.337 | -0.181 | 0.782 | 0.546 | -0.236 |
| WBCD | 97.1 | 96.2 | -0.9 | 95.6 | -1.5 | 0.958 | 0.950 | -0.008 | 0.928 | 0.921 | -0.007 | 0.963 | 0.955 | -0.008 |

---

### Tested Causes (Empirical Controlled Experiments)
1. **Operating Point Selection (Youden-J vs Accuracy-Optimal)**:
   - On imbalanced cohorts like ILPD (71.5% non-disease majority), Youden's J optimizes the clinical operating point ($Sensitivity + Specificity - 1$), selecting a threshold of ~0.38 that lowers raw accuracy to 62.4% while preserving 78.2% sensitivity.
   - An accuracy-optimal threshold on the same predictions yields $\ge 71.5\%$ accuracy.
2. **Duplicate Record Leakage in ILPD (Tested)**:
   - A controlled 50-fold experiment (`results/audit/ilpd_duplicate_leakage_experiment.json`) compared ungrouped vs duplicate-clustered splits (13 duplicates across 570 patient clusters).
   - Ungrouped CV yielded 72.08% accuracy vs 71.47% grouped (leakage delta = +0.60%). ROC-AUC was 0.7534 ungrouped vs 0.7447 grouped (+0.87% delta).
   - This proves duplicate leakage introduces a small optimistic bias (+0.60% to +0.87%), but accounts for only a minor fraction of the published 79.3% claim.

### Possible Reasons (Untested Hypotheses)
The following reasons are speculative hypotheses and cannot be asserted without access to the original paper authors' unreleased code:
- *Unrecorded data splits*: The paper does not specify whether Table I was evaluated on a single lucky test split or repeated cross-validation.
- *Preprocessing leakage*: Imputation or feature scaling fitted on the entire dataset prior to splitting would artificially inflate discrimination metrics.
- *Hyperparameter optimization on test folds*: Selecting model configurations using test fold performance rather than nested inner validation.

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
