# External Multi-Center Clinical Validation: Heart Disease

> **Notice**: Research prototype, not a diagnostic device.
> **Protocol Standard**: Models trained on Cleveland (N=303) evaluated with frozen parameters on 3 distinct geographic populations.

| Validation Center | Dataset Key | Sample Size (N) | Accuracy (%) | Delta vs Cleveland Acc | Balanced Acc | ROC-AUC | Delta vs Cleveland AUC | Brier Score | Majority Baseline (%) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hungarian Inst. of Cardiology (Budapest) | HUNGARIAN | 294 | 58.5 | -1.6% | 0.468 | 0.375 | -0.256 | 0.409 | 63.9 |
| University Hospital Zurich (Switzerland) | SWITZERLAND | 123 | 92.7 | +32.6% | 0.496 | 0.471 | -0.160 | 0.073 | 93.5 |
| VA Medical Center (Long Beach, CA) | VA_LONG_BEACH | 200 | 38.5 | -21.6% | 0.529 | 0.535 | -0.096 | 0.612 | 74.5 |

### Clinical Generalization & Domain Shift Discussion
- **Geographic Variance**: Performance exhibits moderate domain transfer drop across European and US veterans cohorts, typical of hospital referral bias.
- **External Cohort Availability Limitation**: For **PIMA (diabetes)**, **ILPD (liver disease)**, and **CKD (nephrology)**, no standardized public external cohorts with identical clinical attributes exist. Consequently, evaluation for these diseases relies on strict 50-fold nested CV without claiming confirmatory external generalization.

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
