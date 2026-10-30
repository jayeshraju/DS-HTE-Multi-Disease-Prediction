# Ensemble Weights & Model Diversity Audit (Section 0.1)

> **Audit Rule**: Any component with mean weight > 0.90 across outer folds is flagged as ensemble collapse.

### [!] Flagged Collapsed Components (Mean Weight > 0.90)

| Cohort | Component | Mean Weight | Std Dev | Finding |
| :--- | :--- | ---: | ---: | :--- |
| WBCD | `lr` | 0.9217 | 0.0489 | Reduces primarily to LR |

### Full Component Weight Distribution Across 50 Folds

| Cohort | Component | Mean Weight | Std Dev | Min Weight | Max Weight |
| :--- | :--- | ---: | ---: | ---: | ---: |
| PIMA | `catboost` | 0.0104 | 0.0019 | 0.0100 | 0.0213 |
| PIMA | `lgbm` | 0.0923 | 0.0688 | 0.0141 | 0.3513 |
| PIMA | `lr` | 0.6582 | 0.1184 | 0.3104 | 0.8861 |
| PIMA | `rf` | 0.1767 | 0.0984 | 0.0246 | 0.4340 |
| PIMA | `transformer` | 0.0288 | 0.0257 | 0.0100 | 0.1272 |
| PIMA | `xgb` | 0.0336 | 0.0214 | 0.0100 | 0.0874 |
| CLEVELAND | `catboost` | 0.0100 | 0.0003 | 0.0100 | 0.0119 |
| CLEVELAND | `lgbm` | 0.0159 | 0.0195 | 0.0100 | 0.1332 |
| CLEVELAND | `lr` | 0.7204 | 0.1657 | 0.2049 | 0.9500 |
| CLEVELAND | `rf` | 0.1716 | 0.1559 | 0.0100 | 0.6767 |
| CLEVELAND | `transformer` | 0.0682 | 0.0811 | 0.0100 | 0.4373 |
| CLEVELAND | `xgb` | 0.0139 | 0.0184 | 0.0100 | 0.1145 |
| CKD | `catboost` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| CKD | `lgbm` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| CKD | `lr` | 0.6547 | 0.1940 | 0.1100 | 0.9383 |
| CKD | `rf` | 0.3053 | 0.1940 | 0.0217 | 0.8500 |
| CKD | `transformer` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| CKD | `xgb` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| ILPD | `catboost` | 0.0141 | 0.0076 | 0.0100 | 0.0396 |
| ILPD | `lgbm` | 0.0799 | 0.0758 | 0.0100 | 0.3185 |
| ILPD | `lr` | 0.6029 | 0.1349 | 0.3295 | 0.9370 |
| ILPD | `rf` | 0.2112 | 0.1136 | 0.0100 | 0.4408 |
| ILPD | `transformer` | 0.0270 | 0.0318 | 0.0100 | 0.1742 |
| ILPD | `xgb` | 0.0649 | 0.0515 | 0.0100 | 0.2245 |
| WBCD | `catboost` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| WBCD | `lgbm` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
| WBCD | `lr` | 0.9217 | 0.0489 | 0.7576 | 0.9500 |
| WBCD | `rf` | 0.0365 | 0.0465 | 0.0100 | 0.2024 |
| WBCD | `transformer` | 0.0118 | 0.0126 | 0.0100 | 0.0994 |
| WBCD | `xgb` | 0.0100 | 0.0000 | 0.0100 | 0.0100 |
