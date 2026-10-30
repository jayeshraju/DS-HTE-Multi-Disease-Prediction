# Table V: Ablation Study Across 50 Identical Folds

| Dataset | Ablation Variant | Accuracy (%) | F1 | MCC | ROC-AUC | Brier |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| PIMA | Full DS-HTE | 73.7 | 0.670 | 0.477 | 0.836 | 0.156 |
| PIMA | Minus Feature Optimization | 75.8 | 0.695 | 0.512 | 0.837 | 0.155 |
| PIMA | Minus Imbalance Handling | 73.7 | 0.670 | 0.477 | 0.836 | 0.156 |
| PIMA | Minus Calibration | 74.4 | 0.676 | 0.485 | 0.839 | 0.154 |
| PIMA | Minus Threshold Optimization | 76.8 | 0.622 | 0.475 | 0.836 | 0.156 |
| PIMA | GBDT-Only Ensemble | 73.5 | 0.670 | 0.474 | 0.836 | 0.155 |
| PIMA | Minus Transformer | 73.5 | 0.670 | 0.474 | 0.836 | 0.155 |
| PIMA | LR-Only | 73.7 | 0.672 | 0.480 | 0.836 | 0.156 |
| PIMA | Transformer-Only | 72.8 | 0.565 | 0.412 | 0.807 | 0.173 |
| Cleveland | Full DS-HTE | 82.0 | 0.797 | 0.645 | 0.902 | 0.127 |
| Cleveland | Minus Feature Optimization | 81.0 | 0.786 | 0.629 | 0.902 | 0.129 |
| Cleveland | Minus Imbalance Handling | 82.0 | 0.797 | 0.645 | 0.902 | 0.127 |
| Cleveland | Minus Calibration | 81.8 | 0.796 | 0.643 | 0.902 | 0.127 |
| Cleveland | Minus Threshold Optimization | 82.7 | 0.801 | 0.657 | 0.902 | 0.127 |
| Cleveland | GBDT-Only Ensemble | 81.6 | 0.793 | 0.639 | 0.902 | 0.128 |
| Cleveland | Minus Transformer | 81.6 | 0.793 | 0.639 | 0.902 | 0.128 |
| Cleveland | LR-Only | 81.7 | 0.795 | 0.641 | 0.902 | 0.128 |
| Cleveland | Transformer-Only | 76.8 | 0.737 | 0.559 | 0.892 | 0.167 |
| CKD | Full DS-HTE | 98.7 | 0.990 | 0.972 | 0.996 | 0.010 |
| CKD | Minus Feature Optimization | 98.9 | 0.991 | 0.977 | 0.996 | 0.008 |
| CKD | Minus Imbalance Handling | 98.7 | 0.990 | 0.972 | 0.996 | 0.010 |
| CKD | Minus Calibration | 98.6 | 0.989 | 0.971 | 0.999 | 0.012 |
| CKD | Minus Threshold Optimization | 98.7 | 0.990 | 0.972 | 0.996 | 0.010 |
| CKD | GBDT-Only Ensemble | 98.7 | 0.990 | 0.973 | 0.996 | 0.011 |
| CKD | Minus Transformer | 98.7 | 0.990 | 0.973 | 0.996 | 0.011 |
| CKD | LR-Only | 98.8 | 0.990 | 0.974 | 0.996 | 0.010 |
| CKD | Transformer-Only | 98.1 | 0.985 | 0.960 | 0.987 | 0.017 |
| ILPD | Full DS-HTE | 62.4 | 0.670 | 0.337 | 0.736 | 0.178 |
| ILPD | Minus Feature Optimization | 62.3 | 0.669 | 0.336 | 0.734 | 0.178 |
| ILPD | Minus Imbalance Handling | 62.4 | 0.670 | 0.337 | 0.736 | 0.178 |
| ILPD | Minus Calibration | 62.7 | 0.674 | 0.342 | 0.737 | 0.176 |
| ILPD | Minus Threshold Optimization | 71.1 | 0.818 | 0.166 | 0.736 | 0.178 |
| ILPD | GBDT-Only Ensemble | 62.3 | 0.669 | 0.338 | 0.735 | 0.178 |
| ILPD | Minus Transformer | 62.3 | 0.669 | 0.338 | 0.735 | 0.178 |
| ILPD | LR-Only | 61.9 | 0.664 | 0.331 | 0.741 | 0.177 |
| ILPD | Transformer-Only | 67.1 | 0.756 | 0.198 | 0.717 | 0.195 |
| WBCD | Full DS-HTE | 96.2 | 0.950 | 0.921 | 0.991 | 0.026 |
| WBCD | Minus Feature Optimization | 97.0 | 0.959 | 0.936 | 0.993 | 0.022 |
| WBCD | Minus Imbalance Handling | 96.2 | 0.950 | 0.921 | 0.991 | 0.026 |
| WBCD | Minus Calibration | 96.4 | 0.952 | 0.925 | 0.995 | 0.024 |
| WBCD | Minus Threshold Optimization | 96.5 | 0.952 | 0.926 | 0.991 | 0.026 |
| WBCD | GBDT-Only Ensemble | 96.1 | 0.947 | 0.917 | 0.991 | 0.026 |
| WBCD | Minus Transformer | 96.1 | 0.947 | 0.917 | 0.991 | 0.026 |
| WBCD | LR-Only | 96.3 | 0.950 | 0.922 | 0.991 | 0.026 |
| WBCD | Transformer-Only | 94.8 | 0.928 | 0.892 | 0.974 | 0.042 |

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
