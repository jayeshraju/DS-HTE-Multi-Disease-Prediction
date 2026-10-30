# Medical Image Benchmark Results: BREASTMNIST

> **Notice**: Research prototype, not a diagnostic device.

### Dataset Split & Verification
- **Official Split Sizes**: Train=546, Validation=78, Test=156
- **Test Ground Truth**: Negative=42, Positive=114
- **Test Majority Baseline**: **73.08%** (calculated directly from test labels)
- **Operating Point Threshold**: 0.31 (selected on validation split only)

### Performance Comparison

| Metric | Achieved (Frozen ResNet-18) | Published Baseline (Full Fine-Tuning) |
| :--- | :--- | :--- |
| Accuracy | 80.13% | 82.0% - 86.0% |
| ROC-AUC | 0.8187 | 0.88 - 0.91 |
| Brier Score | 0.1449 | N/A |
| ECE | 0.1115 | N/A |
| Majority Baseline | 73.08% | N/A |

### Confusion Matrix (Test Set)
```
Actual\Predicted    Negative    Positive
Negative            21          21
Positive            10          104
```

### Methodological Discussion
Frozen-backbone feature extraction with post-hoc calibration provides rapid, CPU-feasible evaluation. Hyperparameters and decision threshold were selected strictly using inner validation splits, with the test set remaining sealed until final evaluation.

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
