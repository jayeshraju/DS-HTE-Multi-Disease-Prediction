# Medical Image Benchmark Results: PNEUMONIAMNIST

> **Notice**: Research prototype, not a diagnostic device.

### Dataset Split & Verification
- **Official Split Sizes**: Train=4708, Validation=524, Test=624
- **Test Ground Truth**: Negative=234, Positive=390
- **Test Majority Baseline**: **62.5%** (calculated directly from test labels)
- **Operating Point Threshold**: 0.74 (selected on validation split only)

### Performance Comparison

| Metric | Achieved (Frozen ResNet-18) | Published Baseline (Full Fine-Tuning) |
| :--- | :--- | :--- |
| Accuracy | 82.21% | 85.0% - 88.0% |
| ROC-AUC | 0.9100 | 0.94 - 0.96 |
| Brier Score | 0.1605 | N/A |
| ECE | 0.1587 | N/A |
| Majority Baseline | 62.5% | N/A |

### Confusion Matrix (Test Set)
```
Actual\Predicted    Negative    Positive
Negative            141         93
Positive            18          372
```

### Methodological Discussion
Frozen-backbone feature extraction with post-hoc calibration provides rapid, CPU-feasible evaluation. Hyperparameters and decision threshold were selected strictly using inner validation splits, with the test set remaining sealed until final evaluation.

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
