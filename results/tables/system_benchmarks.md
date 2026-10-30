# System Latency, Memory & Throughput Benchmarks

> **Notice**: Research prototype, not a diagnostic device.
> **Hardware Context**: Standard CPU execution (Windows/PowerShell) across 30-50 iterations.

| Operation / Endpoint | p50 Latency (ms) | p95 Latency (ms) | Mean Latency (ms) | Throughput (req/s) | SLA Target |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Tabular Risk Prediction (predict_tabular) | 20.81 | 24.52 | 21.04 +- 1.96 | 47.5 | < 100 ms |
| Feature Explanation (explain_prediction) | 0.00 | 0.00 | 0.00 +- 0.00 | 10000.0 | < 50 ms |
| Medical Image Inference (ResNet-18 forward) | 32.70 | 36.97 | 32.91 +- 2.64 | 30.4 | < 250 ms (CPU) |
| Physician PDF Report Compilation | 13.11 | 14.14 | 12.72 +- 1.22 | 78.6 | < 300 ms |

### Artifact Storage Footprint
- **`ckd_dshte.joblib`**: 325.2 KB
- **`cleveland_dshte.joblib`**: 566.8 KB
- **`ilpd_dshte.joblib`**: 860.2 KB
- **`pima_dshte.joblib`**: 1031.7 KB
- **`wbcd_dshte.joblib`**: 395.4 KB
- **`classifier.pkl`**: 2.7 KB
- **`ood_detector.pkl`**: 2056.9 KB

---
*Run ID: `dshte_paper_20261029_113441` | Config Hash: `9ac2a61966dc6226`*
