# Reproducibility Manifest & Execution Instructions

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. System Environment & Library Versions
The pipeline execution and all artifact generations were conducted under the following exact configuration:

| Component | Specified Version / Parameter |
| :--- | :--- |
| **Operating System** | Windows 11 Enterprise (PowerShell 7.5 / Windows PowerShell 5.1) |
| **Python** | 3.13.12 (CPython x64) |
| **PyTorch** | 2.13.0+cpu |
| **TorchVision** | 0.28.0+cpu |
| **Scikit-Learn** | 1.9.0 |
| **XGBoost** | 3.4.0 |
| **LightGBM** | 4.7.0 |
| **CatBoost** | 1.2.10 |
| **SciPy** | 1.18.0 |
| **Pandas** | 3.0.5 |
| **NumPy** | 2.4.6 |
| **Active Run ID** | `dshte_paper_20261029_113441` |
| **Master Config Hash** | `9ac2a61966dc6226` |
| **Evaluation Seeds** | `[142, 242, 342, 442, 542, 642, 742, 842, 942, 1042]` (10 outer repeats) |

---

## 2. Command Pipeline (In Sequence)

To reproduce all artifacts, tables, figures, benchmarks, and UI builds from scratch:

```powershell
# 1. Setup Python virtual environment
make setup

# 2. Gate 0: Verify Phase 1 integrity and invariants
make verify-phase1

# 3. Full 50-Fold Nested Repeated Cross-Validation Run (Paper Profile)
make run PROFILE=paper

# 4. External Multi-Center Heart Disease Validation
make external-validate

# 5. Medical Image Benchmarks (MedMNIST PneumoniaMNIST & BreastMNIST)
make image-bench

# 6. Run Complete Test Suite
make test

# 7. Build and Run Interactive Dashboard
make ui-build
make ui

# 8. Automated UI & Accessibility Verification
make ui-test

# 9. Compile Phase 2 Conference Paper Package
make paper-phase2
```

---

## 3. Expected Runtimes (Standard 8-Core CPU)
- `verify-phase1`: ~10 seconds
- `image-bench`: ~45 seconds
- `external-validate`: ~5 seconds
- `test`: ~15 seconds
- `ui-build`: ~1 second (Vite production bundle)
- `run PROFILE=paper`: ~1 hour 15 minutes (2,750 outer fold runs across 11 architectures)
