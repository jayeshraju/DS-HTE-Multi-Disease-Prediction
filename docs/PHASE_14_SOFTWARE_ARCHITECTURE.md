# Phase 14: Modular Software Architecture — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction & Architecture Principles

Moving from experimental research scripts to a publication-grade, production-ready biomedical AI framework requires a modular, maintainable software architecture. Clinical AI systems must adhere to strict software engineering standards to guarantee computational reproducibility, pipeline reusability across new disease cohorts, seamless API integration, and clean separation of concerns.

This phase formulates the **Modular Enterprise Software Architecture** for the multi-disease prediction framework, following **Clean Architecture** and **SOLID design principles**.

---

## 2. Enterprise Software Project Directory Tree

```
MULTI_DIS_PREDICT/
├── config/                                     # Global Configuration & Disease Metadata
│   ├── __init__.py
│   ├── config.py                               # Random seeds, file paths, target labels
│   └── disease_metadata.json                   # Physiological reference ranges & zero-flags
│
├── data/                                       # Data Storage & Pipeline Management
│   ├── raw/                                    # Raw benchmark datasets (.csv)
│   ├── processed/                              # Imputed, scaled Parquet files (.parquet)
│   └── artifact_store/                         # Saved preprocessor pickle states (.joblib)
│
├── docs/                                       # Publication & Research Foundation Docs
│   ├── PHASE_1_RESEARCH_FOUNDATION.md
│   ├── PHASE_2_DATASET_DESIGN.md
│   ├── PHASE_3_EXPLORATORY_DATA_ANALYSIS.md
│   ├── PHASE_4_DATA_PREPROCESSING.md
│   ├── PHASE_5_FEATURE_ENGINEERING_AND_AFO.md
│   ├── PHASE_6_DATA_BALANCING.md
│   ├── PHASE_7_BASELINE_MODELS.md
│   ├── PHASE_8_DEEP_LEARNING_ARCHITECTURE.md
│   ├── PHASE_9_PROPOSED_HYBRID_FRAMEWORK.md
│   ├── PHASE_10_HYPERPARAMETER_OPTIMIZATION.md
│   ├── PHASE_11_EXPLAINABLE_AI.md
│   ├── PHASE_12_MODEL_EVALUATION.md
│   ├── PHASE_13_AI_CLINICAL_RISK_REPORT.md
│   └── PHASE_14_SOFTWARE_ARCHITECTURE.md       <-- [DOCUMENTED]
│
├── src/                                        # Primary Core Source Code Engine
│   ├── __init__.py
│   ├── config.py                               # Master configuration class
│   │
│   ├── data/                                   # Data Preprocessing & Balancing Engine
│   │   ├── __init__.py
│   │   ├── preprocessing.py                    # MedicalPreprocessor (KNN + IsolationForest)
│   │   └── balancing.py                        # AdaptiveDataBalancer (SMOTE + Tomek)
│   │
│   ├── features/                               # Feature Engineering & Optimization
│   │   ├── __init__.py
│   │   └── afo.py                              # Adaptive Feature Optimization (Borda Consensus)
│   │
│   ├── models/                                 # Estimator Factory & Model Registry
│   │   ├── __init__.py
│   │   ├── baselines.py                        # 9 Baseline Machine Learning Estimators
│   │   ├── deep_learning.py                    # PyTorch FT-Transformer & Feature Tokenizer
│   │   └── hybrid.py                           # HDA-GB Classifier (FT-Transformer + XGBoost)
│   │
│   ├── utils/                                  # Utilities, Metrics & Reporting Suite
│   │   ├── __init__.py
│   │   ├── optimization.py                     # Optuna TPE Hyperparameter Tuning Engine
│   │   ├── explainability.py                   # ClinicalExplainer (SHAP & LIME Wrappers)
│   │   ├── metrics.py                          # 10 Diagnostic Metrics + McNemar Test
│   │   └── report_generator.py                 # Clinical Risk Report Synthesis Engine
│   │
│   └── api/                                    # Production REST API Engine
│       ├── __init__.py
│       └── app.py                              # FastAPI REST Server Endpoints
│
├── tests/                                      # Automated Unit & Integration Tests
│   ├── test_preprocessing.py                   # Preprocessing & leakage tests
│   ├── test_models.py                          # Model fitting & prediction tests
│   └── test_api.py                             # FastAPI endpoint integration tests
│
├── .gitignore                                  # Git exclusion definitions
├── requirements.txt                            # Locked Python dependency manifest
└── main.py                                     # Command-Line Pipeline Execution Entrypoint
```

---

## 3. High-Level System Architecture & Component Interaction

```
+-----------------------------------------------------------------------------------+
|                           CLI / API / UI Entrypoint                               |
|              (main.py / FastAPI REST Endpoints / Streamlit UI)                    |
+-----------------------------------------------------------------------------------+
                                         |
                                         v
+-----------------------------------------------------------------------------------+
|                             src/config.py (Master Config)                         |
|                 Enforces Random Seed = 42, Paths, Dataset Metadata                |
+-----------------------------------------------------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     |                                                                       |
     v                                                                       v
+---------------------------------------------------+   +---------------------------------------------------+
|               src/data Engine                     |   |              src/features Engine                  |
| - MedicalPreprocessor (Imputation & Scaling)      |   | - Adaptive Feature Optimization (AFO)             |
| - AdaptiveDataBalancer (SMOTE + Tomek)            |   | - Borda Consensus Rank Matrix Calculation         |
+---------------------------------------------------+   +---------------------------------------------------+
                                         |                                   |
                                         +-----------------+-----------------+
                                                           |
                                                           v
                                        +------------------------------------+
                                        |          src/models Engine         |
                                        | - Baseline Model Suite             |
                                        | - PyTorch FT-Transformer Encoder   |
                                        | - Hybrid HDA-GB Classifier         |
                                        +------------------------------------+
                                                           |
                                                           v
                                        +------------------------------------+
                                        |          src/utils Suite           |
                                        | - Optuna Hyperparameter Tuning     |
                                        | - ClinicalExplainer (SHAP/LIME)    |
                                        | - 10 Diagnostic Metrics Engine     |
                                        | - Clinical PDF Report Generator    |
                                        +------------------------------------+
```

---

## 4. Software Design & Clean Architecture Principles

TABLE I: IEEE-FORMATTED SOFTWARE ARCHITECTURE SPECIFICATION

| Clean Architecture Layer | Component Path | Responsibility / Purpose | Design Pattern / Principle |
| :--- | :--- | :--- | :--- |
| **Domain Entities** | `src/config.py` | Defines immutable data structures, disease target labels, and seed constants. | Single Source of Truth (SSOT) |
| **Data Adapter Layer** | `src/data/preprocessing.py` | Transforms raw data into clean model-ready arrays without leakage. | Scikit-Learn Transformer API |
| **Feature Optimization**| `src/features/afo.py` | Computes Borda rank matrices across MI, LASSO, GBDT, and RFE. | Strategy Pattern |
| **Model Engine** | `src/models/` | Encapsulates ML baselines, PyTorch Transformer, and HDA-GB hybrid logic. | Factory & Estimator Pattern |
| **Utilities & XAI** | `src/utils/` | Computes diagnostic metrics, SHAP values, and builds clinical PDF reports. | Dependency Injection |
| **Interface Adapters** | `src/api/app.py` | Exposes REST HTTP endpoints for real-time patient risk predictions. | REST API / OpenAPI Specification |

---

## 5. Expected Outcomes & Phase Summary

* **Expected Software Engineering Standard:** A modular, reproducible codebase adhering to PEP 8 standards, achieving 100% test coverage across core data transformation and inference components.
* **Phase Summary:** Phase 14 establishes a production-grade software architecture structuring data, features, models, utilities, API, and UI components into a clean, maintainable enterprise repository.
