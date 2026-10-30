# Phase 2: Dataset Design — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Scalable Multi-Disease Early Prediction and Risk Stratification Framework using Explainable Artificial Intelligence and Adaptive Feature Optimization
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Dataset Acquisition Methodology

The data acquisition architecture establishes a standardized ingestion protocol for canonical, open-access diagnostic benchmarks. All datasets are sourced from established medical repositories (UCI Machine Learning Repository and Kaggle Medical Databases) to ensure open reproducibility, ethical compliance, and clinical benchmarking validity. The acquisition pipeline enforces rigorous schema validation, feature domain casting, and metadata tracking prior to downstream preprocessing.

---

## 2. Comprehensive Dataset Profiles

### 2.1 PIMA Indian Diabetes Dataset
* **Source:** National Institute of Diabetes and Digestive and Kidney Diseases (UCI ML Repository / Kaggle).
* **Sample Size (Records):** 768 patient instances.
* **Dimensionality (Features):** 8 continuous/discrete clinical attributes + 1 binary target.
* **Target Variable:** `Outcome` (0 = Non-diabetic, 1 = Diabetic).
* **Clinical Meaning:** Assesses 5-year Type 2 Diabetes Mellitus onset risk among female patients of Pima Indian heritage aged $\ge 21$. Features capture key metabolic markers including fasting plasma glucose, blood pressure, triceps skinfold thickness, 2-hour serum insulin, Body Mass Index (BMI), diabetes pedigree function (hereditary risk), and age.
* **Advantages:** Benchmark standard for tabular metabolic machine learning; highly studied feature interactions; clear biological correlation between fasting glucose/BMI and target label.
* **Limitations:** Moderate size ($N=768$); restricted to female demographic cohort; presence of biologically impossible zero-values (e.g., Blood Pressure = 0, Glucose = 0) representing implicit missing data.
* **Selection Justification:** Serves as the global canonical benchmark for tabular diabetes risk modeling, enabling direct performance benchmarking against state-of-the-art literature.

### 2.2 Cleveland Heart Disease Dataset
* **Source:** UCI Machine Learning Repository (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation - Robert Detrano, M.D., Ph.D.).
* **Sample Size (Records):** 303 patient instances.
* **Dimensionality (Features):** 13 clinical attributes (demographic, resting physiological, exercise stress test, cardiac imaging proxies) + 1 target.
* **Target Variable:** `num` / `target` (Original 0–4 severity scale mapped to Binary: 0 = Absence of Angiographic Disease, 1 = Presence of $\ge 50\%$ Diameter Narrowing).
* **Clinical Meaning:** Evaluates coronary artery disease presence based on key cardiovascular biomarkers including resting blood pressure (`trestbps`), serum cholesterol (`chol`), fasting blood sugar (`fbs`), resting ECG (`restecg`), maximum heart rate achieved (`thalach`), exercise-induced angina (`exang`), ST depression (`oldpeak`), slope of peak exercise ST segment (`slope`), number of major vessels colored by fluoroscopy (`ca`), and thalassemia type (`thal`).
* **Advantages:** Rich domain complexity combining resting laboratory metrics with exercise stress parameters; high clinical relevance for ischemic heart disease.
* **Limitations:** Relatively small cohort ($N=303$); minor missing value count in `ca` and `thal` attributes; potential selection bias toward symptomatic patients undergoing angiography.
* **Selection Justification:** Gold standard cardiovascular tabular benchmark widely referenced in clinical decision support literature.

### 2.3 Chronic Kidney Disease (CKD) Dataset
* **Source:** UCI Machine Learning Repository (Apollo Hospitals, Tamil Nadu, India - Dr. P. Soundarapandian).
* **Sample Size (Records):** 400 patient instances.
* **Dimensionality (Features):** 24 clinical attributes (11 numerical, 13 categorical) + 1 binary target.
* **Target Variable:** `classification` (0 = Non-CKD, 1 = CKD).
* **Clinical Meaning:** Comprehensive renal pathology panel evaluating kidney filtration capacity and metabolic imbalances. Attributes include age, blood pressure, specific gravity, albumin, sugar, red blood cells, pus cells, pus cell clumps, bacteria, blood glucose random, blood urea, serum creatinine, sodium, potassium, hemoglobin, packed cell volume, white blood cell count, red blood cell count, hypertension, diabetes mellitus, coronary artery disease, appetite, pedal edema, and anemia.
* **Advantages:** Extensive diagnostic breadth incorporating urinalysis, blood chemistry, hematology, and clinical comorbidities.
* **Limitations:** Significant missing data proportion (~19% total cell missingness); mixed continuous and categorical data types requiring specialized imputation and encoding.
* **Selection Justification:** Highly representative of complex clinical EHR data with multi-system renal manifestations.

### 2.4 Indian Liver Patient Dataset (ILPD)
* **Source:** UCI Machine Learning Repository (Northeast Andhra Pradesh cohort - Bendi Venkata Ramana).
* **Sample Size (Records):** 583 patient instances (416 liver patients, 167 non-patients).
* **Dimensionality (Features):** 10 clinical attributes (9 numerical, 1 binary categorical) + 1 binary target.
* **Target Variable:** `Dataset` / `Is_Patient` (1 = Liver Patient, 2 / 0 = Non-Liver Patient).
* **Clinical Meaning:** Screens for hepatic cellular damage, biliary obstruction, and metabolic dysfunction using standard Liver Function Tests (LFT). Attributes include Age, Gender, Total Bilirubin, Direct Bilirubin, Alkaline Phosphatase (ALP), Alanine Aminotransferase (ALT/SGPT), Aspartate Aminotransferase (AST/SGOT), Total Proteins, Albumin, and Albumin/Globulin (A/G) Ratio.
* **Advantages:** Direct alignment with standard clinical LFT blood panels routinely ordered during primary care assessments.
* **Limitations:** Moderate class imbalance ratio (~71% positive, 29% negative); high multicollinearity between total/direct bilirubin and ALT/AST enzyme metrics.
* **Selection Justification:** Critical benchmark for evaluates hepatic diagnostic risk and testing feature selection on highly collinear biomarkers.

### 2.5 Wisconsin Breast Cancer Dataset (Diagnostic - WBCD)
* **Source:** UCI Machine Learning Repository (University of Wisconsin - Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian).
* **Sample Size (Records):** 569 patient instances.
* **Dimensionality (Features):** 30 continuous features computed from digitized Fine Needle Aspirate (FNA) cytological images + 1 binary target.
* **Target Variable:** `diagnosis` (M = Malignant [1], B = Benign [0]).
* **Clinical Meaning:** Quantitative morphological measurement of cell nuclei present in breast tissue aspirates. Includes 10 real-valued features computed for mean, standard error (SE), and worst/largest values: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, and Fractal Dimension.
* **Advantages:** High signal-to-noise ratio; zero missing values; complete quantitative continuous feature distribution; gold-standard oncology tabular benchmark.
* **Limitations:** High spatial autocorrelation and multicollinearity across mean, SE, and worst dimensions (e.g., radius vs. perimeter vs. area).
* **Selection Justification:** Benchmark for evaluating high-dimensional continuous feature reduction and non-linear diagnostic classification performance.

---

## 3. Dataset Comparison Analysis

TABLE I: IEEE-FORMATTED COMPREHENSIVE DATASET COMPARISON

| Feature Metric | PIMA Diabetes | Cleveland Heart | Chronic Kidney Disease | Indian Liver Patient | Wisconsin Breast Cancer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Medical Domain** | Metabolic / Endocrinology | Cardiovascular | Renal / Nephrology | Hepatic / Gastroenterology | Oncology / Pathology |
| **Record Count ($N$)** | 768 | 303 | 400 | 583 | 569 |
| **Feature Count ($D$)** | 8 | 13 | 24 | 10 | 30 |
| **Data Types** | Continuous / Discrete | Mixed (Cont, Cat, Ord) | Mixed (Cont, Nominal) | Mixed (Cont, Binary) | Continuous (Real-valued) |
| **Missing Values** | Implicit (Zeros in bio-markers) | Minimal (6 cells in ca, thal) | High (~19% total missingness) | Minimal (4 cells in A/G ratio) | Zero (0 missing values) |
| **Class Ratio (Neg : Pos)** | 500 : 268 (65.1% : 34.9%) | 164 : 139 (54.1% : 45.9%) | 150 : 250 (37.5% : 62.5%) | 167 : 416 (28.6% : 71.4%) | 357 : 212 (62.7% : 37.3%) |
| **Primary Biomarkers** | Glucose, Insulin, BMI | ST-Depression, Max HR, Angina | Serum Creatinine, GFR, Albumin | ALT, AST, Bilirubin, ALP | Nucleus Radius, Concavity, Texture |
| **Diagnostic Complexity** | Moderate | High | High (Sparse/Heterogeneous) | Moderate (High Collinearity) | High-Dimensional Continuous |

---

## 4. Feature Taxonomy & Categorization

The combined feature ecosystem across all five target datasets encompasses 75 total input variables, systematically categorized into four distinct clinical domain tiers:

```
                                  +-------------------------------------------------------+
                                  |       Unified Clinical Feature Taxonomy (75 Vars)     |
                                  +-------------------------------------------------------+
                                                              |
          +---------------------------+-----------------------+-----------------------+---------------------------+
          |                           |                                               |                           |
          v                           v                                               v                           v
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
|  Demographic &    |       |   Physiological   |                           |    Biochemical    |       | Cytological &     |
|   Anthropometric  |       |   & Vital Signs   |                           |   Biomarker Array |       | Morphological     |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
| - Age             |       | - Resting BP      |                           | - Fasting Glucose |       | - Cell Radius     |
| - Sex / Gender    |       | - Max Heart Rate  |                           | - Serum Insulin   |       | - Cell Texture    |
| - BMI             |       | - ST-Depression   |                           | - Serum Creatinine|       | - Perimeter/Area  |
| - Skin Thickness  |       | - Exercise Angina |                           | - Total Bilirubin |       | - Smoothness      |
| - Diabetes Ped.   |       | - Pedal Edema     |                           | - ALT / AST       |       | - Concavity       |
+-------------------+       +-------------------+                           +-------------------+       +-------------------+
```

---

## 5. Comprehensive Clinical Data Dictionary

TABLE II: IEEE-FORMATTED UNIFIED CLINICAL DATA DICTIONARY

| Dataset | Variable Identifier | Data Type | Measurement Unit / Scale | Clinical Description & Normal Range | Target Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PIMA** | `Pregnancies` | Discrete Int | Count ($0 - 17$) | Number of times pregnant | Predictor |
| **PIMA** | `Glucose` | Continuous | $\text{mg/dL}$ | 2-hour plasma glucose concentration ($70-99\text{ mg/dL}$) | Predictor |
| **PIMA** | `BloodPressure` | Continuous | $\text{mmHg}$ | Diastolic blood pressure ($60-80\text{ mmHg}$) | Predictor |
| **PIMA** | `SkinThickness` | Continuous | $\text{mm}$ | Triceps skin fold thickness | Predictor |
| **PIMA** | `Insulin` | Continuous | $\mu\text{IU/mL}$ | 2-Hour serum insulin ($16-166\mu\text{IU/mL}$) | Predictor |
| **PIMA** | `BMI` | Continuous | $\text{kg/m}^2$ | Body Mass Index ($18.5-24.9\text{ kg/m}^2$) | Predictor |
| **PIMA** | `DiabetesPedigree` | Continuous | Score ($0.08 - 2.42$) | Genetic hereditary risk score | Predictor |
| **PIMA** | `Age` | Discrete Int | Years | Patient age at exam | Predictor |
| **Cleveland** | `age` | Discrete Int | Years | Patient age | Predictor |
| **Cleveland** | `sex` | Binary Cat | $1 = \text{Male}, 0 = \text{Female}$ | Biological sex | Predictor |
| **Cleveland** | `cp` | Nominal Cat | $1 - 4$ | Chest pain type (Typical/Atypical/Non-anginal/Asymptomatic) | Predictor |
| **Cleveland** | `trestbps` | Continuous | $\text{mmHg}$ | Resting blood pressure on admission | Predictor |
| **Cleveland** | `chol` | Continuous | $\text{mg/dL}$ | Serum cholesterol ($\text{Desirable} < 200\text{ mg/dL}$) | Predictor |
| **Cleveland** | `fbs` | Binary Cat | $1 = >120\text{ mg/dL}, 0 = \le 120$ | Fasting blood sugar indicator | Predictor |
| **Cleveland** | `restecg` | Nominal Cat | $0 - 2$ | Resting ECG results (Normal/ST-T wave abnormality/LV hypertrophy) | Predictor |
| **Cleveland** | `thalach` | Continuous | $\text{bpm}$ | Maximum heart rate achieved during stress test | Predictor |
| **Cleveland** | `exang` | Binary Cat | $1 = \text{Yes}, 0 = \text{No}$ | Exercise-induced angina | Predictor |
| **Cleveland** | `oldpeak` | Continuous | $\text{mm}$ | ST depression induced by exercise relative to rest | Predictor |
| **Cleveland** | `slope` | Ordinal Cat | $1 - 3$ | Slope of peak exercise ST segment (Upsloping/Flat/Downsloping) | Predictor |
| **Cleveland** | `ca` | Discrete Int | $0 - 3$ | Number of major vessels colored by fluoroscopy | Predictor |
| **Cleveland** | `thal` | Nominal Cat | $3 = \text{Norm}, 6 = \text{Fixed}, 7 = \text{Revers}$| Thalassemia scan outcome | Predictor |
| **CKD** | `age` | Continuous | Years | Patient age | Predictor |
| **CKD** | `bp` | Continuous | $\text{mmHg}$ | Blood pressure | Predictor |
| **CKD** | `sg` | Ordinal Cat | $1.005 - 1.025$ | Urine specific gravity | Predictor |
| **CKD** | `al` | Ordinal Cat | $0 - 5$ | Urine albumin level | Predictor |
| **CKD** | `su` | Ordinal Cat | $0 - 5$ | Urine sugar level | Predictor |
| **CKD** | `rbc` | Binary Cat | $\text{normal} / \text{abnormal}$ | Red blood cells in urine | Predictor |
| **CKD** | `pc` | Binary Cat | $\text{normal} / \text{abnormal}$ | Pus cells in urine | Predictor |
| **CKD** | `pcc` | Binary Cat | $\text{present} / \text{notpresent}$ | Pus cell clumps | Predictor |
| **CKD** | `ba` | Binary Cat | $\text{present} / \text{notpresent}$ | Bacteria in urine | Predictor |
| **CKD** | `bgr` | Continuous | $\text{mg/dL}$ | Blood glucose random | Predictor |
| **CKD** | `bu` | Continuous | $\text{mg/dL}$ | Blood urea | Predictor |
| **CKD** | `sc` | Continuous | $\text{mg/dL}$ | Serum creatinine ($0.6-1.2\text{ mg/dL}$) | Predictor |
| **CKD** | `sod` | Continuous | $\text{mEq/L}$ | Blood sodium ($135-145\text{ mEq/L}$) | Predictor |
| **CKD** | `pot` | Continuous | $\text{mEq/L}$ | Blood potassium ($3.5-5.0\text{ mEq/L}$) | Predictor |
| **CKD** | `hemo` | Continuous | $\text{gms}$ | Hemoglobin level | Predictor |
| **CKD** | `pcv` | Continuous | $\%$ | Packed cell volume | Predictor |
| **CKD** | `wc` | Continuous | $\text{cells/cumm}$ | White blood cell count | Predictor |
| **CKD** | `rc` | Continuous | $\text{millions/cmm}$| Red blood cell count | Predictor |
| **CKD** | `htn` | Binary Cat | $\text{yes} / \text{no}$ | History of hypertension | Predictor |
| **CKD** | `dm` | Binary Cat | $\text{yes} / \text{no}$ | History of diabetes mellitus | Predictor |
| **CKD** | `cad` | Binary Cat | $\text{yes} / \text{no}$ | History of coronary artery disease | Predictor |
| **CKD** | `appet` | Binary Cat | $\text{good} / \text{poor}$ | Clinical appetite assessment | Predictor |
| **CKD** | `pe` | Binary Cat | $\text{yes} / \text{no}$ | Presence of pedal edema | Predictor |
| **CKD** | `ane` | Binary Cat | $\text{yes} / \text{no}$ | Presence of anemia | Predictor |
| **ILPD** | `Age` | Discrete Int | Years | Patient age | Predictor |
| **ILPD** | `Gender` | Binary Cat | $\text{Male} / \text{Female}$ | Biological gender | Predictor |
| **ILPD** | `Total_Bilirubin`| Continuous | $\text{mg/dL}$ | Serum total bilirubin ($0.1-1.2\text{ mg/dL}$) | Predictor |
| **ILPD** | `Direct_Bilirubin`| Continuous | $\text{mg/dL}$ | Serum direct bilirubin ($0.0-0.3\text{ mg/dL}$) | Predictor |
| **ILPD** | `ALP` | Continuous | $\text{IU/L}$ | Alkaline Phosphatase ($44-147\text{ IU/L}$) | Predictor |
| **ILPD** | `ALT` | Continuous | $\text{IU/L}$ | Alanine Aminotransferase / SGPT ($7-56\text{ IU/L}$) | Predictor |
| **ILPD** | `AST` | Continuous | $\text{IU/L}$ | Aspartate Aminotransferase / SGOT ($10-40\text{ IU/L}$) | Predictor |
| **ILPD** | `Total_Proteins` | Continuous | $\text{g/dL}$ | Total serum protein ($6.0-8.3\text{ g/dL}$) | Predictor |
| **ILPD** | `Albumin` | Continuous | $\text{g/dL}$ | Serum albumin ($3.5-5.0\text{ g/dL}$) | Predictor |
| **ILPD** | `A/G_Ratio` | Continuous | Ratio | Albumin / Globulin Ratio ($1.1-2.5$) | Predictor |
| **WBCD** | `radius_mean` | Continuous | $\mu\text{m}$ | Mean distance from center to perimeter points | Predictor |
| **WBCD** | `texture_mean` | Continuous | Gray-scale | Standard deviation of gray-scale values | Predictor |
| **WBCD** | `perimeter_mean`| Continuous | $\mu\text{m}$ | Mean size of core tumor perimeter | Predictor |
| **WBCD** | `area_mean` | Continuous | $\mu\text{m}^2$ | Mean tumor area | Predictor |
| **WBCD** | `smoothness_mean`| Continuous | Ratio | Mean local variation in radius lengths | Predictor |
| **WBCD** | `concavity_mean` | Continuous | Score | Mean severity of concave portions of contour | Predictor |
| **WBCD** | `... [x30 Total]`| Continuous | Various | [Includes standard error and worst values for all 10 features] | Predictor |

---

## 6. Label Mapping & Target Definitions

TABLE III: IEEE-FORMATTED TARGET LABEL MAPPING ARCHITECTURE

| Dataset | Raw Source Label Identifiers | Mapping Logic / Transformation Rule | Normalized Target Label ($y$) | Binary Target Meaning |
| :--- | :--- | :--- | :--- | :--- |
| **PIMA Diabetes** | `Outcome` $\in \{0, 1\}$ | Identity Mapping | $y \in \{0, 1\}$ | $0 = \text{Negative}, 1 = \text{Diabetic}$ |
| **Cleveland Heart** | `num` $\in \{0, 1, 2, 3, 4\}$ | $y = 0 \text{ if } \text{num} = 0 \text{ else } 1$ | $y \in \{0, 1\}$ | $0 = \text{Healthy}, 1 = \text{Cardiovascular Disease}$ |
| **CKD** | `classification` $\in \{\text{'ckd'}, \text{'notckd'}\}$ | $y = 1 \text{ if } \text{'ckd'} \text{ else } 0$ | $y \in \{0, 1\}$ | $0 = \text{Non-CKD}, 1 = \text{Chronic Kidney Disease}$ |
| **ILPD** | `Dataset` $\in \{1, 2\}$ | $y = 1 \text{ if } \text{Dataset} = 1 \text{ else } 0$ | $y \in \{0, 1\}$ | $0 = \text{Non-Liver Patient}, 1 = \text{Liver Patient}$ |
| **WBCD** | `diagnosis` $\in \{\text{'M'}, \text{'B'}\}$ | $y = 1 \text{ if } \text{'M'} \text{ else } 0$ | $y \in \{0, 1\}$ | $0 = \text{Benign}, 1 = \text{Malignant Tumor}$ |

---

## 7. Storage Strategy & Dataset Organization

To maintain data integrity, prevent data leakage, and ensure auditability, a structured multi-tier file organization strategy is implemented.

### 7.1 File System Organization Structure
```
MULTI_DIS_PREDICT/
├── data/
│   ├── raw/                           # Immutable raw benchmark CSV files
│   │   ├── pima_diabetes_raw.csv
│   │   ├── cleveland_heart_raw.csv
│   │   ├── ckd_kidney_raw.csv
│   │   ├── ilpd_liver_raw.csv
│   │   └── wbcd_breast_cancer_raw.csv
│   ├── interim/                       # Imputed, cleaned, & encoded data
│   │   ├── pima_interim.parquet
│   │   ├── cleveland_interim.parquet
│   │   ├── ckd_interim.parquet
│   │   ├── ilpd_interim.parquet
│   │   └── wbcd_interim.parquet
│   └── processed/                     # Scaled, AFO-selected, split subsets
│       ├── diabetes/                  # (train/val/test splits + metadata)
│       ├── heart/
│       ├── kidney/
│       ├── liver/
│       └── cancer/
├── docs/                              # Project technical foundation docs
│   ├── PHASE_1_RESEARCH_FOUNDATION.md
│   └── PHASE_2_DATASET_DESIGN.md
```

### 7.2 Storage Format Justification
* **Apache Parquet for Interim/Processed Layers:** Selected over standard CSV due to schema enforcement, columnar binary storage, support for rich metadata, fast IO throughput, and efficient memory footprint during hyperparameter tuning iterations.

---

## 8. Architectural Justification: Separate vs. Integrated Datasets

A critical design choice in multi-disease predictive modeling is deciding between a **Unified Mega-Matrix (Integrated Data Model)** versus a **Multi-Task Modular Architecture (Separate Datasets)**. 

### 8.1 Evaluated Architectural Approaches

1. **Integrated Mega-Matrix Paradigm (Rejected):**
   * *Concept:* Merging all five datasets into a single unified table with 75 columns, padding missing features with NaNs when a patient record originates from a single disease diagnostic trial.
   * *Critical Drawbacks:* Results in extreme structural sparsity (>80% missing data values overall), creates artificial cross-disease feature dependencies, induces severe data leakage across distinct demographic populations, and degrades model performance due to massive uninformative missingness padding.

2. **Multi-Task Modular Architecture (Adopted):**
   * *Concept:* Maintaining independent, high-density feature spaces ($D_k$) optimized specifically for each organ domain ($k \in \{1 \dots 5\}$), governed by a standardized, multi-disease master orchestration interface.
   * *Key Advantages:*
     - **Pathophysiological Integrity:** Preserves the specific biological relationships unique to each disease domain.
     - **Computational Efficiency:** Eliminates massive sparse matrix operations.
     - **Modular Extensibility:** Permits seamless integration of future disease models (e.g., stroke or pulmonary risk) without forcing schema migration across existing models.
     - **Clinical Realism:** Mirrors real-world medical practice where distinct laboratory diagnostic panels are ordered based on targeted clinical presentations.
