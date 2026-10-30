# Clinical Dataset Datasheet: Indian Liver Patient Dataset (ILPD)

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Dataset Overview
- **Dataset Name**: Indian Liver Patient Dataset (ILPD)
- **Primary Clinical Target**: Liver disease diagnosis (binary: 1 = liver disease patient, 2 = non-patient; recoded to 0 = healthy, 1 = disease).
- **Source**: Collected from North East of Andhra Pradesh, India, hosted at UCI ML Repository.
- **Sample Size (N)**: 583 records (441 male, 142 female).
- **Class Balance**: 416 Disease patients (71.4%), 167 Non-disease controls (28.6%). Majority baseline: **71.4%**.

---

## 2. Clinical Attributes & Ratios
1. `Age`: Patient age in years.
2. `Gender`: Biological sex (Male / Female).
3. `Total_Bilirubin`: Total serum bilirubin (mg/dL).
4. `Direct_Bilirubin`: Conjugated direct bilirubin (mg/dL).
5. `Alkaline_Phosphotase`: Serum alkaline phosphatase (IU/L).
6. `Alamine_Aminotransferase`: Serum alanine aminotransferase (ALT) (IU/L).
7. `Aspartate_Aminotransferase`: Serum aspartate aminotransferase (AST) (IU/L).
8. `Total_Protiens`: Total serum protein (g/dL).
9. `Albumin`: Serum albumin (g/dL).
10. `Albumin_and_Globulin_Ratio`: Albumin-globulin ratio (A/G).

### Duplicate Records & Grouped Splitting
The ILPD dataset contains 13 identical duplicate records across 570 unique patient clusters. 
- **Controlled Leakage Experiment**: Splitting without duplicate grouping inflates accuracy by $+0.60\%$ and ROC-AUC by $+0.87\%$.
- **Grouping Enforcement**: Grouped cross-validation (`StratifiedGroupKFold`) is strictly enforced to prevent identical biomarker combinations from leaking across outer evaluation folds.

---

## 3. Discrepancy & Operating Point Reality
- When Youden's J optimizes sensitivity (54.6% sensitivity, 66.8% balanced accuracy), raw classification accuracy drops to **62.4%**.
- When thresholding is optimized for accuracy, performance matches the majority rate at **71.5%**.
- Published claims of 79.3% accuracy reflect either unrecorded duplicate leakage or severe test-set selection bias.
