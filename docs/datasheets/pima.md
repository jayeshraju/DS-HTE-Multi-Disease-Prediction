# Clinical Dataset Datasheet: PIMA Indian Diabetes

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Dataset Overview
- **Dataset Name**: Pima Indians Diabetes Database
- **Primary Clinical Target**: Onset of diabetes mellitus within 5 years (binary: 0 = Negative, 1 = Positive).
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK), hosted at UCI ML Repository.
- **Sample Size (N)**: 768 female patients of Pima Indian heritage aged $\ge 21$ years.
- **Class Balance**: 500 Negative (65.1%), 268 Positive (34.9%). Majority class baseline: **65.1%**.

---

## 2. Attributes & Missingness Characteristics
The dataset contains 8 physiological features:
1. `Pregnancies`: Number of times pregnant (range 0–17).
2. `Glucose`: Plasma glucose concentration at 2 hours in an oral glucose tolerance test (OGTT).
3. `BloodPressure`: Diastolic blood pressure (mm Hg).
4. `SkinThickness`: Triceps skin fold thickness (mm).
5. `Insulin`: 2-Hour serum insulin (mu U/ml).
6. `BMI`: Body mass index ($\text{weight in kg}/(\text{height in m})^2$).
7. `DiabetesPedigreeFunction`: Diabetes pedigree score based on genetic family history.
8. `Age`: Age in years (range 21–81).

### Zero-Coded Missingness Artifacts
In raw data, missing entries for physiological markers were coded as zero:
- Glucose: 5 instances coded 0 (biologically impossible in living humans).
- Diastolic Blood Pressure: 35 instances coded 0.
- Triceps Skinfold: 227 instances coded 0 (29.6% missing).
- Serum Insulin: 374 instances coded 0 (48.7% missing).
- BMI: 11 instances coded 0.

**Pipeline Handling**: Explicit binary missingness indicators created prior to median imputation fitted strictly on inner training partitions.

---

## 3. Evaluation ceiling
- Under 50-fold nested CV without data leakage, realistic accuracy lands between 73.7% (Balanced Youden-J) and 76.8% (Accuracy-Optimal).
- The historical claim of 81.2% in published literature is not reachable without feature leakage or unrecorded test-set selection.
