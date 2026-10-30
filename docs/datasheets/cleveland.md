# Clinical Dataset Datasheet: Cleveland Heart Disease

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Dataset Overview
- **Dataset Name**: Cleveland Heart Disease Database (UCI)
- **Primary Clinical Target**: Presence of coronary heart disease (angiographic stenosis > 50% in any major artery; binary target 0 = absence, 1 = presence).
- **Source**: Cleveland Clinic Foundation, Dr. Robert Detrano, MD, PhD.
- **Sample Size (N)**: 303 adult patients referred for coronary angiography.
- **Class Balance**: 164 Absence (54.1%), 139 Presence (45.9%). Majority class baseline: **54.1%**.

---

## 2. Clinical Attributes & Schema
1. `age`: Age in years.
2. `sex`: Sex (1 = male, 0 = female).
3. `cp`: Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic).
4. `trestbps`: Resting blood pressure on admission to the hospital (mmHg).
5. `chol`: Serum cholesterol in mg/dl.
6. `fbs`: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).
7. `restecg`: Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy).
8. `thalach`: Maximum heart rate achieved during exercise.
9. `exang`: Exercise-induced angina (1 = yes, 0 = no).
10. `oldpeak`: ST depression induced by exercise relative to rest.
11. `slope`: The slope of the peak exercise ST segment.
12. `ca`: Number of major vessels (0-3) colored by flourosopy (4 missing values).
13. `thal`: Thallium scintigraphy (3: normal, 6: fixed defect, 7: reversible defect; 2 missing values).

---

## 3. External Multi-Center Compatibility
The Cleveland dataset represents the primary training reference for multi-center cardiac benchmarking. Standardized external evaluations are performed against matching schemas from Hungarian, Zurich, and Long Beach VA cohorts.
