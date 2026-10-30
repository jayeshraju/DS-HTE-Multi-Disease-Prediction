# Clinical Dataset Datasheet: Chronic Kidney Disease (CKD)

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Dataset Overview
- **Dataset Name**: Chronic Kidney Disease Dataset (Apollo Hospitals, Tamil Nadu, India)
- **Primary Clinical Target**: Presence of Chronic Kidney Disease (binary: 0 = notckd, 1 = ckd).
- **Source**: Dr. P. Soundarapandian, M.D., D.N.B. (Apollo Hospitals), hosted on UCI ML Repository.
- **Sample Size (N)**: 400 patient records collected over approximately two months.
- **Class Balance**: 250 CKD (62.5%), 150 NotCKD (37.5%). Majority baseline: **62.5%**.

---

## 2. Clinical Attributes & Linearity
Contains 24 physical and laboratory attributes (11 numeric, 13 nominal):
- Blood chemistry: `bgr` (glucose), `bu` (blood urea), `sc` (serum creatinine), `sod` (sodium), `pot` (potassium), `hemo` (hemoglobin).
- Urine analysis: `sg` (specific gravity), `al` (albumin), `su` (sugar), `rbc` (red blood cells), `pc` (pus cell).
- Systemic markers: `bp` (blood pressure), `htn` (hypertension), `dm` (diabetes mellitus), `cad` (coronary artery disease), `pe` (pedal edema), `ane` (anemia).

### High Linear Separability
Due to strong diagnostic determinism in serum creatinine ($sc$) and hemoglobin ($hemo$), standard regularized linear models and gradient trees easily achieve $>98\%$ accuracy. High metrics reflect strong disease marker divergence rather than model overfitting.
