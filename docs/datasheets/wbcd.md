# Clinical Dataset Datasheet: Wisconsin Diagnostic Breast Cancer (WBCD)

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Dataset Overview
- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Primary Clinical Target**: Tumor diagnosis (binary: M = Malignant, B = Benign; coded 1 = Malignant, 0 = Benign).
- **Source**: Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian (University of Wisconsin), hosted at UCI.
- **Sample Size (N)**: 569 fine-needle aspirate (FNA) digitized images.
- **Class Balance**: 357 Benign (62.7%), 212 Malignant (37.3%). Majority baseline: **62.7%**.

---

## 2. Cell Nuclei Morphological Features
Ten real-valued features computed for each cell nucleus:
1. `radius`: Mean of distances from center to points on the perimeter.
2. `texture`: Standard deviation of gray-scale values.
3. `perimeter`: Nuclear perimeter.
4. `area`: Nuclear cross-sectional area.
5. `smoothness`: Local variation in radius lengths.
6. `compactness`: $\text{perimeter}^2 / \text{area} - 1.0$.
7. `concavity`: Severity of concave portions of the contour.
8. `concave points`: Number of concave portions of the contour.
9. `symmetry`: Nuclear symmetry.
10. `fractal dimension`: "Coastline approximation" $- 1$.

For each FNA image, the mean, standard error, and "worst" (mean of the three largest values) are computed, yielding 30 input features.

---

## 3. Ensemble Collapse to Regularized Logistic Regression
Stacking ensemble weight analysis reveals that on WBCD, regularized Logistic Regression captures **92.2%** of the meta-learner weight. Due to strong linear separability in morphological nuclear dimensions, complex non-linear models and Tabular Transformers add negligible marginal benefit, collapsing the meta-learner to linear representation.
