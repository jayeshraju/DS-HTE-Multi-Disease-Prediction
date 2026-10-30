# Phase 6: Data Balancing and Imbalanced Learning — Multi-Disease Early Prediction Framework

**Document Control & Metadata**
* **Project Title:** Explainable Multi-Disease Prediction Framework Using Hybrid Deep Learning, Adaptive Feature Optimization, and Explainable Artificial Intelligence
* **Target Context:** IEEE Transactions / Scopus Indexed Conference & Final-Year Dissertation Foundation
* **Version:** 1.0.0
* **Date:** July 2026

---

## 1. Introduction

Clinical diagnostic datasets derived from routine health screenings and electronic health records (EHR) frequently exhibit severe target class imbalance. In epidemiological data distributions, healthy control instances typically outnumber confirmed pathological instances. When standard supervised machine learning and deep learning algorithms are trained on imbalanced class distributions, the objective function (e.g., empirical risk minimization based on standard cross-entropy or zero-one loss) becomes heavily biased toward predicting the majority class. 

In medical diagnostic systems, this algorithmic bias results in elevated false-negative rates—a critical failure mode wherein patients harboring severe chronic conditions (such as Chronic Kidney Disease, Hepatitis/Liver Cirrhosis, or Type 2 Diabetes) are misclassified as healthy. This phase formulates a mathematically rigorous, clinically safe **Adaptive Data Balancing Framework** engineered to mitigate class imbalance without introducing data leakage, decision boundary distortion, or synthetic pattern artifacts.

---

## 2. Research Motivation & Domain Analysis

### 2.1 Clinical Importance of Imbalance Handling
In clinical diagnostic workflows, the costs of diagnostic errors are highly asymmetric:
* **False Positive ($FP$ - Type I Error):** A healthy individual is incorrectly flagged as high-risk. This leads to diagnostic anxiety and secondary follow-up testing (e.g., confirmatory blood panels or biopsy), but presents zero direct physiological threat to patient survival.
* **False Negative ($FN$ - Type II Error):** A pathological patient is incorrectly classified as healthy. This results in delayed clinical intervention, unmitigated disease progression, irreversible tissue/organ damage, and heightened patient mortality risk.

Standard un-balanced machine learning models maximize overall classification accuracy ($\frac{TP+TN}{TP+TN+FP+FN}$). In a dataset with a $90:10$ healthy-to-diseased ratio, a trivial classifier predicting all instances as healthy achieves $90\%$ accuracy while exhibiting a catastrophic $0\%$ sensitivity ($\text{Recall} = 0$).

### 2.2 Mathematical Impact on Metric Optimization
Class imbalance skews standard performance estimators:
1. **Accuracy Inflation:** Accuracy fails to reflect minority class discriminative performance.
2. **Receiver Operating Characteristic (ROC) Optimism:** The True Positive Rate ($\text{TPR} = \frac{TP}{TP+FN}$) plotted against False Positive Rate ($\text{FPR} = \frac{FP}{FP+TN}$) can mask poor minority precision when $TN$ is extremely large, rendering the area under the ROC curve ($\text{ROC-AUC}$) misleadingly optimistic.
3. **Precision-Recall (PR) Curve Dependency:** The Precision-Recall curve evaluates $\text{Precision} = \frac{TP}{TP+FP}$ against $\text{Recall} = \frac{TP}{TP+FN}$, directly capturing minority class degradation without dilution by large $TN$ counts. Therefore, PR-AUC and Adjusted F1-Score serve as primary loss optimization targets during imbalance resampling.

---

## 3. Technical Discussion: Comparative Analysis of Resampling Paradigms

Resampling strategies are classified into three primary paradigms:
1. **Under-sampling Methods:** Prune majority class instances to equalize class frequencies.
2. **Over-sampling Methods:** Synthesize or replicate minority class instances.
3. **Hybrid Resampling Methods:** Combine synthetic minority over-sampling with data cleaning algorithms to prune ambiguous decision boundaries.

```
                                  +-------------------------------------------------------+
                                  |         Imbalanced Learning Resampling Taxonomy       |
                                  +-------------------------------------------------------+
                                                              |
          +---------------------------------------------------+---------------------------------------------------+
          |                                                   |                                                   |
          v                                                   v                                                   v
+-------------------+                               +-------------------+                               +-------------------+
|   Under-Sampling  |                               |   Over-Sampling   |                               | Hybrid Resampling |
+-------------------+                               +-------------------+                               +-------------------+
| - Random Under-   |                               | - Random Over-    |                               | - SMOTE + Tomek   |
|   sampling (RUS)  |                               |   sampling (ROS)  |                               |   Links           |
| - Tomek Links     |                               | - Standard SMOTE  |                               | - SMOTE + ENN     |
|   Cleaner         |                               | - Borderline-SMOTE|                               |   (Edited NN)     |
| - Edited Nearest  |                               | - SMOTE-NC        |                               |                   |
|   Neighbours (ENN)|                               | - ADASYN          |                               |                   |
+-------------------+                               +-------------------+                               +-------------------+
```

### 3.1 Mathematical Formulations & Working Principles

#### A. Random Over-sampling (ROS)
* **Working Principle:** Randomly duplicates minority class instances with replacement until class parity is reached.
* **Mathematical Intuition:** Increases the spatial weight of existing minority instances in empirical loss functions.
* **Formulation:** For minority set $\mathcal{S}_{\text{min}}$, select instance $\mathbf{x}_i \in \mathcal{S}_{\text{min}}$ uniformly at random and append to $\mathcal{S}_{\text{min}}' = \mathcal{S}_{\text{min}} \cup \{\mathbf{x}_i\}$.
* **Computational Complexity:** $\mathcal{O}(N_{\text{target}} - N_{\text{min}})$.
* **Clinical Suitability:** Low. Replicating identical patient blood profiles creates exact feature duplicates, inducing severe decision tree overfitting.

#### B. Random Under-sampling (RUS)
* **Working Principle:** Randomly discards majority class instances without replacement.
* **Formulation:** Uniformly sample subset $\mathcal{S}_{\text{maj}}' \subset \mathcal{S}_{\text{maj}}$ such that $|\mathcal{S}_{\text{maj}}'| = |\mathcal{S}_{\text{min}}|$.
* **Computational Complexity:** $\mathcal{O}(N_{\text{maj}} - N_{\text{target}})$.
* **Clinical Suitability:** Low to Moderate. Discards valuable clinical data points, eliminating unique demographic variability.

#### C. Standard Synthetic Minority Over-sampling Technique (SMOTE)
* **Working Principle:** Synthesizes new non-identical minority instances along line segments connecting $k$-nearest intra-class neighbors.
* **Mathematical Intuition:** Given a minority instance $\mathbf{x}_i \in \mathcal{S}_{\text{min}}$, find its $k$-nearest neighbors in $\mathcal{S}_{\text{min}}$. Randomly select one neighbor $\mathbf{x}_{\text{zi}}$, and generate synthetic instance $\mathbf{x}_{\text{new}}$:
  $$\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda (\mathbf{x}_{\text{zi}} - \mathbf{x}_i), \quad \lambda \sim U(0, 1)$$
* **Computational Complexity:** $\mathcal{O}(N_{\text{min}} \cdot k \cdot D)$.
* **Clinical Suitability:** Moderate. Synthesizes plausible intermediate clinical values, but can create synthetic samples inside majority class clusters if noise is present.

#### D. Borderline-SMOTE (Borderline-1 & Borderline-2)
* **Working Principle:** Identifies minority instances located strictly near the decision boundary (where $m$ of its $k$-nearest neighbors belong to the majority class) and applies over-sampling exclusively to these "DANGER" instances.
* **Mathematical Intuition:** For each $\mathbf{x}_i \in \mathcal{S}_{\text{min}}$, let $m' \le k$ be the number of majority instances among its $k$-nearest neighbors in the full dataset:
  $$\mathbf{x}_i \in \begin{cases} \text{NOISE}, & \text{if } m' = k \\ \text{DANGER}, & \text{if } \frac{k}{2} \le m' < k \\ \text{SAFE}, & \text{if } 0 \le m' < \frac{k}{2} \end{cases}$$
  Synthetic generation is restricted to the $\text{DANGER}$ set.
* **Computational Complexity:** $\mathcal{O}(N \cdot k \cdot D)$.
* **Clinical Suitability:** High. Focuses synthetic generation on borderline patient profiles, sharpening the diagnostic boundary.

#### E. SMOTE-NC (Nominal-Continuous)
* **Working Principle:** Extends SMOTE to mixed continuous-categorical feature spaces (such as CKD and Cleveland datasets) by calculating Euclidean distances with a categorical penalty factor $\sigma$, and taking the modal categorical value among neighbors.
* **Mathematical Intuition:** Continuous components interpolate linearly, while categorical attributes are assigned via majority voting among $k$-nearest neighbors weighted by feature variance.
* **Computational Complexity:** $\mathcal{O}(N_{\text{min}} \cdot k \cdot D)$.
* **Clinical Suitability:** High for mixed clinical panels.

#### F. Adaptive Synthetic Sampling (ADASYN)
* **Working Principle:** Uses a weighted distribution $\Gamma_i$ based on the learning difficulty of each minority instance, generating proportionally more synthetic samples for minority points that are difficult to learn.
* **Mathematical Intuition:** Compute density ratio $r_i = \frac{\Delta_i}{k}$, where $\Delta_i$ is the number of majority instances in the $k$-nearest neighborhood of $\mathbf{x}_i$. Normalize $r_i$ to yield probability distribution $p_i = \frac{r_i}{\sum r_i}$, and generate $g_i = p_i \cdot G$ synthetic samples for instance $\mathbf{x}_i$.
* **Computational Complexity:** $\mathcal{O}(N \cdot k \cdot D)$.
* **Clinical Suitability:** Moderate. Focuses heavily on sparse outliers, which may inadvertently over-emphasize clinical measurement noise.

#### G. Tomek Links
* **Working Principle:** Under-sampling data cleaning technique that identifies pairs of minimally distant instances belonging to opposite classes and removes the majority class instance (or both).
* **Mathematical Intuition:** Instances $\mathbf{x}_i \in \mathcal{S}_{\text{min}}$ and $\mathbf{x}_j \in \mathcal{S}_{\text{maj}}$ form a Tomek Link if no instance $\mathbf{x}_k$ exists such that:
  $$d(\mathbf{x}_i, \mathbf{x}_k) < d(\mathbf{x}_i, \mathbf{x}_j) \quad \text{or} \quad d(\mathbf{x}_j, \mathbf{x}_k) < d(\mathbf{x}_i, \mathbf{x}_j)$$
* **Computational Complexity:** $\mathcal{O}(N^2 \cdot D)$.
* **Clinical Suitability:** High for post-processing cleaning.

#### H. Hybrid SMOTE + Tomek Links
* **Working Principle:** Applies SMOTE to synthesize minority instances, followed by Tomek Link identification and pruning to remove overlapping boundary pairs.
* **Mathematical Intuition:** Combines interpolation capacity with decision space boundary clearing.
* **Computational Complexity:** $\mathcal{O}(N_{\text{min}} \cdot k \cdot D + N_{\text{post}}^2 \cdot D)$.
* **Clinical Suitability:** **Very High (Selected Strategy).**

#### I. Hybrid SMOTE + ENN (Edited Nearest Neighbours)
* **Working Principle:** Applies SMOTE over-sampling, then uses Wilson's $k$-Nearest Neighbours rule ($k=3$) to prune any instance whose class label differs from the majority label of its 3 nearest neighbors.
* **Mathematical Intuition:** Provides aggressive boundary cleaning compared to Tomek Links.
* **Computational Complexity:** $\mathcal{O}(N_{\text{min}} \cdot k \cdot D + N_{\text{post}} \cdot k \cdot D)$.
* **Clinical Suitability:** High for noisy clinical data.

---

## 4. Comprehensive IEEE-Style Resampling Algorithm Comparison

TABLE I: IEEE-FORMATTED COMPARATIVE ANALYSIS OF CLASS BALANCING ALGORITHMS

| Algorithm | Paradigm | Mathematical Intuition | Computational Complexity | Primary Advantages | Critical Limitations | Clinical Suitability Ranking |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Over-sampling (ROS)** | Over-sampling | Direct random duplication of minority instances. | $\mathcal{O}(N_{\text{target}})$ | Simple execution; zero information loss. | Severe risk of decision tree overfitting due to identical duplicate instances. | Low (9th) |
| **Random Under-sampling (RUS)** | Under-sampling | Direct random pruning of majority instances. | $\mathcal{O}(N_{\text{maj}})$ | Drastically reduces training runtime and memory. | Discards valuable clinical data points and diagnostic variance. | Low (8th) |
| **Standard SMOTE** | Over-sampling | Linear interpolation between $k$-nearest intra-class neighbors. | $\mathcal{O}(N_{\text{min}} \cdot k \cdot D)$ | Expands minority decision region; avoids exact duplicates. | Synthesizes noisy instances if minority overlaps majority space. | Moderate (6th) |
| **Borderline-SMOTE** | Over-sampling | Over-samples minority instances in boundary "DANGER" regions ($m'/k \in [0.5, 1.0)$). | $\mathcal{O}(N \cdot k \cdot D)$ | Strengthens boundary discriminative power; avoids over-sampling safe interior points. | Sensitive to hyperparameter choice of $k$. | High (3rd) |
| **SMOTE-NC** | Over-sampling | Interpolates continuous features while assigning modal categorical attributes weighted by variance. | $\mathcal{O}(N_{\text{min}} \cdot k \cdot D)$ | Handles mixed continuous-categorical EHR data natively. | Requires manual specification of categorical attribute indices. | High (4th) |
| **ADASYN** | Over-sampling | Adaptive density-weighted interpolation proportional to neighbor difficulty. | $\mathcal{O}(N \cdot k \cdot D)$ | Automatically shifts focus toward hard-to-learn minority regions. | Highly vulnerable to over-sampling isolated measurement outliers/noise. | Moderate (7th) |
| **Tomek Links** | Under-sampling | Prunes mutually closest opposite-class instance pairs. | $\mathcal{O}(N^2 \cdot D)$ | Cleans decision boundary ambiguity without altering interior distributions. | Does not increase minority instance counts on its own. | High (5th - Cleaner) |
| **SMOTE + Tomek** | Hybrid | Combines SMOTE synthetic over-sampling with Tomek Link boundary pruning. | $\mathcal{O}(N_{\text{min}} k D + N_{\text{post}}^2 D)$ | **Simultaneously balances class ratios and removes ambiguous boundary noise.** | Slightly higher computational complexity. | **Most Suitable (1st - Adopted)** |
| **SMOTE + ENN** | Hybrid | Combines SMOTE over-sampling with $k$-NN majority cleaning. | $\mathcal{O}(N_{\text{min}} k D + N_{\text{post}} k D)$ | Aggressive noise removal; generates cleaner class margins. | Can over-prune minority instances in highly overlapped disease spaces. | High (2nd) |

---

## 5. Proposed Adaptive Data Balancing & Leakage-Free Workflow Architecture

### 5.1 Strict Leakage-Free Balancing Protocol
Resampling **must never** be performed on the entire dataset prior to data partitioning. Performing SMOTE or under-sampling on the full dataset before splitting introduces synthetic representations of validation/test data into the training fold, resulting in severe data leakage and invalid performance claims.

```
====================== LEAKAGE-FREE RESAMPLING BOUNDARY ======================

  [ Complete Diagnostic Dataset (100%) ]
                 |
                 v
  [ Stratified Split: 80% Train-Val / 20% Sealed Test ]
                 |
                 +-----------------------------------+
                 |                                   |
                 v                                   v
    [ Training Folds (80%) ]                 [ Sealed Test Set (20%) ]
                 |                                   |
                 | (Fit Preprocessing Transformers)  | (Transform ONLY)
                 v                                   |
    [ Preprocessed Train Fold ]                      |
                 |                                   |
                 v                                   |
    [ Apply Adaptive SMOTE + Tomek ]                 |
                 |                                   |
                 v                                   |
    [ Balanced Train Set (50:50) ]                   |
                 |                                   |
                 v                                   v
    [ Train Model Classifiers ] ------------> [ Final Model Evaluation ]
```

### 5.2 Algorithmic Resampling Workflow
```
Input : Training Fold Feature Matrix X_train, Target Label Vector y_train
Output: Balanced & Boundary-Cleaned Feature Matrix X_balanced, Target Vector y_balanced

1. Compute initial class frequencies: N_maj = count(y_train == 0), N_min = count(y_train == 1)
2. Compute Imbalance Ratio: IR = N_maj / N_min
3. IF IR <= 1.2 THEN
4.     X_balanced, y_balanced <- X_train, y_train  // Dataset is balanced; skip resampling
5. ELSE
6.     IF Dataset contains Categorical Features THEN
7.         X_res, y_res <- Apply SMOTE-NC(X_train, y_train, categorical_features=indices, sampling_strategy=0.85)
8.     ELSE
9.         X_res, y_res <- Apply Borderline-SMOTE(X_train, y_train, sampling_strategy=0.85, k_neighbors=5)
10.    ENDIF
11.    // Apply Tomek Links cleaning post over-sampling
12.    X_balanced, y_balanced <- Apply TomekLinks(X_res, y_res, sampling_strategy='all')
13. ENDIF
14. RETURN X_balanced, y_balanced
```

---

## 6. Clinical & Machine Learning Justifications

### 6.1 Clinical Justification
Applying **SMOTE + Tomek Links** ensures that synthetic patient profiles reflect valid biological states along disease transition boundaries (e.g., pre-diabetic to diabetic metabolic transition) without populating unphysiological feature space. By cleaning Tomek Links post-interpolation, ambiguous overlapping profiles are removed, sharpening the diagnostic boundary for critical clinical decision-making.

### 6.2 Machine Learning Perspective
By setting a target over-sampling ratio ($\text{sampling\_strategy} = 0.85$) followed by Tomek Link pruning, the target class ratio achieves near 50:50 parity. This shifts the classification decision threshold toward optimal F1-Score and PR-AUC performance without over-saturating the feature space with synthetic noise.

---

## 7. Expected Outcomes & Phase Summary

* **Expected Outcome:** Elimination of algorithmic majority bias across all five target disease datasets, yielding statistically significant sensitivity ($\text{Recall} \ge 90\%$) improvements on minority disease instances while maintaining high specificity.
* **Phase Summary:** Phase 6 establishes a publication-ready, mathematically grounded, and leakage-free data balancing framework utilizing **Borderline-SMOTE / SMOTE-NC** paired with **Tomek Links** cleaning, ensuring optimal model stability for baseline ML and deep learning model development in subsequent phases.
