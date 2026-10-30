# DS-HTE Multi-Disease Dashboard User Guide

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Overview & Navigation Structure
The web dashboard provides an interactive user interface organized into three primary operational groups:
1. **Tabular Multi-Disease**: Comprehensive exploration of 5 clinical cohorts across 50 outer folds.
2. **AI Assistant (Copilot)**: Grounded AI agent for natural language Q&A, diagnostics, and sandboxed improver experiments.
3. **Image Lab (Medical Vision)**: Ingestion, training, results analysis, Grad-CAM visualization, and inference on medical images.

---

## 2. Tabular Multi-Disease Views
- **Overview & Cohorts**: Displays summary statistics for PIMA, Cleveland, CKD, ILPD, and WBCD, including class prevalence and missingness indicators.
- **Model Performance**:
  - Displays primary 50-fold nested repeated cross-validation metrics.
  - **Two Operating Point Toggle**: Switch between **Balanced (Youden's J)** (screening-oriented) and **Accuracy-Optimal** thresholds.
  - **Previously Inspected Holdout (Secondary)**: Clearly separated secondary validation view.
  - **Trivial Baseline**: Majority-class rate horizontal line plotted on accuracy charts.
- **Threshold Explorer**: Interactive cutoff slider dynamically updating Sensitivity, Specificity, NPV, PPV, and Decision Curve Analysis (DCA) Net Benefit.
- **Compare Models**: Side-by-side performance benchmarks across all 11 model architectures with the **DS-HTE Stacking Ensemble Weights** panel displaying component diversity floors ($w_i \ge 0.01$) and collapse alerts (WBCD LR 92.2%).
- **Calibration & Reliability**: Reliability diagrams, Brier score decompositions, and ECE before vs after temperature scaling.
- **Ablation Study**: Table V architectural component impact across all 5 clinical cohorts.
- **Explainability & Audits**: SHAP global summary beeswarm plots, LIME local explanation bars, and explainer consensus agreements.
- **Try a Patient**: Live clinical risk scoring interface generating downloadable physician PDF reports.
- **Paper vs Reproduced**: Side-by-side comparison with historical paper numbers and signed deltas.

---

## 3. Image Lab Views
- **Upload Dataset**: Drag-and-drop zip ingestion supporting folder structures or CSV label files. Automatic PHI de-identification and duplicate detection.
- **Training Jobs**: Select backbone (ResNet-18, DenseNet-121, EfficientNet-B0), configure patient-grouped splits, and monitor live progress bars.
- **Image Results**: Displays ROC/PR curves, confusion matrices, and Grad-CAM galleries of confident-correct, confident-wrong, and ambiguous cases.
- **Predict Image**: Upload raw radiographs/slices to receive calibrated class probabilities, Mahalanobis OOD flags ("Needs Expert Review"), and interactive Grad-CAM heatmap overlay opacity sliders.

---

## 4. DS-HTE Copilot
- Select Mode: *Explain*, *Analyst*, *Improver*, or *Reporter*.
- Suggested question chips allow instant querying of weak cohorts, baseline comparisons, and honest optimization pathways.
- Click **"Show Work (Tool Trace)"** to inspect underlying tool calls and click **Source Chips** to view the underlying tables.
- Approve or reject sandboxed optimization trials in the **Experiment Queue**.
