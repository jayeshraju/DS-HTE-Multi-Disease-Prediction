# DS-HTE AI Copilot User & Developer Guide

> **Notice**: Research prototype, not a diagnostic device.

---

## 1. Copilot Architecture & Operating Modes
The **DS-HTE Copilot** is a deterministic, tool-using assistant embedded directly into the dashboard. It serves four distinct modes:

1. **Explain Mode (Q&A)**: Explains empirical discrepancies, model architectures, and statistical comparisons using verified artifact citations.
2. **Analyst Mode (Diagnostics)**: Proactively diagnoses dataset vulnerabilities, including class imbalance penalties, majority-rate traps, ensemble collapse, and small-N variance.
3. **Improver Mode (Sandboxed Optimization)**: Pre-registers and executes experimental model/feature modifications strictly on inner-validation development splits (capped at 40 trials per cohort).
4. **Reporter Mode (Documentation)**: Drafts structured scientific report summaries and tables directly from computed artifacts.

---

## 2. Tool Inventory (16 Allow-Listed Tools)
The Copilot accesses 16 typed, sandboxed tools that query artifacts from `results/`:
- `list_runs`: Lists active run manifest metadata and config hashes.
- `get_metrics`: Queries 11-metric performance profiles from Table II.
- `compare_models`: Compares baseline classifiers from Table III.
- `get_stats`: Retrieves paired statistical tests with direction and Holm/BH p-values.
- `get_ablation`: Retrieves architectural ablation deltas from Table V.
- `get_calibration`: Checks ECE, Brier score, and calibration curves.
- `get_ensemble_weights`: Inspects per-component stacking weights and collapse flags.
- `get_explanations`: Inspects top SHAP features and SHAP-LIME consensus scores.
- `get_dataset_profile`: Loads data profiling reports and duplicate cluster audits.
- `validate_dataset`: Verifies file integrity, schema alignment, and magic bytes.
- `predict_tabular`: Runs live calibrated risk prediction.
- `predict_image`: Runs image inference, Grad-CAM saliency, and OOD detection.
- `explain_prediction`: Generates local patient feature attributions.
- `propose_experiment`: Proposes candidate trials for the user approval queue.
- `run_experiment`: Executes approved trials inside the restricted inner-CV sandbox.
- `build_report_section`: Assembles markdown sections from verified tables.

---

## 3. Grounding Verification & Zero-Hallucination Policy
- Every numerical value generated in Copilot answers is intercepted by `GroundingVerifier`.
- The verifier extracts all floats and percentages from the draft text and matches them against tool output arrays.
- Any ungrounded number is flagged with an explicit warning banner:
  > ⚠️ **Grounding Verification Notice**: The following values could not be matched with computed artifact evidence...
- Every message includes clickable **Source Chips** (`[table:table_2_dshte_metrics]`) and an expandable **"Show Work (Tool Trace)"** drawer.

---

## 4. Safety Guardrails & Refusals
- **Clinical Refusals**: Requests for individual medical diagnoses, treatment choices, medication dosing, or clinical prognosis are strictly rejected with an explicit clinical disclaimer and referral to medical doctors.
- **Passive Data Principle**: All uploaded CSV cells, filenames, and image metadata are sanitized to neutralize prompt injection attacks (e.g., `"ignore previous instructions"`).
- **Inner-CV Boundary**: The improver tool cannot read outer test labels or sealed holdout partitions under any circumstance.
