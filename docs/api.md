# DS-HTE REST & SSE API Reference

> **Notice**: Research prototype, not a diagnostic device.

Base URL: `http://localhost:8000`

---

## 1. Core Endpoints

### `GET /health`
Returns service operational health, framework version, and loaded pipeline statuses.
```json
{
  "status": "HEALTHY",
  "framework": "DS-HTE Multi-Disease Prediction Framework",
  "version": "2.0.0",
  "supported_cohorts": ["pima", "cleveland", "ckd", "ilpd", "wbcd"]
}
```

### `POST /predict/{cohort}` or `POST /api/predict/{cohort}`
Runs calibrated inference on tabular patient attributes.
```bash
curl -X POST http://localhost:8000/api/predict/pima \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "P_001",
    "features": {
      "Pregnancies": 2, "Glucose": 140, "BloodPressure": 80,
      "SkinThickness": 28, "Insulin": 110, "BMI": 31.2,
      "DiabetesPedigreeFunction": 0.52, "Age": 45
    }
  }'
```

### `POST /report/{cohort}` or `POST /api/report/{cohort}`
Generates a downloadable, high-resolution physician PDF clinical risk report.

---

## 2. Copilot Endpoints (Gate 1)

### `POST /api/copilot/ask`
Submits a query to the DS-HTE Copilot deterministic state machine.
```json
// Request
{
  "query": "Why is ILPD weak?",
  "mode": "explain",
  "cohort": "ILPD"
}

// Response
{
  "answer": "### Analysis of ILPD Performance\n\nOn the ILPD clinical cohort...",
  "mode": "explain",
  "cohort": "ILPD",
  "tool_calls": [...],
  "sources": [{"tool": "get_metrics", "source": "results/tables/table_2_dshte_metrics.csv", "label": "table_2_dshte_metrics"}],
  "is_grounded": true,
  "disclaimer": "Research prototype, not a diagnostic device."
}
```

### `GET /api/copilot/proposals`
Returns queue of pending and executed experiment proposals.

### `POST /api/copilot/proposals/{proposal_id}/approve`
Executes an approved trial inside the restricted inner-CV sandbox.

---

## 3. Medical Vision & Image Lab Endpoints (Gate 2)

### `POST /api/image/datasets`
Accepts a `.zip` archive containing medical images and metadata for ingestion and PHI de-identification.

### `POST /api/image/models/{id}/predict`
Accepts an image file (`multipart/form-data`) and returns:
- `predicted_class`: string
- `class_probabilities`: dictionary of floats
- `uncertainty_score`: float
- `is_ood`: boolean
- `needs_expert_review`: boolean
- `gradcam_overlay`: base64 encoded PNG overlay
- `disclaimer`: "Research prototype, not a diagnostic device."
