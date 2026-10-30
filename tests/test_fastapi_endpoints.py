"""
Tests for Section 10: FastAPI service endpoints.
Validates:
- GET /health -> 200
- POST /predict/{cohort} -> 200 on valid input, 422 on invalid input
- POST /explain/{cohort} -> 200
- POST /report/{cohort} -> 200, Content-Type application/pdf, starts with %PDF
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "HEALTHY"


def test_predict_endpoint_valid_and_invalid(client):
    # Valid input
    valid_payload = {
        "patient_id": "PT-101",
        "features": {"Glucose": 120.0, "BMI": 28.5, "Age": 45.0}
    }
    res_val = client.post("/predict/pima", json=valid_payload)
    assert res_val.status_code == 200
    data = res_val.json()
    assert 0.0 <= data["risk_probability"] <= 1.0
    assert data["prediction"] in [0, 1]

    # Invalid input (missing patient_id -> 422 Unprocessable Entity)
    invalid_payload = {"features": {"Glucose": 120.0}}
    res_inval = client.post("/predict/pima", json=invalid_payload)
    assert res_inval.status_code == 422


def test_explain_endpoint(client):
    payload = {
        "patient_id": "PT-102",
        "features": {"Glucose": 140.0, "BMI": 32.0, "BloodPressure": 80.0}
    }
    res = client.post("/explain/pima", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) > 0


def test_report_endpoint_pdf_and_disclaimer(client):
    payload = {
        "patient_id": "PT-103",
        "features": {"Glucose": 150.0, "BMI": 35.0, "Age": 60.0}
    }
    res = client.post("/report/pima", json=payload)
    assert res.status_code == 200
    assert res.headers["content-type"] == "application/pdf"
    
    # Verify PDF starts with %PDF
    content = res.content
    assert content.startswith(b"%PDF"), f"Expected %PDF header, got {content[:10]}"
    
    # Verify disclaimer is embedded in the PDF byte stream
    assert b"Decision support only, not a diagnosis" in content
