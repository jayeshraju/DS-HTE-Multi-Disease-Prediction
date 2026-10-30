"""
Unit tests for Medical Imaging and Agent FastAPI Endpoints.
Follows Section 4.1:
- Image API endpoints, schema validation, 404/422 responses
- Agent ask endpoint
- Safe deletion endpoint
"""

import io
import json
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_imaging_jobs_list(client):
    res = client.get("/api/image/jobs")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_api_agent_ask_refusal(client):
    payload = {
        "job_id": "nonexistent_job",
        "question": "Can you diagnose if I have pneumonia and what medication to take?"
    }
    res = client.post("/api/agent/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "cannot provide clinical diagnoses" in data["answer"].lower()
    assert "Research prototype, not a diagnostic device." in data["disclaimer"]


def test_api_imaging_predict_missing_model_404(client):
    # Create a small valid test image in memory
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf, format="PNG")
    buf.seek(0)

    files = {"file": ("test.png", buf, "image/png")}
    res = client.post("/api/image/models/nonexistent_model/predict", files=files)
    assert res.status_code == 404
    assert "Model artifacts not found" in res.json()["detail"]


def test_api_imaging_dataset_delete_not_found_404(client):
    res = client.delete("/api/image/datasets/nonexistent_dataset_999")
    assert res.status_code == 404
