import pytest
from fastapi.testclient import TestClient
from src.api.app import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_api_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "loaded_pipelines" in data

def test_api_predict_diabetes(client):
    payload = {
        "patient_id": "PT-TEST-001",
        "disease_name": "diabetes",
        "age": 45,
        "gender": "Female",
        "features": {
            "Pregnancies": 2.0,
            "Glucose": 120.0,
            "BloodPressure": 75.0,
            "SkinThickness": 25.0,
            "Insulin": 50.0,
            "BMI": 28.5,
            "DiabetesPedigreeFunction": 0.45,
            "Age": 45
        }
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["patient_id"] == "PT-TEST-001"
    assert data["disease_name"] == "diabetes"
    assert 0.0 <= data["pathological_probability"] <= 1.0
    assert 0.0 <= data["prediction_confidence"] <= 1.0
    assert data["status"] == "SUCCESS"
    assert "risk_tier" in data
    assert "feature_attributions" in data
    assert len(data["selected_features"]) > 0

def test_api_report_generation(client):
    payload = {
        "patient_id": "PT-TEST-002",
        "disease_name": "heart",
        "age": 55,
        "gender": "Male",
        "features": {
            "age": 55.0,
            "sex": 1.0,
            "cp": 3.0,
            "trestbps": 140.0,
            "chol": 250.0,
            "fbs": 0.0,
            "restecg": 1.0,
            "thalach": 150.0,
            "exang": 0.0,
            "oldpeak": 1.5,
            "slope": 2.0,
            "ca": 0.0,
            "thal": 3.0
        }
    }
    
    response = client.post("/report", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert len(response.content) > 1000 # Verify it returns a substantial PDF file
