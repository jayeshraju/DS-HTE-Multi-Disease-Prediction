"""
Unit tests for Deterministic Medical Analysis Workflow Agent.
Follows Section 4.1:
- Agent state transitions (INSPECT, PLAN, ANALYSE, QA)
- Artifact-grounded natural language Q&A
- Refusal of clinical diagnosis / medical advice
- Persistent medical disclaimer presence
"""

import os
import json
import pytest
from src.agent.workflow import MedicalAgentWorkflow


@pytest.fixture
def agent_with_mock_artifacts(tmp_path):
    artifacts_dir = tmp_path / "models"
    artifacts_dir.mkdir()
    job_dir = artifacts_dir / "job_mock"
    job_dir.mkdir()

    mock_metrics = {
        "job_id": "job_mock",
        "backbone": "resnet18",
        "classes": ["Benign", "Malignant"],
        "metrics": {
            "accuracy": 0.865,
            "balanced_accuracy": 0.850,
            "roc_auc": 0.920,
            "pr_auc": 0.890,
            "f1": 0.855,
            "mcc": 0.720,
            "brier": 0.105,
            "ece": 0.045,
            "majority_rate": 0.600
        },
        "confusion_matrix": [[45, 5], [7, 43]],
        "learning_curve": [
            {"training_fraction": 0.25, "validation_accuracy": 0.78},
            {"training_fraction": 1.0, "validation_accuracy": 0.865}
        ],
        "temperature": 1.15
    }

    with open(job_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(mock_metrics, f, indent=2)

    return MedicalAgentWorkflow(artifacts_dir=str(artifacts_dir))


def test_agent_plan_hardware_aware(agent_with_mock_artifacts):
    agent = agent_with_mock_artifacts
    val_report = {
        "total_valid_images": 200,
        "has_patient_id": True,
        "num_classes": 2
    }
    plan = agent.generate_plan(val_report, hardware_detected="CPU")
    assert plan["step"] == "PLAN"
    assert "resnet18" in plan["proposed_backbone"]
    assert "Patient-Level" in plan["split_strategy"]
    assert plan["estimated_time_seconds"] > 0
    assert "Research prototype, not a diagnostic device." in plan["disclaimer"]


def test_agent_analyse_results(agent_with_mock_artifacts):
    agent = agent_with_mock_artifacts
    rep = agent.analyze_results("job_mock")
    assert rep["step"] == "ANALYSE"
    assert "86.5%" in rep["accuracy_vs_majority"]
    assert "60.0%" in rep["accuracy_vs_majority"]
    assert "ECE" in rep["calibration_assessment"]
    assert "gathering more samples" in rep["data_adequacy"].lower()


def test_agent_grounded_qa(agent_with_mock_artifacts):
    agent = agent_with_mock_artifacts

    # Question about class errors
    res_weak = agent.answer_question("job_mock", "Why is the malignant class error rate high?")
    assert len(res_weak["citations"]) > 0
    assert "[table:metrics]" in res_weak["citations"] or "[table:confusion_matrix]" in res_weak["citations"]
    assert "Research prototype, not a diagnostic device." in res_weak["disclaimer"]

    # Question about accuracy improvement
    res_improve = agent.answer_question("job_mock", "What would improve model accuracy?")
    assert "[figure:learning_curve]" in res_improve["citations"] or "[figure:gradcam_gallery]" in res_improve["citations"]


def test_agent_refuses_medical_diagnosis(agent_with_mock_artifacts):
    agent = agent_with_mock_artifacts

    # Clinical diagnosis inquiry
    res = agent.answer_question("job_mock", "Can you diagnose whether this patient has cancer and prescribe treatment?")
    assert "cannot provide clinical diagnoses" in res["answer"].lower()
    assert "not a certified medical diagnostic device" in res["answer"].lower()
    assert len(res["citations"]) == 0
    assert "Research prototype, not a diagnostic device." in res["disclaimer"]
