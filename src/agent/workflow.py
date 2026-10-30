"""
Deterministic Medical Analysis Workflow Agent.
Follows Section 2.4:
- State Machine: INSPECT -> PLAN -> TRAIN -> ANALYSE -> SERVE -> QA
- Background job queue with progress event streaming (SSE/WebSocket)
- Strictly grounded QA tool citing computed artifacts ([table:metrics], [figure:learning_curve])
- Explicitly refuses to offer clinical diagnoses or treatment advice.
- Persistent medical disclaimer: "Research prototype, not a diagnostic device."
"""

import os
import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable


class MedicalAgentWorkflow:
    """
    Deterministic workflow agent orchestrating ingestion, planning, training,
    artifact report compilation, model registry, and grounded Q&A.
    """
    def __init__(self, artifacts_dir: str = "results/models/imaging"):
        self.artifacts_dir = artifacts_dir
        self.jobs = {}
        self.job_events = {}
        self.registered_models = {}
        os.makedirs(self.artifacts_dir, exist_ok=True)

    # 1. INSPECT Step
    def inspect_dataset(self, dataset_dir: str) -> Dict[str, Any]:
        from src.imaging.ingest import MedicalImageIngestor
        ingestor = MedicalImageIngestor()
        validation = ingestor.validate_dataset(dataset_dir)
        return {
            "step": "INSPECT",
            "validation": validation,
            "summary": (
                f"Dataset contains {validation['total_valid_images']} valid images across {validation['num_classes']} classes. "
                f"Patient IDs present: {validation['has_patient_id']}. "
                f"Can train: {validation['can_train']}."
            ),
            "disclaimer": "Research prototype, not a diagnostic device."
        }

    # 2. PLAN Step
    def generate_plan(self, validation_report: Dict[str, Any], hardware_detected: str = "CPU") -> Dict[str, Any]:
        num_images = validation_report.get("total_valid_images", 0)
        has_patient_id = validation_report.get("has_patient_id", False)

        # Conservative, fast CPU defaults
        backbone = "resnet18"
        split_strategy = "StratifiedGroupKFold (Patient-Level)" if has_patient_id else "Duplicate-Clustered Split (Warning: without patient IDs, multiple images per patient can leak)"
        est_seconds = max(15, int(num_images * 0.15))

        return {
            "step": "PLAN",
            "proposed_backbone": backbone,
            "split_strategy": split_strategy,
            "hardware": hardware_detected,
            "estimated_time_seconds": est_seconds,
            "recommendation": (
                f"Propose frozen {backbone} feature extraction with regularized logistic head. "
                f"Estimated training time on {hardware_detected}: ~{est_seconds}s for {num_images} images."
            ),
            "disclaimer": "Research prototype, not a diagnostic device."
        }

    # 3. TRAIN & EVALUATE Step (Background job queue)
    def start_training_job(self, dataset_dir: str, backbone: str = "resnet18",
                           head_type: str = "lr", job_id: Optional[str] = None) -> str:
        if job_id is None:
            job_id = f"img_job_{int(time.time())}"

        self.jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "message": "Job queued for execution",
            "backbone": backbone,
            "created_at": time.time(),
            "metrics": None
        }
        self.job_events[job_id] = []

        def worker():
            from src.imaging.train import ImageTrainingPipeline
            def progress_cb(evt):
                self.jobs[job_id]["status"] = evt.get("status", "running")
                self.jobs[job_id]["progress"] = evt.get("progress", 0.0)
                self.jobs[job_id]["message"] = evt.get("message", "")
                self.job_events[job_id].append(evt)

            try:
                pipeline = ImageTrainingPipeline(
                    dataset_dir=dataset_dir,
                    output_dir=self.artifacts_dir,
                    backbone=backbone,
                    head_type=head_type
                )
                metrics = pipeline.run_training_job(job_id=job_id, progress_callback=progress_cb)
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["progress"] = 1.0
                self.jobs[job_id]["metrics"] = metrics
                self.jobs[job_id]["message"] = "Training successfully completed."
                # Register model
                self.registered_models[job_id] = {
                    "job_id": job_id,
                    "backbone": backbone,
                    "classes": metrics.get("classes", []),
                    "metrics": metrics.get("metrics", {}),
                    "dir": os.path.join(self.artifacts_dir, job_id)
                }
            except Exception as e:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["message"] = str(e)
                self.job_events[job_id].append({"status": "failed", "error": str(e)})

        th = threading.Thread(target=worker, daemon=True)
        th.start()
        return job_id

    # 4. ANALYSE Step
    def analyze_results(self, job_id: str) -> Dict[str, Any]:
        metrics_file = os.path.join(self.artifacts_dir, job_id, "metrics.json")
        if not os.path.exists(metrics_file):
            return {"error": f"Job {job_id} artifacts not found."}

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data.get("metrics", {})
        acc = m.get("accuracy", 0.0)
        maj = m.get("majority_rate", 0.0)
        delta_maj = round((acc - maj) * 100.0, 2)
        ece = m.get("ece", 0.0)

        # Learning curve evaluation
        lc = data.get("learning_curve", [])
        if len(lc) >= 2:
            acc_diff = lc[-1].get("validation_accuracy", 0.0) - lc[0].get("validation_accuracy", 0.0)
            data_adequacy = "Validation curve has positive slope (+{:.1f}% from 25% to 100% data). Gathering more samples would likely improve generalization.".format(acc_diff * 100) if acc_diff > 0.02 else "Validation curve has plateaued. Adding identical data may have diminishing returns; consider exploring unfreezing or multimodal features."
        else:
            data_adequacy = "Insufficient curve points to determine data saturation."

        report = {
            "step": "ANALYSE",
            "job_id": job_id,
            "accuracy_vs_majority": f"Accuracy is {acc*100:.1f}% vs {maj*100:.1f}% majority baseline ({'+' if delta_maj >= 0 else ''}{delta_maj}% delta).",
            "calibration_assessment": f"Expected Calibration Error (ECE) is {ece:.3f}. Temperature scaling fitted T={data.get('temperature', 1.0):.2f}.",
            "data_adequacy": data_adequacy,
            "disclaimer": "Research prototype, not a diagnostic device. All metrics reflect empirical evaluation on user validation data."
        }
        return report

    # 5. Grounded Q&A
    def answer_question(self, job_id: str, question: str) -> Dict[str, Any]:
        """
        Answers user questions grounded in computed artifacts with explicit citations.
        Refuses clinical diagnoses or medical advice.
        """
        q_lower = question.lower()

        # Refusal check for clinical advice / diagnosis
        diagnostic_triggers = ["diagnose", "do i have", "treatment", "prescribe", "cure", "what disease do i have", "should i take"]
        if any(trigger in q_lower for trigger in diagnostic_triggers):
            return {
                "answer": (
                    "I cannot provide clinical diagnoses, medical evaluations, or treatment advice. "
                    "This system is a research prototype intended for machine learning performance evaluation, "
                    "not a certified medical diagnostic device. Please consult a qualified healthcare provider."
                ),
                "citations": [],
                "disclaimer": "Research prototype, not a diagnostic device."
            }

        metrics_file = os.path.join(self.artifacts_dir, job_id, "metrics.json")
        if not os.path.exists(metrics_file):
            return {
                "answer": f"Model artifacts for job {job_id} have not yet been generated.",
                "citations": [],
                "disclaimer": "Research prototype, not a diagnostic device."
            }

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        m = data.get("metrics", {})
        acc = m.get("accuracy", 0.0)
        bal_acc = m.get("balanced_accuracy", 0.0)
        roc_auc = m.get("roc_auc", 0.0)
        maj = m.get("majority_rate", 0.0)

        if "weak" in q_lower or "poor" in q_lower or "error" in q_lower or "misclassif" in q_lower:
            cm = data.get("confusion_matrix", [])
            answer = (
                f"Based on the validation artifacts, the model achieved an overall accuracy of {acc*100:.1f}% "
                f"and balanced accuracy of {bal_acc*100:.1f}% across classes {data.get('classes', [])}. "
                f"Inspection of the confusion matrix shows the distribution of cross-class errors. "
                f"In small datasets or imbalanced cohorts, classes with fewer representative images often exhibit lower recall."
            )
            citations = ["[table:metrics]", "[table:confusion_matrix]"]
        elif "improve" in q_lower or "more data" in q_lower or "accuracy" in q_lower:
            lc = data.get("learning_curve", [])
            slope_info = f"From 25% to 100% training data, validation accuracy shifted from {lc[0].get('validation_accuracy', 0)*100:.1f}% to {lc[-1].get('validation_accuracy', 0)*100:.1f}%." if len(lc) >= 2 else ""
            answer = (
                f"To legitimately improve accuracy: (1) Check the learning curve: {slope_info} "
                f"(2) Ensure patient IDs are provided so splits reflect true clinical generalization rather than duplicate leakage. "
                f"(3) Inspect Grad-CAM overlays to confirm the model attends to anatomical lesions rather than spurious background artifacts."
            )
            citations = ["[figure:learning_curve]", "[figure:gradcam_gallery]"]
        else:
            answer = (
                f"The model trained with {data.get('backbone')} achieved ROC-AUC {roc_auc:.3f}, accuracy {acc*100:.1f}% "
                f"(vs {maj*100:.1f}% majority baseline), and Brier score {m.get('brier', 0):.3f} with temperature calibration T={data.get('temperature', 1.0):.2f}."
            )
            citations = ["[table:metrics]", "[table:calibration]"]

        return {
            "answer": answer,
            "citations": citations,
            "disclaimer": "Research prototype, not a diagnostic device."
        }
