"""
Unit tests for Medical Imaging Models, Grad-CAM, Calibration, and OOD Safety Layer.
Follows Section 4.1:
- Feature extractor output shapes
- Fast CPU smoke training
- Temperature scaling calibration on validation logits
- Grad-CAM saliency map spatial dimensions
- Mahalanobis OOD detector separation on noise vs in-distribution
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.imaging.preprocessing import MedicalImagePreprocessor
from src.imaging.models import MedicalFeatureExtractor, MedicalImageClassifier, TemperatureScaler
from src.imaging.gradcam import GradCAM
from src.imaging.ood import MahalanobisOODDetector


def test_feature_extractor_shapes():
    extractor = MedicalFeatureExtractor(backbone_name="resnet18", device="cpu")
    dummy_batch = torch.randn(4, 3, 224, 224)
    feats = extractor.extract_features(dummy_batch)
    assert feats.shape == (4, 512)


def test_temperature_scaler_optimization():
    rng = np.random.RandomState(42)
    N = 100
    y_true = rng.binomial(1, 0.4, size=N)
    # Overconfident uncalibrated logits
    logits = (y_true * 6.0 - 3.0) + rng.normal(0, 1.0, size=N)

    scaler = TemperatureScaler()
    opt_temp = scaler.fit(logits, y_true)

    assert opt_temp > 0.05
    cal_probs = scaler.scale_probabilities(logits)
    assert cal_probs.shape == (N, 2)
    assert np.allclose(np.sum(cal_probs, axis=1), 1.0)


def test_gradcam_overlay_and_shape():
    extractor = MedicalFeatureExtractor(backbone_name="resnet18", device="cpu")
    gradcam = GradCAM(extractor.model, target_layer_name="layer4")

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate_heatmap(dummy_input, class_idx=0)

    assert heatmap.shape == (224, 224)
    assert 0.0 <= np.min(heatmap)
    assert np.max(heatmap) <= 1.0

    orig_pil = Image.new("RGB", (224, 224), color=(100, 150, 200))
    blended, b64_overlay = gradcam.generate_overlay(orig_pil, dummy_input, class_idx=0)
    assert blended.size == (224, 224)
    assert b64_overlay.startswith("data:image/png;base64,")


def test_mahalanobis_ood_separation():
    rng = np.random.RandomState(42)
    # In-distribution class clusters
    c0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(100, 2))
    c1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(100, 2))
    X_train = np.vstack([c0, c1])
    y_train = np.array([0] * 100 + [1] * 100)

    val_c0 = rng.normal(loc=[-2.0, -2.0], scale=0.5, size=(20, 2))
    val_c1 = rng.normal(loc=[2.0, 2.0], scale=0.5, size=(20, 2))
    X_val = np.vstack([val_c0, val_c1])

    ood_detector = MahalanobisOODDetector()
    ood_detector.fit(X_train, y_train)
    threshold = ood_detector.calibrate_threshold(X_val, percentile=95.0)

    # In-distribution sample
    in_dist_sample = np.array([-2.1, -1.9])
    res_in = ood_detector.evaluate_sample(in_dist_sample, prob_distribution=np.array([0.9, 0.1]))
    assert not res_in["is_ood"]
    assert res_in["mahalanobis_distance"] < threshold

    # Severe out-of-distribution noise sample far from centroids
    ood_sample = np.array([15.0, -18.0])
    res_ood = ood_detector.evaluate_sample(ood_sample, prob_distribution=np.array([0.5, 0.5]))
    assert res_ood["is_ood"]
    assert res_ood["needs_expert_review"]
    assert res_ood["mahalanobis_distance"] > threshold
    assert res_ood["safety_banner"] is not None


def test_fast_training_smoke(tmp_path):
    """Smoke test: end-to-end training and evaluation on 24 synthetic images completes in under 10 seconds."""
    from src.imaging.train import ImageTrainingPipeline
    dataset_dir = tmp_path / "synthetic_cohort"
    c0_dir = dataset_dir / "benign"
    c1_dir = dataset_dir / "malignant"
    c0_dir.mkdir(parents=True)
    c1_dir.mkdir(parents=True)

    for i in range(12):
        im0 = Image.new("RGB", (64, 64), color=(50, 100, 150))
        im0.save(c0_dir / f"b_{i}.png")
        im1 = Image.new("RGB", (64, 64), color=(200, 80, 40))
        im1.save(c1_dir / f"m_{i}.png")

    pipeline = ImageTrainingPipeline(
        dataset_dir=str(dataset_dir),
        output_dir=str(tmp_path / "models"),
        backbone="resnet18",
        head_type="lr",
        n_splits=2,
        target_size=224,
        random_state=42
    )

    metrics = pipeline.run_training_job(job_id="smoke_job")
    assert "metrics" in metrics
    assert "accuracy" in metrics["metrics"]
    assert "roc_auc" in metrics["metrics"]
    assert "galleries" in metrics
    assert metrics["metrics"]["accuracy"] >= 0.50
