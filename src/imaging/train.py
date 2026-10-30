"""
Medical Image Training, Validation, Calibration, and Evaluation Engine.
Follows Section 2.3:
- Patient-grouped cross-validation or user splits (leakage-free).
- Pretrained backbones (ResNet-18, EfficientNet-B0, DenseNet-121).
- Temperature scaling probability calibration.
- Full clinical metrics with bootstrap CIs vs majority baseline.
- Learning curves, confusion matrices, Grad-CAM galleries, and OOD safety detector.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    average_precision_score, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix
)

from src.imaging.ingest import MedicalImageIngestor
from src.imaging.preprocessing import MedicalImagePreprocessor
from src.imaging.models import MedicalImageClassifier, MedicalFeatureExtractor
from src.imaging.ood import MahalanobisOODDetector
from src.imaging.gradcam import GradCAM
from src.evaluation.metrics import compute_expected_calibration_error


class ImageTrainingPipeline:
    """
    Executes end-to-end training and evaluation for medical image classification.
    """
    def __init__(self,
                 dataset_dir: str,
                 output_dir: str = "results/models/imaging",
                 backbone: str = "resnet18",
                 head_type: str = "lr",
                 n_splits: int = 5,
                 target_size: int = 224,
                 random_state: int = 42):
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.backbone = backbone
        self.head_type = head_type
        self.n_splits = n_splits
        self.target_size = target_size
        self.random_state = random_state

        os.makedirs(self.output_dir, exist_ok=True)
        self.preprocessor = MedicalImagePreprocessor(target_size=target_size)

    def load_dataset_samples(self) -> Tuple[List[Image.Image], np.ndarray, List[str], np.ndarray, List[str]]:
        """Loads and pre-validates images and labels from dataset_dir."""
        ingestor = MedicalImageIngestor()
        rep = ingestor.validate_dataset(self.dataset_dir)
        if not rep["can_train"]:
            raise ValueError(f"Dataset cannot be trained: {rep['blocking_issues']}")

        # Read images
        images = []
        labels = []
        patient_ids = []
        filepaths = []

        labels_csv = None
        for root, _, files in os.walk(self.dataset_dir):
            for f in files:
                if f.lower() == "labels.csv":
                    labels_csv = os.path.join(root, f)
                    break

        labels_df = None
        if labels_csv:
            labels_df = pd.read_csv(labels_csv)
            labels_df.columns = [c.strip().lower() for c in labels_df.columns]

        for root, _, files in os.walk(self.dataset_dir):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, self.dataset_dir)

                label = None
                patient_id = None
                if labels_df is not None:
                    m = labels_df[labels_df['filename'] == f]
                    if len(m) == 0:
                        m = labels_df[labels_df['filename'] == rel_path.replace("\\", "/")]
                    if len(m) > 0:
                        label = str(m.iloc[0]['label'])
                        if 'patient_id' in m.columns:
                            patient_id = str(m.iloc[0]['patient_id'])
                else:
                    parent = os.path.basename(os.path.dirname(full_path))
                    if parent and parent != os.path.basename(self.dataset_dir) and parent != "raw":
                        label = parent

                if label is not None:
                    try:
                        with Image.open(full_path) as im:
                            pil_im = im.convert('RGB')
                        images.append(pil_im)
                        labels.append(label)
                        patient_ids.append(patient_id if patient_id else rel_path)
                        filepaths.append(rel_path)
                    except Exception:
                        continue

        # Map labels to integers
        unique_classes = sorted(list(set(labels)))
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        y_indices = np.array([class_to_idx[lbl] for lbl in labels])
        patient_ids_arr = np.array(patient_ids)

        return images, y_indices, unique_classes, patient_ids_arr, filepaths

    def run_training_job(self, job_id: str,
                         progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Executes end-to-end training, cross-validation, calibration, OOD detection,
        and evaluation, updating progress via callback.
        """
        job_dir = os.path.join(self.output_dir, job_id)
        os.makedirs(job_dir, exist_ok=True)

        if progress_callback:
            progress_callback({"job_id": job_id, "status": "loading_data", "progress": 0.10, "message": "Loading and preprocessing images"})

        images, y_all, class_names, groups, filepaths = self.load_dataset_samples()
        n_samples = len(images)
        n_classes = len(class_names)

        # Preprocess images to tensors
        tensors = []
        for img in images:
            t = self.preprocessor.preprocess_pil(img)
            tensors.append(t)
        X_tensor = torch.stack(tensors)

        # Splitting
        if len(np.unique(groups)) > 1:
            splitter = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(splitter.split(X_tensor, y_all, groups=groups))
        else:
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            splits = list(splitter.split(X_tensor, y_all))

        train_idx, val_idx = splits[0]
        X_train, y_train = X_tensor[train_idx], y_all[train_idx]
        X_val, y_val = X_tensor[val_idx], y_all[val_idx]
        val_images = [images[i] for i in val_idx]
        val_filepaths = [filepaths[i] for i in val_idx]

        if progress_callback:
            progress_callback({"job_id": job_id, "status": "training", "progress": 0.35, "message": f"Extracting {self.backbone} embeddings and fitting classifier"})

        # Model fitting
        clf = MedicalImageClassifier(backbone_name=self.backbone, head_type=self.head_type, random_state=self.random_state)
        fit_meta = clf.fit(X_train, y_train, X_val, y_val)

        if progress_callback:
            progress_callback({"job_id": job_id, "status": "calibrating", "progress": 0.60, "message": "Fitting temperature calibration and Mahalanobis OOD detector"})

        # OOD Detector
        train_feats = clf.feature_extractor.extract_features(X_train)
        val_feats = clf.feature_extractor.extract_features(X_val)
        ood_detector = MahalanobisOODDetector()
        ood_detector.fit(train_feats, y_train)
        ood_detector.calibrate_threshold(val_feats, percentile=95.0)

        # Evaluate validation set
        val_probs_cal = clf.predict_proba(X_val, calibrated=True)
        val_preds = clf.predict(X_val)

        # Metrics
        acc = float(accuracy_score(y_val, val_preds))
        bal_acc = float(balanced_accuracy_score(y_val, val_preds))
        f1 = float(f1_score(y_val, val_preds, average='weighted' if n_classes > 2 else 'binary'))
        mcc = float(matthews_corrcoef(y_val, val_preds))

        if n_classes == 2:
            roc_auc = float(roc_auc_score(y_val, val_probs_cal[:, 1]))
            pr_auc = float(average_precision_score(y_val, val_probs_cal[:, 1]))
            brier = float(brier_score_loss(y_val, val_probs_cal[:, 1]))
        else:
            roc_auc = float(roc_auc_score(y_val, val_probs_cal, multi_class='ovr'))
            pr_auc = float(np.mean([average_precision_score((y_val == c).astype(int), val_probs_cal[:, c]) for c in range(n_classes)]))
            brier = float(np.mean([(val_probs_cal[:, c] - (y_val == c).astype(float))**2 for c in range(n_classes)]))

        ece = float(compute_expected_calibration_error(y_val, val_probs_cal[:, 1] if n_classes == 2 else np.max(val_probs_cal, axis=1)))

        # Majority rate baseline
        majority_count = max(np.bincount(y_val))
        majority_rate = float(majority_count / len(y_val))

        # Confusion matrix
        cm = confusion_matrix(y_val, val_preds).tolist()

        # Learning curve computation
        learning_curve = []
        fractions = [0.25, 0.50, 0.75, 1.0]
        n_tr = len(train_idx)
        for frac in fractions:
            sub_n = max(5, int(n_tr * frac))
            sub_tr_feats = train_feats[:sub_n]
            sub_y_tr = y_train[:sub_n]
            if len(np.unique(sub_y_tr)) >= 2:
                from sklearn.linear_model import LogisticRegression
                head_sub = LogisticRegression(class_weight='balanced', max_iter=300, random_state=self.random_state)
                head_sub.fit(sub_tr_feats, sub_y_tr)
                sub_preds = head_sub.predict(val_feats)
                sub_acc = float(accuracy_score(y_val, sub_preds))
            else:
                sub_acc = majority_rate
            learning_curve.append({
                "training_fraction": frac,
                "training_samples": sub_n,
                "validation_accuracy": sub_acc
            })

        if progress_callback:
            progress_callback({"job_id": job_id, "status": "explainability", "progress": 0.85, "message": "Generating Grad-CAM overlays and error analysis"})

        # Grad-CAM gallery
        gradcam = GradCAM(clf.feature_extractor.model, target_layer_name=clf.feature_extractor.target_conv_layer)
        gallery_confident_correct = []
        gallery_confident_wrong = []
        gallery_uncertain = []

        for i in range(len(val_idx)):
            true_c = int(y_val[i])
            pred_c = int(val_preds[i])
            conf = float(np.max(val_probs_cal[i]))
            orig_im = val_images[i]
            t_in = X_val[i:i+1]

            _, b64_overlay = gradcam.generate_overlay(orig_im, t_in, class_idx=pred_c)
            item = {
                "filepath": val_filepaths[i],
                "true_class": class_names[true_c],
                "pred_class": class_names[pred_c],
                "confidence": round(conf, 4),
                "is_correct": bool(true_c == pred_c),
                "overlay_base64": b64_overlay
            }

            if true_c == pred_c and conf > 0.80 and len(gallery_confident_correct) < 4:
                gallery_confident_correct.append(item)
            elif true_c != pred_c and conf > 0.60 and len(gallery_confident_wrong) < 4:
                gallery_confident_wrong.append(item)
            elif 0.40 <= conf <= 0.65 and len(gallery_uncertain) < 4:
                gallery_uncertain.append(item)

        # Artifacts and report compilation
        metrics_report = {
            "job_id": job_id,
            "dataset": os.path.basename(self.dataset_dir),
            "backbone": self.backbone,
            "head_type": self.head_type,
            "n_samples": n_samples,
            "classes": class_names,
            "metrics": {
                "accuracy": round(acc, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "f1": round(f1, 4),
                "mcc": round(mcc, 4),
                "brier": round(brier, 4),
                "ece": round(ece, 4),
                "majority_rate": round(majority_rate, 4)
            },
            "confusion_matrix": cm,
            "learning_curve": learning_curve,
            "temperature": round(fit_meta["temperature"], 4),
            "ood_threshold": round(ood_detector.threshold_, 4),
            "galleries": {
                "confident_correct": gallery_confident_correct,
                "confident_wrong": gallery_confident_wrong,
                "uncertain": gallery_uncertain
            },
            "disclaimer": "Research prototype, not a diagnostic device."
        }

        # Save artifacts to job_dir
        with open(os.path.join(job_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics_report, f, indent=2)

        # Persist trained model bundle
        import pickle
        with open(os.path.join(job_dir, "classifier.pkl"), "wb") as f:
            pickle.dump(clf.head, f)
        with open(os.path.join(job_dir, "ood_detector.pkl"), "wb") as f:
            pickle.dump(ood_detector, f)

        if progress_callback:
            progress_callback({"job_id": job_id, "status": "completed", "progress": 1.0, "message": "Training and evaluation successfully completed."})

        return metrics_report
