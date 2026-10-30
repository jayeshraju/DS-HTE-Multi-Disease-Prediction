"""
Medical Image Classification Backbones and Calibration Engine.
Follows Section 2.3:
- Pretrained Torchvision backbones: ResNet-18, EfficientNet-B0, DenseNet-121.
- Fast CPU-feasible feature extraction + regularized head (LogisticRegression / GBDT) / optional fine-tuning.
- Post-hoc temperature scaling calibration on validation logits.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy.optimize import minimize_scalar


class TemperatureScaler(nn.Module):
    """
    Applies Platt-like temperature scaling to multi-class or binary logits:
    p = softmax(logits / T) or sigmoid(logits / T).
    Temperature T is optimized strictly on validation logits without touching test data.
    """
    def __init__(self):
        super().__init__()
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """Optimizes scalar temperature T > 0 on validation set."""
        y_true = np.asarray(y_true, dtype=int)
        is_binary = (len(np.unique(y_true)) <= 2) and (logits.ndim == 1 or logits.shape[1] == 1)

        if is_binary and logits.ndim == 2:
            logits = logits[:, 0]

        def nll_obj(t_val):
            if t_val <= 1e-4:
                return 1e6
            scaled = logits / t_val
            if is_binary:
                # Binary cross entropy
                probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
                probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
                loss = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1.0 - probs))
            else:
                # Multiclass cross entropy
                exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
                probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
                probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
                correct_log_probs = np.log(probs[np.arange(len(y_true)), y_true])
                loss = -np.mean(correct_log_probs)
            return float(loss)

        res = minimize_scalar(nll_obj, bounds=(0.05, 5.0), method='bounded')
        self.temperature = float(res.x)
        return self.temperature

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Returns calibrated probabilities using fitted temperature."""
        scaled = logits / max(1e-4, self.temperature)
        if scaled.ndim == 1 or (scaled.ndim == 2 and scaled.shape[1] == 1):
            if scaled.ndim == 2:
                scaled = scaled[:, 0]
            probs = 1.0 / (1.0 + np.exp(-np.clip(scaled, -30, 30)))
            return np.vstack([1.0 - probs, probs]).T
        else:
            exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            return exp_s / np.sum(exp_s, axis=1, keepdims=True)


class MedicalFeatureExtractor:
    """
    Extracts deep visual embeddings from pretrained CNN/Vision backbones.
    """
    def __init__(self, backbone_name: str = "resnet18", device: Optional[str] = None):
        self.backbone_name = backbone_name.lower()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._build_backbone()

    def _build_backbone(self):
        if self.backbone_name == "resnet18":
            weights = tv_models.ResNet18_Weights.DEFAULT
            model = tv_models.resnet18(weights=weights)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.model = model
            self.target_conv_layer = "layer4"
        elif self.backbone_name == "efficientnet_b0":
            weights = tv_models.EfficientNet_B0_Weights.DEFAULT
            model = tv_models.efficientnet_b0(weights=weights)
            self.feature_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            self.model = model
            self.target_conv_layer = "features.8"
        elif self.backbone_name == "densenet121":
            weights = tv_models.DenseNet121_Weights.DEFAULT
            model = tv_models.densenet121(weights=weights)
            self.feature_dim = model.classifier.in_features
            model.classifier = nn.Identity()
            self.model = model
            self.target_conv_layer = "features.denseblock4"
        else:
            # Default to resnet18
            weights = tv_models.ResNet18_Weights.DEFAULT
            model = tv_models.resnet18(weights=weights)
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            self.model = model
            self.target_conv_layer = "layer4"

        self.model.eval()
        self.model.to(self.device)

    def extract_features(self, tensors: torch.Tensor, batch_size: int = 32) -> np.ndarray:
        """
        Extracts pooled feature embeddings for a batch of images.
        """
        self.model.eval()
        embeddings = []
        n_samples = len(tensors)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = tensors[i:i + batch_size].to(self.device)
                feats = self.model(batch)
                embeddings.append(feats.cpu().numpy())

        return np.vstack(embeddings)


class MedicalImageClassifier:
    """
    Combined Medical Image Classifier supporting:
    - Backbone feature extraction
    - High-performance regularized head (Logistic Regression with balanced class weights)
    - Temperature scaling calibration
    """
    def __init__(self,
                 backbone_name: str = "resnet18",
                 head_type: str = "lr",
                 random_state: int = 42):
        self.backbone_name = backbone_name
        self.head_type = head_type
        self.random_state = random_state
        self.feature_extractor = MedicalFeatureExtractor(backbone_name)
        self.scaler = TemperatureScaler()
        self.classes_ = None

        if head_type == "gbdt":
            self.head = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=random_state)
        else:
            self.head = LogisticRegression(class_weight='balanced', max_iter=500, random_state=random_state)

    def fit(self, X_train_tensors: torch.Tensor, y_train: np.ndarray,
            X_val_tensors: torch.Tensor, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Fits classification head on training features and fits temperature calibration on validation.
        """
        self.classes_ = np.unique(y_train)
        
        # 1. Extract training features
        train_features = self.feature_extractor.extract_features(X_train_tensors)
        val_features = self.feature_extractor.extract_features(X_val_tensors)

        # 2. Fit head
        self.head.fit(train_features, y_train)

        # 3. Predict validation logits / raw scores
        if hasattr(self.head, "decision_function"):
            val_logits = self.head.decision_function(val_features)
        else:
            val_probs = self.head.predict_proba(val_features)
            val_logits = np.log(np.clip(val_probs, 1e-7, 1 - 1e-7))

        # 4. Fit temperature scaling on validation
        fitted_temp = self.scaler.fit(val_logits, y_val)

        return {
            "temperature": fitted_temp,
            "feature_dim": self.feature_extractor.feature_dim,
            "classes": list(self.classes_)
        }

    def predict_logits(self, tensors: torch.Tensor) -> np.ndarray:
        feats = self.feature_extractor.extract_features(tensors)
        if hasattr(self.head, "decision_function"):
            return self.head.decision_function(feats)
        else:
            probs = self.head.predict_proba(feats)
            return np.log(np.clip(probs, 1e-7, 1 - 1e-7))

    def predict_proba(self, tensors: torch.Tensor, calibrated: bool = True) -> np.ndarray:
        logits = self.predict_logits(tensors)
        if calibrated:
            return self.scaler.scale_probabilities(logits)
        else:
            if logits.ndim == 1:
                p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                return np.vstack([1.0 - p, p]).T
            else:
                exp_s = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                return exp_s / np.sum(exp_s, axis=1, keepdims=True)

    def predict(self, tensors: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(tensors, calibrated=True)
        if len(self.classes_) == 2:
            return (probs[:, 1] >= threshold).astype(int)
        else:
            return np.argmax(probs, axis=1)
