"""
Out-of-Distribution (OOD) Safety Layer and Uncertainty Estimator.
Follows Section 2.3:
- Feature-space Mahalanobis distance calibrated on validation embeddings.
- Predicts normalized uncertainty score based on predictive entropy and feature distance.
- Flags samples exceeding 95th percentile validation threshold as OOD ("Needs Expert Review").
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class MahalanobisOODDetector:
    """
    Fits class-conditional Gaussian centroids and pooled covariance on backbone feature embeddings.
    Computes Mahalanobis distance to nearest class cluster.
    """
    def __init__(self, reg_eps: float = 1e-4):
        self.reg_eps = reg_eps
        self.class_means_ = {}
        self.precision_ = None
        self.threshold_ = None
        self.val_distances_ = []

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MahalanobisOODDetector":
        """
        Computes class-conditional means and pooled tied covariance precision matrix.
        """
        features = np.asarray(features, dtype=float)
        labels = np.asarray(labels)
        unique_classes = np.unique(labels)
        n_samples, n_dim = features.shape

        # Class centroids
        diffs = []
        for c in unique_classes:
            idx = np.where(labels == c)[0]
            mean_c = np.mean(features[idx], axis=0)
            self.class_means_[c] = mean_c
            diffs.append(features[idx] - mean_c)

        # Pooled covariance
        all_diffs = np.vstack(diffs)
        cov = np.dot(all_diffs.T, all_diffs) / max(1, n_samples)
        # Regularization for numerical stability
        cov += np.eye(n_dim) * self.reg_eps

        # Precision matrix (inverse covariance)
        try:
            self.precision_ = np.linalg.pinv(cov)
        except Exception:
            self.precision_ = np.eye(n_dim)

        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """
        Computes minimum Mahalanobis distance across class centroids.
        Distance D(z) = min_c sqrt((z - mu_c)^T Sigma^{-1} (z - mu_c))
        """
        features = np.asarray(features, dtype=float)
        if self.precision_ is None or not self.class_means_:
            return np.zeros(len(features))

        distances = []
        for z in features:
            min_dist = float("inf")
            for c, mu_c in self.class_means_.items():
                diff = z - mu_c
                # Mahalanobis quadratic form
                quad = np.dot(np.dot(diff, self.precision_), diff)
                dist = np.sqrt(max(0.0, quad))
                if dist < min_dist:
                    min_dist = dist
            distances.append(min_dist)

        return np.array(distances)

    def calibrate_threshold(self, val_features: np.ndarray, percentile: float = 95.0) -> float:
        """
        Sets OOD detection threshold at declared percentile of validation in-distribution distances.
        """
        self.val_distances_ = self.score_samples(val_features)
        self.threshold_ = float(np.percentile(self.val_distances_, percentile))
        return self.threshold_

    def evaluate_sample(self, feature: np.ndarray, prob_distribution: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluates a single sample:
        Returns distance, threshold, is_ood flag, predictive entropy, and clinical safety banner.
        """
        if feature.ndim == 1:
            feature = feature.reshape(1, -1)
        dist = float(self.score_samples(feature)[0])

        is_ood = False
        if self.threshold_ is not None:
            is_ood = bool(dist > self.threshold_)

        # Predictive entropy uncertainty
        entropy = 0.0
        normalized_uncertainty = 0.0
        if prob_distribution is not None:
            p = np.clip(prob_distribution, 1e-7, 1.0)
            n_classes = len(p)
            entropy = float(-np.sum(p * np.log2(p)))
            max_entropy = np.log2(max(2, n_classes))
            normalized_uncertainty = float(min(1.0, max(0.0, entropy / max_entropy)))

        # Combined decision
        needs_expert_review = is_ood or (normalized_uncertainty > 0.85)

        banner = None
        if needs_expert_review:
            if is_ood:
                banner = "Out-of-Distribution Warning: Image features deviate significantly from training cohort. Needs Expert Review."
            else:
                banner = "High Clinical Uncertainty: Low model confidence. Needs Expert Review."

        return {
            "mahalanobis_distance": round(dist, 4),
            "ood_threshold": round(self.threshold_, 4) if self.threshold_ is not None else None,
            "is_ood": is_ood,
            "predictive_entropy": round(entropy, 4),
            "uncertainty_score": round(normalized_uncertainty, 4),
            "needs_expert_review": needs_expert_review,
            "safety_banner": banner
        }
