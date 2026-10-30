"""
Probability calibration module.
Follows Section III-F:
- Platt scaling (logistic/sigmoid) or Isotonic regression
- Evaluated on un-resampled inner validation predictions
- Fits probability calibration map without leakage
"""

from typing import Optional
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


class ModelCalibrator(BaseEstimator):
    def __init__(self, method: str = "sigmoid"):
        self.method = method.lower()
        self.calibrator_ = None

    def fit(self, val_probs: np.ndarray, val_y: np.ndarray):
        """
        Fit calibration model on validation probabilities and true binary labels.
        val_probs: 1D array of predicted probabilities for class 1
        val_y: 1D array of true labels {0, 1}
        """
        val_probs = np.clip(val_probs, 1e-6, 1.0 - 1e-6)

        if self.method in ["sigmoid", "platt"]:
            # Logistic regression on log-odds
            logits = np.log(val_probs / (1.0 - val_probs)).reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs")
            clf.fit(logits, val_y)
            self.calibrator_ = clf
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(val_probs, val_y)
            self.calibrator_ = iso
        else:
            raise ValueError(f"Unknown calibration method: '{self.method}'. Use 'sigmoid' or 'isotonic'.")

        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Calibrate probabilities.
        probs: 1D or 2D array of predicted probabilities.
        Returns calibrated 1D probability for positive class.
        """
        if self.calibrator_ is None:
            return probs

        p = probs[:, 1] if probs.ndim == 2 else probs
        p = np.clip(p, 1e-6, 1.0 - 1e-6)

        if self.method in ["sigmoid", "platt"]:
            logits = np.log(p / (1.0 - p)).reshape(-1, 1)
            calibrated = self.calibrator_.predict_proba(logits)[:, 1]
        elif self.method == "isotonic":
            calibrated = self.calibrator_.predict(p)
        else:
            calibrated = p

        return np.clip(calibrated, 0.0, 1.0)


def choose_best_calibrator(val_probs: np.ndarray, val_y: np.ndarray) -> ModelCalibrator:
    """
    Selects Platt or Isotonic by lowest Brier score on validation predictions.
    Prefers Platt for small samples (N < 200).
    """
    if len(val_probs) < 200:
        cal = ModelCalibrator(method="sigmoid")
        cal.fit(val_probs, val_y)
        return cal

    c_sig = ModelCalibrator(method="sigmoid").fit(val_probs, val_y)
    c_iso = ModelCalibrator(method="isotonic").fit(val_probs, val_y)

    brier_sig = brier_score_loss(val_y, c_sig.calibrate(val_probs))
    brier_iso = brier_score_loss(val_y, c_iso.calibrate(val_probs))

    if brier_iso < brier_sig:
        return c_iso
    return c_sig
