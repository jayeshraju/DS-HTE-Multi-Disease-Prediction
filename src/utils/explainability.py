"""
Explainable Artificial Intelligence (XAI) Interpretation Module.
Wraps SHAP (TreeSHAP / KernelSHAP) and LIME explainer routines for patient risk attribution.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


class ClinicalExplainer:
    """
    Unified Explainable AI wrapper producing local patient attributions and global feature rankings.
    """
    def __init__(self, model: Any, feature_names: List[str], model_type: str = "tree"):
        self.model = model
        self.feature_names = feature_names
        self.model_type = model_type
        self.shap_explainer = None
        self.lime_explainer = None

    def fit_explainers(self, X_train: np.ndarray):
        """
        Initializes SHAP and LIME background explainer states on training data.
        """
        if SHAP_AVAILABLE:
            if self.model_type == "tree":
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.KernelExplainer(self.model.predict_proba, shap.sample(X_train, 50))

        if LIME_AVAILABLE:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=X_train,
                feature_names=self.feature_names,
                class_names=["Healthy", "Pathological"],
                mode="classification"
            )

    def explain_patient_instance(self, x_patient: np.ndarray) -> Dict[str, Any]:
        """
        Generates local patient risk attributions via SHAP values.
        """
        if x_patient.ndim == 1:
            x_patient = x_patient.reshape(1, -1)

        result = {}
        if self.shap_explainer is not None:
            shap_values = self.shap_explainer.shap_values(x_patient)
            
            if isinstance(shap_values, list):
                # Old shape format: list of length n_classes
                shap_vals = shap_values[1]
                if shap_vals.ndim == 2:
                    shap_vals = shap_vals[0]
            else:
                # New shape format: ndarray of shape (N, n_features, n_classes) or (N, n_features)
                if shap_values.ndim == 3:
                    # (N, n_features, n_classes) -> extract positive class (index 1)
                    shap_vals = shap_values[0, :, 1]
                elif shap_values.ndim == 2:
                    # (N, n_features)
                    shap_vals = shap_values[0]
                else:
                    shap_vals = shap_values

            result["shap_values"] = list(map(float, shap_vals))
            result["feature_attributions"] = {name: float(val) for name, val in zip(self.feature_names, shap_vals)}

        return result
