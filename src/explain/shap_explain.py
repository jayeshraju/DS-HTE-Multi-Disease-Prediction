"""
SHAP explainability module.
Follows Section III-G:
- Global feature importance
- Per-patient local risk attribution
- Summary data generation
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import shap


def compute_shap_explanations(model, X_train: np.ndarray, X_test: np.ndarray,
                              feature_names: List[str],
                              n_background: int = 30) -> Dict[str, Any]:
    """
    Computes KernelExplainer SHAP values for DS-HTE ensemble or tree model.
    Uses kmeans/sample background for computational tractability.
    """
    # Sample background
    if len(X_train) > n_background:
        rng = np.random.RandomState(42)
        bg_idx = rng.choice(len(X_train), size=n_background, replace=False)
        background = X_train[bg_idx]
    else:
        background = X_train

    # Prediction function for class 1 probability
    def predict_p1(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_p1, background)
    
    # Explain test samples (up to 50 for speed)
    test_subset = X_test[:min(50, len(X_test))]
    shap_values = explainer.shap_values(test_subset, nsamples=100)

    # Global importance = mean absolute SHAP value
    global_importance = np.mean(np.abs(shap_values), axis=0)
    top_indices = np.argsort(-global_importance)

    ranked_features = [
        {
            "feature": feature_names[i],
            "importance": float(global_importance[i]),
            "rank": r + 1
        }
        for r, i in enumerate(top_indices)
    ]

    return {
        "ranked_features": ranked_features,
        "global_importance": {feature_names[i]: float(global_importance[i]) for i in range(len(feature_names))},
        "shap_values": shap_values,
        "test_subset": test_subset,
        "feature_names": feature_names
    }
