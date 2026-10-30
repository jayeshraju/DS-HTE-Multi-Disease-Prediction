"""
LIME explainability and SHAP-LIME agreement analysis.
Follows Section III-G:
- Instance-level local explanations
- Agreement analysis (Spearman rank correlation) between SHAP and LIME
"""

from typing import Dict, Any, List
import numpy as np
import scipy.stats as stats
import lime
import lime.lime_tabular


def compute_lime_explanations(model, X_train: np.ndarray, X_test: np.ndarray,
                              feature_names: List[str],
                              n_instances: int = 10) -> Dict[str, Any]:
    """
    Computes LIME explanations for sample instances.
    """
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=["Healthy", "Disease"],
        mode="classification",
        random_state=42
    )

    n_samples = min(n_instances, len(X_test))
    explanations = []

    for idx in range(n_samples):
        exp = explainer.explain_instance(
            X_test[idx],
            model.predict_proba,
            num_features=len(feature_names),
            num_samples=500
        )
        # Extract feature weights
        weights = dict(exp.as_list())
        explanations.append({
            "instance_idx": idx,
            "weights": weights,
            "prediction": model.predict_proba(X_test[idx:idx+1])[0].tolist()
        })

    return {
        "explanations": explanations,
        "feature_names": feature_names
    }


def compute_shap_lime_agreement(shap_global: Dict[str, float],
                                lime_explanations: List[Dict[str, Any]],
                                feature_names: List[str]) -> float:
    """
    Computes mean Spearman rank correlation between SHAP global importance
    and LIME feature importance.
    """
    # Aggregate LIME importance across instances
    lime_agg = {feat: 0.0 for feat in feature_names}
    count = 0
    for exp in lime_explanations:
        weights = exp["weights"]
        for k, v in weights.items():
            for feat in feature_names:
                if feat in k:
                    lime_agg[feat] += abs(v)
        count += 1

    if count > 0:
        for feat in lime_agg:
            lime_agg[feat] /= count

    # Vectorize
    shap_vec = [shap_global.get(f, 0.0) for f in feature_names]
    lime_vec = [lime_agg.get(f, 0.0) for f in feature_names]

    if np.all(np.array(shap_vec) == shap_vec[0]) or np.all(np.array(lime_vec) == lime_vec[0]):
        return 0.5

    rho, _ = stats.spearmanr(shap_vec, lime_vec)
    return float(rho if not np.isnan(rho) else 0.5)
