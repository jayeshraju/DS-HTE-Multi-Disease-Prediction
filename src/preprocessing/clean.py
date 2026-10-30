"""
Data cleaning and validation module.
Verifies data constraints and validity before model ingestion.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd


def validate_and_clean_inputs(X: pd.DataFrame, y: pd.Series,
                              numerical_cols: List[str],
                              categorical_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Validates input shapes, types, and enforces consistency.
    """
    assert len(X) == len(y), f"Mismatch: X has {len(X)} rows, y has {len(y)} rows"
    assert set(y.unique()).issubset({0, 1}), f"Target y must be binary {0, 1}, got {y.unique()}"
    
    # Ensure all numerical cols are numeric
    X_clean = X.copy()
    for col in numerical_cols:
        if col in X_clean.columns:
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
            
    # Categorical cols as object/string
    for col in categorical_cols:
        if col in X_clean.columns:
            X_clean[col] = X_clean[col].astype(str).replace({'nan': np.nan, 'None': np.nan})
            
    return X_clean, y
