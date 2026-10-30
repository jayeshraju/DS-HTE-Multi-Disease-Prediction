"""
Leakage-free preprocessing pipeline for clinical tabular data.
Follows Section III-B:
- Missing-value treatment (median/mode, KNN, MICE via IterativeImputer)
- Invalid-value handling
- Categorical encoding (one-hot or ordinal)
- Numerical standardization
Fitted strictly on training splits.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder


class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self,
                 numerical_cols: Optional[List[str]] = None,
                 categorical_cols: Optional[List[str]] = None,
                 impute_strategy: str = "median",
                 scale_numeric: bool = True,
                 categorical_encoding: str = "onehot",
                 random_state: int = 42):
        self.numerical_cols = list(numerical_cols) if numerical_cols else []
        self.categorical_cols = list(categorical_cols) if categorical_cols else []
        self.impute_strategy = impute_strategy
        self.scale_numeric = scale_numeric
        self.categorical_encoding = categorical_encoding
        self.random_state = random_state

        self.num_imputer_ = None
        self.num_scaler_ = None
        self.cat_imputer_ = None
        self.cat_encoder_ = None
        self.feature_names_out_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1. Numerical pipeline
        if self.numerical_cols:
            X_num = X[self.numerical_cols].astype(float)
            if self.impute_strategy == "median":
                self.num_imputer_ = SimpleImputer(strategy="median")
            elif self.impute_strategy == "knn":
                self.num_imputer_ = KNNImputer(n_neighbors=5)
            elif self.impute_strategy == "mice":
                self.num_imputer_ = IterativeImputer(random_state=self.random_state, max_iter=10)
            else:
                self.num_imputer_ = SimpleImputer(strategy="mean")

            X_num_imp = self.num_imputer_.fit_transform(X_num)
            if self.scale_numeric:
                self.num_scaler_ = StandardScaler()
                self.num_scaler_.fit(X_num_imp)

        # 2. Categorical pipeline
        if self.categorical_cols:
            X_cat = X[self.categorical_cols].astype(str)
            self.cat_imputer_ = SimpleImputer(strategy="most_frequent")
            X_cat_imp = self.cat_imputer_.fit_transform(X_cat)

            if self.categorical_encoding == "onehot":
                self.cat_encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                self.cat_encoder_.fit(X_cat_imp)
            elif self.categorical_encoding == "ordinal":
                self.cat_encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                self.cat_encoder_.fit(X_cat_imp)

        # Build feature names
        feature_names = []
        if self.numerical_cols:
            feature_names.extend(self.numerical_cols)
        if self.categorical_cols and self.cat_encoder_:
            if hasattr(self.cat_encoder_, "get_feature_names_out"):
                feature_names.extend(list(self.cat_encoder_.get_feature_names_out(self.categorical_cols)))
            else:
                feature_names.extend(self.categorical_cols)
        self.feature_names_out_ = feature_names

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        parts = []
        if self.numerical_cols:
            X_num = X[self.numerical_cols].astype(float)
            X_num_imp = self.num_imputer_.transform(X_num)
            if self.scale_numeric and self.num_scaler_:
                X_num_trans = self.num_scaler_.transform(X_num_imp)
            else:
                X_num_trans = X_num_imp
            parts.append(X_num_trans)

        if self.categorical_cols:
            X_cat = X[self.categorical_cols].astype(str)
            X_cat_imp = self.cat_imputer_.transform(X_cat)
            X_cat_trans = self.cat_encoder_.transform(X_cat_imp)
            parts.append(X_cat_trans)

        if not parts:
            return np.empty((len(X), 0))

        return np.hstack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
