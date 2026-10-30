"""
Standardized Medical Preprocessing Framework.
Implements missing value imputation (KNN/Iterative), outlier detection (Isolation Forest),
and robust scaling without data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import RobustScaler, StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.mixture import GaussianMixture

class MedicalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leakage-free medical tabular preprocessor supporting KNN/MICE/MissForest imputation,
    outlier handling (deletion/winsorization), clinical feature engineering, GMM bimodal scaling,
    and robust rescaling.
    """
    def __init__(
        self,
        numerical_cols: List[str],
        categorical_cols: List[str],
        zero_as_missing: Optional[List[str]] = None,
        imputation_strategy: str = "knn",
        n_neighbors: int = 5,
        scaling_strategy: str = "robust",
        outlier_contamination: float = 0.04,
        outlier_strategy: str = "delete",  # "delete" or "winsorize"
        use_power_transform: bool = False,
        disease: Optional[str] = None,
        bimodal_cols: Optional[List[str]] = None,
        add_missing_indicators: bool = False,
        add_clinical_flags: bool = False
    ):
        self.numerical_cols = list(numerical_cols)
        self.raw_numerical_cols = list(numerical_cols)
        self.categorical_cols = list(categorical_cols)
        self.zero_as_missing = zero_as_missing or []
        self.imputation_strategy = imputation_strategy
        self.n_neighbors = n_neighbors
        self.scaling_strategy = scaling_strategy
        self.outlier_contamination = outlier_contamination
        self.outlier_strategy = outlier_strategy
        self.use_power_transform = use_power_transform
        self.disease = disease
        self.bimodal_cols = bimodal_cols or []
        self.add_missing_indicators = add_missing_indicators
        self.add_clinical_flags = add_clinical_flags

        # Initialize Imputers
        if self.imputation_strategy in ["mice", "iterative"]:
            self.num_imputer = IterativeImputer(estimator=BayesianRidge(), random_state=42)
        elif self.imputation_strategy == "missforest":
            self.num_imputer = IterativeImputer(
                estimator=ExtraTreesRegressor(n_estimators=20, random_state=42),
                random_state=42
            )
        elif self.imputation_strategy == "knn":
            self.num_imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            self.num_imputer = SimpleImputer(strategy="median")

        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        
        # Power Transformer (Yeo-Johnson)
        if self.use_power_transform:
            self.power_transformer = PowerTransformer(method='yeo-johnson')
            
        self.scaler = RobustScaler() if scaling_strategy == "robust" else StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        # GMM Scalers for bimodal columns
        self.gmm_scalers_ = {}
        # Winsorization bounds
        self.winsor_bounds_ = {}
        # Columns that had missingness indicator generated
        self.imputed_cols_with_indicators = []

    def _engineer_clinical_features(self, df_num: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs clinically validated domain features on imputed continuous columns.
        """
        df_num = df_num.copy()
        if self.disease == "diabetes":
            # 1. HOMA-IR Proxy = (Glucose * Insulin) / 405
            if "Glucose" in df_num.columns and "Insulin" in df_num.columns:
                df_num["HOMA_IR_Proxy"] = (df_num["Glucose"] * df_num["Insulin"]) / 405.0
            # 2. Metabolic Burden = BMI * Age
            if "BMI" in df_num.columns and "Age" in df_num.columns:
                df_num["Metabolic_Burden"] = df_num["BMI"] * df_num["Age"]
            # 3. Adiposity Ratio = SkinThickness / (BMI + 1e-5)
            if "SkinThickness" in df_num.columns and "BMI" in df_num.columns:
                df_num["Adiposity_Ratio"] = df_num["SkinThickness"] / (df_num["BMI"] + 1e-5)
            # 4. Glycemic Index = Glucose / (BMI + 1e-5)
            if "Glucose" in df_num.columns and "BMI" in df_num.columns:
                df_num["Glycemic_Index"] = df_num["Glucose"] / (df_num["BMI"] + 1e-5)
                
        elif self.disease == "liver":
            # 1. De Ritis Ratio = AST / ALT
            ast_col = "Aspartate_Aminotransferase"
            alt_col = "Alamine_Aminotransferase"
            if ast_col in df_num.columns and alt_col in df_num.columns:
                df_num["De_Ritis_Ratio"] = df_num[ast_col] / (df_num[alt_col] + 1e-5)
            # 2. Bilirubin Index = Direct_Bilirubin / Total_Bilirubin
            if "Direct_Bilirubin" in df_num.columns and "Total_Bilirubin" in df_num.columns:
                df_num["Bilirubin_Index"] = df_num["Direct_Bilirubin"] / (df_num["Total_Bilirubin"] + 1e-5)
            # 3. Globulin = Total_Protiens - Albumin
            if "Total_Protiens" in df_num.columns and "Albumin" in df_num.columns:
                df_num["Globulin"] = df_num["Total_Protiens"] - df_num["Albumin"]
            # 4. AG Discrepancy = Albumin / Globulin - Albumin_and_Globulin_Ratio
            if "Albumin" in df_num.columns and "Globulin" in df_num.columns and "Albumin_and_Globulin_Ratio" in df_num.columns:
                df_num["AG_Discrepancy"] = df_num["Albumin"] / (df_num["Globulin"] + 1e-5) - df_num["Albumin_and_Globulin_Ratio"]
                
        return df_num

    def _add_clinical_flags(self, df_num: pd.DataFrame) -> pd.DataFrame:
        df_num = df_num.copy()
        if self.disease == "diabetes":
            if "Glucose" in df_num.columns:
                df_num["is_hyperglycemic"] = (df_num["Glucose"] >= 126.0).astype(float)
            if "BMI" in df_num.columns:
                df_num["is_obese"] = (df_num["BMI"] >= 30.0).astype(float)
            if "BloodPressure" in df_num.columns:
                df_num["is_hypertensive"] = (df_num["BloodPressure"] >= 130.0).astype(float)
            if "Age" in df_num.columns:
                df_num["is_elderly"] = (df_num["Age"] >= 45.0).astype(float)
        elif self.disease == "liver":
            if "Aspartate_Aminotransferase" in df_num.columns:
                df_num["is_abnormal_AST"] = (df_num["Aspartate_Aminotransferase"] >= 40.0).astype(float)
            if "Alamine_Aminotransferase" in df_num.columns:
                df_num["is_abnormal_ALT"] = (df_num["Alamine_Aminotransferase"] >= 45.0).astype(float)
            if "Total_Bilirubin" in df_num.columns:
                df_num["is_abnormal_Bilirubin"] = (df_num["Total_Bilirubin"] >= 1.2).astype(float)
            if "Age" in df_num.columns:
                df_num["is_elderly"] = (df_num["Age"] >= 50.0).astype(float)
        return df_num

    def fit_transform_train(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, pd.Series]:
        """
        Leakage-free training fit: Performs imputation, applies winsorization or deletion,
        skewness power transforms, robust scaling, and encodes categorical columns.
        """
        X = X.copy()
        y = y.copy()

        # 1. Flag invalid physiological zeros as NaN
        for col in self.zero_as_missing:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

        # 2. Extract missingness indicator flags before imputation
        if self.add_missing_indicators:
            self.imputed_cols_with_indicators = []
            for col in self.raw_numerical_cols:
                if col in X.columns:
                    if col in self.zero_as_missing or X[col].isna().any():
                        self.imputed_cols_with_indicators.append(col)

        # 3. Fit and Transform Imputer
        if self.raw_numerical_cols:
            num_data = X[self.raw_numerical_cols]
            self.num_imputer.fit(num_data)
            num_imputed = self.num_imputer.transform(num_data)
            df_num_imputed = pd.DataFrame(num_imputed, columns=self.raw_numerical_cols)
        else:
            df_num_imputed = pd.DataFrame()

        # 4. Outlier Handling / consensus pruning
        if self.raw_numerical_cols and len(X) > 10:
            if self.outlier_strategy == "delete":
                iforest = IsolationForest(contamination=self.outlier_contamination, random_state=42)
                lof = LocalOutlierFactor(n_neighbors=min(20, len(df_num_imputed) - 1), contamination=self.outlier_contamination)
                
                iforest_preds = iforest.fit_predict(df_num_imputed.values)
                lof_preds = lof.fit_predict(df_num_imputed.values)
                
                is_outlier = (iforest_preds == -1) & (lof_preds == -1)
                keep_mask = ~is_outlier
                
                X = X.loc[keep_mask].reset_index(drop=True)
                y = y.loc[keep_mask].reset_index(drop=True)
                
                # Re-impute pruned training dataset to maintain proper statistical priors
                if self.raw_numerical_cols:
                    num_data = X[self.raw_numerical_cols]
                    self.num_imputer.fit(num_data)
                    num_imputed = self.num_imputer.transform(num_data)
                    df_num_imputed = pd.DataFrame(num_imputed, columns=self.raw_numerical_cols)
            elif self.outlier_strategy == "winsorize":
                # Winsorization of outliers (Clipping to 1st and 99th percentiles)
                for col in self.raw_numerical_cols:
                    q_lower = df_num_imputed[col].quantile(0.01)
                    q_upper = df_num_imputed[col].quantile(0.99)
                    self.winsor_bounds_[col] = (q_lower, q_upper)
                    df_num_imputed[col] = df_num_imputed[col].clip(lower=q_lower, upper=q_upper)

        # 5. Fit bimodal GMM mixture scaling
        for col in self.bimodal_cols:
            if col in df_num_imputed.columns:
                gmm = GaussianMixture(n_components=2, random_state=42)
                gmm.fit(df_num_imputed[[col]])
                self.gmm_scalers_[col] = (gmm, np.argmax(gmm.means_.flatten()))
                # Transform column to GMM probability
                probs = gmm.predict_proba(df_num_imputed[[col]])
                df_num_imputed[f"{col}_gmm_prob"] = probs[:, self.gmm_scalers_[col][1]]

        # 6. Feature Engineering & Clinical Flags
        if self.raw_numerical_cols:
            df_num_engineered = self._engineer_clinical_features(df_num_imputed)
            if self.add_clinical_flags:
                df_num_engineered = self._add_clinical_flags(df_num_engineered)

            # Append missingness indicators
            if self.add_missing_indicators:
                for col in self.imputed_cols_with_indicators:
                    if col in X.columns:
                        # 1.0 if was originally missing (represented by NaN or zero if in zero_as_missing)
                        indicator = X[col].isna() | (X[col] == 0 if col in self.zero_as_missing else False)
                        df_num_engineered[f"{col}_is_missing"] = indicator.values.astype(float)

            self.numerical_cols = list(df_num_engineered.columns)
            
            # Skewness correction (Yeo-Johnson)
            if self.use_power_transform:
                self.power_transformer.fit(df_num_engineered)
                num_transformed = self.power_transformer.transform(df_num_engineered)
            else:
                num_transformed = df_num_engineered.values
                
            self.scaler.fit(num_transformed)
            num_scaled = self.scaler.transform(num_transformed)
            processed_parts = [num_scaled]
        else:
            processed_parts = []

        # 7. Fit Categorical Imputer & Encoder
        if self.categorical_cols:
            cat_data = X[self.categorical_cols]
            self.cat_imputer.fit(cat_data)
            cat_imputed = self.cat_imputer.transform(cat_data)
            self.ohe.fit(cat_imputed)
            cat_encoded = self.ohe.transform(cat_imputed)
            processed_parts.append(cat_encoded)

        return np.hstack(processed_parts), y

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Standard fit method.
        """
        X = X.copy()
        for col in self.zero_as_missing:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

        if self.add_missing_indicators:
            self.imputed_cols_with_indicators = []
            for col in self.raw_numerical_cols:
                if col in X.columns:
                    if col in self.zero_as_missing or X[col].isna().any():
                        self.imputed_cols_with_indicators.append(col)

        if self.raw_numerical_cols:
            num_data = X[self.raw_numerical_cols]
            self.num_imputer.fit(num_data)
            num_imputed = self.num_imputer.transform(num_data)
            df_num_imputed = pd.DataFrame(num_imputed, columns=self.raw_numerical_cols)

            if self.outlier_strategy == "winsorize":
                for col in self.raw_numerical_cols:
                    q_lower = df_num_imputed[col].quantile(0.01)
                    q_upper = df_num_imputed[col].quantile(0.99)
                    self.winsor_bounds_[col] = (q_lower, q_upper)
                    df_num_imputed[col] = df_num_imputed[col].clip(lower=q_lower, upper=q_upper)

            for col in self.bimodal_cols:
                if col in df_num_imputed.columns:
                    gmm = GaussianMixture(n_components=2, random_state=42)
                    gmm.fit(df_num_imputed[[col]])
                    self.gmm_scalers_[col] = (gmm, np.argmax(gmm.means_.flatten()))
                    probs = gmm.predict_proba(df_num_imputed[[col]])
                    df_num_imputed[f"{col}_gmm_prob"] = probs[:, self.gmm_scalers_[col][1]]

            df_num_engineered = self._engineer_clinical_features(df_num_imputed)
            if self.add_clinical_flags:
                df_num_engineered = self._add_clinical_flags(df_num_engineered)

            if self.add_missing_indicators:
                for col in self.imputed_cols_with_indicators:
                    if col in X.columns:
                        indicator = X[col].isna() | (X[col] == 0 if col in self.zero_as_missing else False)
                        df_num_engineered[f"{col}_is_missing"] = indicator.values.astype(float)

            self.numerical_cols = list(df_num_engineered.columns)
            
            if self.use_power_transform:
                self.power_transformer.fit(df_num_engineered)
                num_transformed = self.power_transformer.transform(df_num_engineered)
            else:
                num_transformed = df_num_engineered.values
            self.scaler.fit(num_transformed)

        if self.categorical_cols:
            cat_data = X[self.categorical_cols]
            self.cat_imputer.fit(cat_data)
            cat_imputed = self.cat_imputer.transform(cat_data)
            self.ohe.fit(cat_imputed)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms unseen evaluation or testing partitions.
        """
        X = X.copy()
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.raw_numerical_cols + self.categorical_cols)

        # Replace invalid zeros
        for col in self.zero_as_missing:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)

        processed_parts = []

        # Process Continuous Attributes
        if self.raw_numerical_cols:
            num_imputed = self.num_imputer.transform(X[self.raw_numerical_cols])
            df_num_imputed = pd.DataFrame(num_imputed, columns=self.raw_numerical_cols)
            
            if self.outlier_strategy == "winsorize":
                for col in self.raw_numerical_cols:
                    if col in self.winsor_bounds_:
                        q_lower, q_upper = self.winsor_bounds_[col]
                        df_num_imputed[col] = df_num_imputed[col].clip(lower=q_lower, upper=q_upper)

            for col in self.bimodal_cols:
                if col in df_num_imputed.columns and col in self.gmm_scalers_:
                    gmm, idx = self.gmm_scalers_[col]
                    probs = gmm.predict_proba(df_num_imputed[[col]])
                    df_num_imputed[f"{col}_gmm_prob"] = probs[:, idx]

            df_num_engineered = self._engineer_clinical_features(df_num_imputed)
            if self.add_clinical_flags:
                df_num_engineered = self._add_clinical_flags(df_num_engineered)

            if self.add_missing_indicators:
                for col in self.imputed_cols_with_indicators:
                    if col in X.columns:
                        indicator = X[col].isna() | (X[col] == 0 if col in self.zero_as_missing else False)
                        df_num_engineered[f"{col}_is_missing"] = indicator.values.astype(float)
            
            if self.use_power_transform:
                num_transformed = self.power_transformer.transform(df_num_engineered)
            else:
                num_transformed = df_num_engineered.values
                
            num_scaled = self.scaler.transform(num_transformed)
            processed_parts.append(num_scaled)

        # Process Categorical Attributes
        if self.categorical_cols:
            cat_imputed = self.cat_imputer.transform(X[self.categorical_cols])
            cat_encoded = self.ohe.transform(cat_imputed)
            processed_parts.append(cat_encoded)

        return np.hstack(processed_parts)

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "raw_numerical_cols" not in self.__dict__:
            self.raw_numerical_cols = list(self.numerical_cols)
        if "use_power_transform" not in self.__dict__:
            self.use_power_transform = False
        if "disease" not in self.__dict__:
            self.disease = None
