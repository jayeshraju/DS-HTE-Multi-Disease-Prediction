import numpy as np
import pandas as pd
import pytest
from src.data.preprocessing import MedicalPreprocessor

def test_medical_preprocessor_basic():
    # Construct a sample dataset with missing values and outliers
    data = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 4.0, 100.0],
        "num2": [10.0, 20.0, 30.0, np.nan, 50.0],
        "cat1": ["A", "B", "A", "B", "A"]
    })
    
    preprocessor = MedicalPreprocessor(
        numerical_cols=["num1", "num2"],
        categorical_cols=["cat1"],
        zero_as_missing=False,
        imputation_strategy="knn"
    )
    
    # Fit and transform
    X_trans = preprocessor.fit_transform(data)
    
    # Shape check
    # 2 numerical cols + 2 dummy columns for cat1 (A and B) = 4 columns
    assert X_trans.shape == (5, 4)
    assert not np.isnan(X_trans).any()

def test_clinical_feature_engineering_diabetes():
    data = pd.DataFrame({
        "Glucose": [120.0, 140.0],
        "Insulin": [80.0, 100.0],
        "BMI": [30.0, 35.0],
        "Age": [33, 45],
        "SkinThickness": [20.0, 25.0]
    })
    preprocessor = MedicalPreprocessor(
        numerical_cols=["Glucose", "Insulin", "BMI", "Age", "SkinThickness"],
        categorical_cols=[],
        disease="diabetes"
    )
    preprocessor.fit(data)
    X_trans = preprocessor.transform(data)
    
    # New features: HOMA_IR_Proxy, Metabolic_Burden, Adiposity_Ratio, Glycemic_Index
    # Original 5 + 4 new = 9 features
    assert X_trans.shape[1] == 9
    assert "HOMA_IR_Proxy" in preprocessor.numerical_cols

def test_clinical_feature_engineering_liver():
    data = pd.DataFrame({
        "Aspartate_Aminotransferase": [40.0, 45.0],
        "Alamine_Aminotransferase": [30.0, 35.0],
        "Direct_Bilirubin": [0.3, 0.4],
        "Total_Bilirubin": [1.0, 1.2],
        "Total_Protiens": [6.5, 7.0],
        "Albumin": [3.5, 4.0],
        "Albumin_and_Globulin_Ratio": [1.1, 1.2]
    })
    preprocessor = MedicalPreprocessor(
        numerical_cols=[
            "Aspartate_Aminotransferase", "Alamine_Aminotransferase",
            "Direct_Bilirubin", "Total_Bilirubin",
            "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"
        ],
        categorical_cols=[],
        disease="liver"
    )
    preprocessor.fit(data)
    X_trans = preprocessor.transform(data)
    
    # New features: De_Ritis_Ratio, Bilirubin_Index, Globulin, AG_Discrepancy
    # Original 7 + 4 new = 11 features
    assert X_trans.shape[1] == 11
    assert "De_Ritis_Ratio" in preprocessor.numerical_cols

def test_outlier_filtering_train():
    # Make a larger dataset to trigger Isolation Forest and LOF
    np.random.seed(42)
    n_samples = 50
    data = pd.DataFrame({
        "num1": np.random.normal(0, 1, n_samples).tolist() + [20.0, 30.0], # Add outliers
        "num2": np.random.normal(0, 1, n_samples).tolist() + [25.0, 35.0]
    })
    y = pd.Series([0] * n_samples + [1, 1])
    
    preprocessor = MedicalPreprocessor(
        numerical_cols=["num1", "num2"],
        categorical_cols=[],
        outlier_contamination=0.1
    )
    
    X_trans, y_clean = preprocessor.fit_transform_train(data, y)
    
    # We should have fewer than n_samples + 2 rows because outliers were dropped
    assert len(y_clean) < n_samples + 2
    assert X_trans.shape[0] == len(y_clean)
