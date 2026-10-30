"""
Global Configuration for Scalable Multi-Disease Prediction Framework.
Author: Multi-Disease AI Research Team
Context: IEEE Transactions / Scopus & Dissertation Implementation
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "saved_models"

# Create directories if they do not exist
for directory in [RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, DOCS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Global Reproducibility Seed
RANDOM_SEED = 42

# Target Disease Datasets Metadata
DATASET_CONFIGS = {
    "diabetes": {
        "name": "PIMA Indian Diabetes",
        "raw_file": RAW_DATA_DIR / "pima_diabetes_raw.csv",
        "url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "target": "Outcome",
        "zero_as_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "categorical_cols": [],
        "numerical_cols": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    },
    "heart": {
        "name": "Cleveland Heart Disease",
        "raw_file": RAW_DATA_DIR / "cleveland_heart_raw.csv",
        "url": "https://archive.ics.uci.edu/static/public/45/heart+disease.zip",
        "target": "target",
        "zero_as_missing": [],
        "categorical_cols": ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"],
        "numerical_cols": ["age", "trestbps", "chol", "thalach", "oldpeak"]
    },
    "kidney": {
        "name": "Chronic Kidney Disease",
        "raw_file": RAW_DATA_DIR / "ckd_kidney_raw.csv",
        "url": "https://raw.githubusercontent.com/mesfind/datasets/master/chronic_kidney_disease.csv",
        "target": "classification",
        "zero_as_missing": [],
        "categorical_cols": ["sg", "al", "su", "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"],
        "numerical_cols": ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]
    },
    "liver": {
        "name": "Indian Liver Patient Dataset (ILPD)",
        "raw_file": RAW_DATA_DIR / "ilpd_liver_raw.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        "target": "Dataset",
        "zero_as_missing": [],
        "categorical_cols": ["Gender"],
        "numerical_cols": ["Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio"]
    },
    "cancer": {
        "name": "Wisconsin Breast Cancer Diagnostic (WBCD)",
        "raw_file": RAW_DATA_DIR / "wbcd_breast_cancer_raw.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "target": "diagnosis",
        "zero_as_missing": [],
        "categorical_cols": [],
        "numerical_cols": [
            "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
            "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
            "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
            "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
            "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
            "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
        ]
    }
}
