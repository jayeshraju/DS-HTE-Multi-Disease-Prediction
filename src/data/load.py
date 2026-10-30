"""
Dataset loading and initial parsing for all 5 clinical cohorts.
Ensures uniform interface: (X, y, numerical_cols, categorical_cols).
Applies paper-specified data cleansing and invalid-value handling.
"""

import os
from typing import Tuple, List
import numpy as np
import pandas as pd


def load_pima(raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    path = os.path.join(raw_dir, "pima_diabetes_raw.csv")
    df = pd.read_csv(path)
    
    # Biologically impossible zeros in glucose, BP, skin thickness, insulin, BMI become missing
    zero_invalid_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_invalid_cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            
    y = df['Outcome'].astype(int)
    X = df.drop(columns=['Outcome'])
    numerical_cols = list(X.columns)
    categorical_cols = []
    return X, y, numerical_cols, categorical_cols


def load_cleveland(raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    path = os.path.join(raw_dir, "cleveland_heart_raw.csv")
    df = pd.read_csv(path)
    
    # Missing marked with ?
    df = df.replace('?', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    target_col = 'target' if 'target' in df.columns else 'num'
    y = (df[target_col] > 0).astype(int)
    X = df.drop(columns=[target_col])
    
    categorical_cols = [c for c in ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'] if c in X.columns]
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    return X, y, numerical_cols, categorical_cols


def load_ckd(raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    path = os.path.join(raw_dir, "ckd_kidney_raw.csv")
    df = pd.read_csv(path)
    
    # Irrelevant attribute removal: drop id
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Clean whitespace and tab typos in all categorical / text columns
    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].iloc[0], str):
            df[col] = df[col].astype(str).str.strip().str.replace('\t', '')
            df[col] = df[col].replace({'?': np.nan, 'nan': np.nan, '': np.nan})
            
    # Coerce known numerical columns
    numeric_candidates = ['age', 'bp', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # Target
    target_col = 'classification'
    df[target_col] = df[target_col].str.lower()
    y = df[target_col].map({'ckd': 1, 'notckd': 0}).astype(int)
    X = df.drop(columns=[target_col])
    
    numerical_cols = [c for c in numeric_candidates if c in X.columns]
    categorical_cols = [c for c in X.columns if c not in numerical_cols]
    return X, y, numerical_cols, categorical_cols


def load_ilpd(raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    path = os.path.join(raw_dir, "ilpd_liver_raw.csv")
    df = pd.read_csv(path)
    
    # Encode gender
    df['Gender'] = df['Gender'].astype(str).str.strip().map({'Male': 1, 'Female': 0, '1': 1, '0': 0})
    
    # Map labels {1, 2} to {1, 0}
    target_col = 'Dataset' if 'Dataset' in df.columns else df.columns[-1]
    y = df[target_col].map({1: 1, 2: 0, '1': 1, '2': 0}).astype(int)
    X = df.drop(columns=[target_col])
    
    categorical_cols = ['Gender'] if 'Gender' in X.columns else []
    numerical_cols = [c for c in X.columns if c not in categorical_cols]
    return X, y, numerical_cols, categorical_cols


def load_wbcd(raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    path = os.path.join(raw_dir, "wbcd_breast_cancer_raw.csv")
    df = pd.read_csv(path)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Diagnosis: 'M' -> 1 (disease), 'B' -> 0 (benign)
    target_col = 'diagnosis'
    if target_col in df.columns:
        y = df[target_col].map({'M': 1, 'B': 0, 1: 1, 0: 0}).astype(int)
        X = df.drop(columns=[target_col])
    else:
        y = df.iloc[:, 0].astype(int)
        X = df.iloc[:, 1:]
        
    # Unnamed trailing columns if any
    X = X.loc[:, ~X.columns.str.contains('^Unnamed')]
    numerical_cols = list(X.columns)
    categorical_cols = []
    return X, y, numerical_cols, categorical_cols


def load_cohort(cohort: str, raw_dir: str = "data/raw") -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    cohort = cohort.lower()
    loaders = {
        'pima': load_pima,
        'cleveland': load_cleveland,
        'ckd': load_ckd,
        'ilpd': load_ilpd,
        'wbcd': load_wbcd
    }
    if cohort not in loaders:
        raise ValueError(f"Unknown cohort '{cohort}'. Supported: {list(loaders.keys())}")
    return loaders[cohort](raw_dir)
