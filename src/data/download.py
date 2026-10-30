"""
Data download module for DS-HTE framework.
Automates acquisition of all 5 clinical datasets:
- PIMA Indian Diabetes (UCI/NIDDK)
- Cleveland Heart Disease (UCI)
- Chronic Kidney Disease (UCI CKD)
- Indian Liver Patient Dataset (ILPD, UCI)
- Wisconsin Breast Cancer Dataset (WBCD, sklearn / UCI)
"""

import os
import urllib.request
import pandas as pd
from sklearn.datasets import load_breast_cancer


def download_all_datasets(raw_dir: str = "data/raw"):
    os.makedirs(raw_dir, exist_ok=True)
    
    # 1. PIMA
    pima_path = os.path.join(raw_dir, "pima_diabetes_raw.csv")
    if not os.path.exists(pima_path):
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        try:
            cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                    'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
            df = pd.read_csv(url, header=None, names=cols)
            df.to_csv(pima_path, index=False)
            print(f"Downloaded PIMA -> {pima_path}")
        except Exception as e:
            print(f"PIMA download warning: {e}")

    # 2. Cleveland Heart
    cleveland_path = os.path.join(raw_dir, "cleveland_heart_raw.csv")
    if not os.path.exists(cleveland_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        try:
            cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
            df = pd.read_csv(url, header=None, names=cols, na_values="?")
            df.to_csv(cleveland_path, index=False)
            print(f"Downloaded Cleveland -> {cleveland_path}")
        except Exception as e:
            print(f"Cleveland download warning: {e}")

    # 3. CKD
    ckd_path = os.path.join(raw_dir, "ckd_kidney_raw.csv")
    if not os.path.exists(ckd_path):
        url = "https://archive.ics.uci.edu/static/public/338/chronic_kidney_disease.zip"
        # If not present, we will rely on existing file or fetch from openml
        try:
            from sklearn.datasets import fetch_openml
            ckd_data = fetch_openml("chronic_kidney_disease", version=1, as_frame=True)
            df = ckd_data.frame
            df.to_csv(ckd_path, index=False)
            print(f"Fetched CKD via OpenML -> {ckd_path}")
        except Exception as e:
            print(f"CKD download warning: {e}")

    # 4. ILPD
    ilpd_path = os.path.join(raw_dir, "ilpd_liver_raw.csv")
    if not os.path.exists(ilpd_path):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
        try:
            cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
                    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
                    'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset']
            df = pd.read_csv(url, header=None, names=cols)
            df.to_csv(ilpd_path, index=False)
            print(f"Downloaded ILPD -> {ilpd_path}")
        except Exception as e:
            print(f"ILPD download warning: {e}")

    # 5. WBCD
    wbcd_path = os.path.join(raw_dir, "wbcd_breast_cancer_raw.csv")
    if not os.path.exists(wbcd_path):
        cancer = load_breast_cancer(as_frame=True)
        df = cancer.data.copy()
        # In sklearn, 0=malignant, 1=benign. Map to diagnosis 'M' and 'B'
        df.insert(0, 'diagnosis', ['M' if t == 0 else 'B' for t in cancer.target])
        df.to_csv(wbcd_path, index=False)
        print(f"Generated WBCD from sklearn -> {wbcd_path}")


if __name__ == "__main__":
    download_all_datasets()
