"""
Data Ingestion and Synthetic Fallback Generator Module.
Downloads the benchmark datasets from remote repositories or generates synthetic clinical cohorts as a fallback.
"""

import os
import zipfile
import urllib.request
import io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.config import DATASET_CONFIGS, RAW_DATA_DIR, RANDOM_SEED

# Exact column orders of raw files as they are structured in remote repositories
RAW_COLUMNS = {
    "diabetes": ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"],
    "heart": ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"],
    "kidney": ["age", "bp", "sg", "al", "su", "rbc", "pc", "pcc", "ba", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc", "htn", "dm", "cad", "appet", "pe", "ane", "classification"],
    "liver": ["Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase", "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens", "Albumin", "Albumin_and_Globulin_Ratio", "Dataset"],
    "cancer": ["id", "diagnosis"] + [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
    ]
}

def parse_arff_to_dataframe(arff_text: str) -> pd.DataFrame:
    """
    Parses a standard ARFF file text into a Pandas DataFrame.
    """
    lines = arff_text.splitlines()
    columns = []
    data_started = False
    data_lines = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
        
        line_lower = line_str.lower()
        if line_lower.startswith("@attribute"):
            parts = line_str.split(None, 2)
            if len(parts) >= 2:
                col_name = parts[1].strip("'").strip('"')
                columns.append(col_name)
        elif line_lower.startswith("@data"):
            data_started = True
            continue
        elif data_started:
            if not line_str.startswith("@") and not line_str.startswith("%"):
                data_lines.append(line_str)

    parsed_rows = []
    for dline in data_lines:
        row_vals = [v.strip().strip("'").strip('"') for v in dline.split(",")]
        if len(row_vals) == len(columns):
            parsed_rows.append(row_vals)
        else:
            row_vals = row_vals[:len(columns)]
            parsed_rows.append(row_vals)

    df = pd.DataFrame(parsed_rows, columns=columns)
    df = df.replace("?", np.nan)
    return df

def generate_synthetic_cohorts() -> Dict[str, pd.DataFrame]:
    """
    Generates statistically plausible synthetic datasets for all 5 diseases
    to serve as a fallback when internet connection is unavailable.
    """
    np.random.seed(RANDOM_SEED)
    cohorts = {}

    # 1. Diabetes (PIMA)
    n_diabetes = 768
    glucose = np.random.normal(120, 30, n_diabetes)
    bmi = np.random.normal(32, 7, n_diabetes)
    age = np.random.normal(33, 11, n_diabetes).astype(int)
    bp = np.random.normal(69, 12, n_diabetes)
    skin = np.random.normal(20, 15, n_diabetes)
    insulin = np.random.normal(80, 115, n_diabetes)
    pregnancies = np.random.poisson(3.8, n_diabetes)
    pedigree = np.random.exponential(0.47, n_diabetes) + 0.08
    
    glucose = np.clip(glucose, 40, 200)
    bmi = np.clip(bmi, 18, 60)
    age = np.clip(age, 21, 81)
    bp = np.clip(bp, 30, 122)
    skin = np.clip(skin, 5, 99)
    insulin = np.clip(insulin, 15, 846)
    pregnancies = np.clip(pregnancies, 0, 17)
    pedigree = np.clip(pedigree, 0.08, 2.42)

    # Induce missingness zeros (PIMA format uses 0 as missing)
    glucose[np.random.rand(n_diabetes) < 0.05] = 0.0
    bp[np.random.rand(n_diabetes) < 0.05] = 0.0
    skin[np.random.rand(n_diabetes) < 0.30] = 0.0
    insulin[np.random.rand(n_diabetes) < 0.40] = 0.0
    bmi[np.random.rand(n_diabetes) < 0.02] = 0.0

    outcome_prob = 1.0 / (1.0 + np.exp(-(-6.0 + 0.035 * np.where(glucose==0, 100, glucose) + 0.08 * np.where(bmi==0, 25, bmi) + 0.02 * age)))
    outcome = (np.random.rand(n_diabetes) < outcome_prob).astype(int)

    df_dia = pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": bp,
        "SkinThickness": skin,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
        "Outcome": outcome
    })
    cohorts["diabetes"] = df_dia

    # 2. Heart Disease (Cleveland)
    n_heart = 303
    age_h = np.random.normal(54, 9, n_heart).astype(int)
    sex_h = np.random.choice([0, 1], size=n_heart, p=[0.32, 0.68])
    cp_h = np.random.choice([1.0, 2.0, 3.0, 4.0], size=n_heart, p=[0.08, 0.16, 0.28, 0.48])
    trestbps_h = np.random.normal(131, 17, n_heart)
    chol_h = np.random.normal(246, 51, n_heart)
    fbs_h = np.random.choice([0.0, 1.0], size=n_heart, p=[0.85, 0.15])
    restecg_h = np.random.choice([0.0, 1.0, 2.0], size=n_heart, p=[0.49, 0.02, 0.49])
    thalach_h = np.random.normal(149, 23, n_heart)
    exang_h = np.random.choice([0.0, 1.0], size=n_heart, p=[0.67, 0.33])
    oldpeak_h = np.random.exponential(1.0, n_heart)
    slope_h = np.random.choice([1.0, 2.0, 3.0], size=n_heart, p=[0.46, 0.46, 0.08])
    ca_h = np.random.choice(["0.0", "1.0", "2.0", "3.0", "?"], size=n_heart, p=[0.58, 0.22, 0.12, 0.06, 0.02])
    thal_h = np.random.choice(["3.0", "6.0", "7.0", "?"], size=n_heart, p=[0.54, 0.06, 0.38, 0.02])

    target_prob = 1.0 / (1.0 + np.exp(-(-4.0 + 0.1 * (cp_h == 4) + 0.8 * oldpeak_h - 0.02 * thalach_h + 1.2 * (exang_h == 1))))
    target = (np.random.rand(n_heart) < target_prob).astype(int)

    df_heart = pd.DataFrame({
        "age": age_h,
        "sex": sex_h,
        "cp": cp_h,
        "trestbps": trestbps_h,
        "chol": chol_h,
        "fbs": fbs_h,
        "restecg": restecg_h,
        "thalach": thalach_h,
        "exang": exang_h,
        "oldpeak": oldpeak_h,
        "slope": slope_h,
        "ca": ca_h,
        "thal": thal_h,
        "target": target
    })
    cohorts["heart"] = df_heart

    # 3. Kidney Disease (CKD) — clinically separable cohort
    n_ckd = 400
    n_ckd_pos  = int(n_ckd * 0.62)   # 62 % have CKD
    n_ckd_neg  = n_ckd - n_ckd_pos   # 38 % are healthy

    def _ckd_col(ckd_mean, notckd_mean, ckd_std, notckd_std):
        return np.concatenate([
            np.random.normal(ckd_mean,    ckd_std,    n_ckd_pos),
            np.random.normal(notckd_mean, notckd_std, n_ckd_neg)
        ])

    age_k    = _ckd_col(55, 42, 15, 14)
    bp_k     = _ckd_col(80, 70, 15, 10)
    bgr_k    = _ckd_col(170, 95, 90, 30)
    bu_k     = _ckd_col(90, 28, 60, 12)
    sc_k     = _ckd_col(6.5, 0.9, 5.0, 0.3)    # creatinine: CKD >> normal
    sod_k    = _ckd_col(130, 141, 12, 4)
    pot_k    = _ckd_col(5.2, 4.1, 1.5, 0.5)
    hemo_k   = _ckd_col(10.2, 14.8, 2.0, 1.5)  # anaemia in CKD
    pcv_k    = _ckd_col(30, 45, 7, 5)
    wc_k     = _ckd_col(9500, 7200, 3000, 2000)
    rc_k     = _ckd_col(3.5, 5.0, 0.8, 0.5)

    # clip to physiological limits
    sc_k  = np.clip(sc_k, 0.4, 76)
    bu_k  = np.clip(bu_k, 1.5, 391)
    hemo_k= np.clip(hemo_k, 3.1, 17.8)
    pcv_k = np.clip(pcv_k, 9, 54)
    wc_k  = np.clip(wc_k, 2200, 26400)
    rc_k  = np.clip(rc_k, 1.0, 8.0)
    sod_k = np.clip(sod_k, 111, 163)
    pot_k = np.clip(pot_k, 2.5, 47)

    def _ckd_cat(ckd_vals, notckd_vals, ckd_probs, notckd_probs):
        return np.concatenate([
            np.random.choice(ckd_vals,    n_ckd_pos, p=ckd_probs),
            np.random.choice(notckd_vals, n_ckd_neg, p=notckd_probs)
        ])

    sg_k  = _ckd_cat(["1.005","1.010","1.015","1.020","1.025"],
                     ["1.005","1.010","1.015","1.020","1.025"],
                     [0.20,0.30,0.25,0.15,0.10], [0.02,0.05,0.13,0.40,0.40])
    al_k  = _ckd_cat(["0","1","2","3","4","5"],
                     ["0","1","2","3","4","5"],
                     [0.15,0.15,0.20,0.20,0.15,0.15], [0.80,0.10,0.04,0.03,0.02,0.01])
    su_k  = _ckd_cat(["0","1","2","3","4","5"],
                     ["0","1","2","3","4","5"],
                     [0.40,0.15,0.15,0.10,0.10,0.10], [0.95,0.02,0.01,0.01,0.005,0.005])
    rbc_k = _ckd_cat(["normal","abnormal"], ["normal","abnormal"],
                     [0.35,0.65], [0.90,0.10])
    pc_k  = _ckd_cat(["normal","abnormal"], ["normal","abnormal"],
                     [0.40,0.60], [0.88,0.12])
    pcc_k = _ckd_cat(["present","notpresent"], ["present","notpresent"],
                     [0.30,0.70], [0.02,0.98])
    ba_k  = _ckd_cat(["present","notpresent"], ["present","notpresent"],
                     [0.20,0.80], [0.01,0.99])
    htn_k = _ckd_cat(["yes","no"], ["yes","no"],
                     [0.70,0.30], [0.15,0.85])
    dm_k  = _ckd_cat(["yes","no"], ["yes","no"],
                     [0.50,0.50], [0.15,0.85])
    cad_k = _ckd_cat(["yes","no"], ["yes","no"],
                     [0.20,0.80], [0.02,0.98])
    appet_k = _ckd_cat(["good","poor"], ["good","poor"],
                       [0.40,0.60], [0.95,0.05])
    pe_k  = _ckd_cat(["yes","no"], ["yes","no"],
                     [0.50,0.50], [0.02,0.98])
    ane_k = _ckd_cat(["yes","no"], ["yes","no"],
                     [0.55,0.45], [0.03,0.97])

    classification_k = np.array(["ckd"] * n_ckd_pos + ["notckd"] * n_ckd_neg)

    # Shuffle all together
    perm = np.random.permutation(n_ckd)
    df_ckd = pd.DataFrame({
        "age": age_k[perm], "bp": bp_k[perm], "sg": sg_k[perm], "al": al_k[perm],
        "su": su_k[perm], "rbc": rbc_k[perm], "pc": pc_k[perm],
        "pcc": pcc_k[perm], "ba": ba_k[perm],
        "bgr": bgr_k[perm], "bu": bu_k[perm], "sc": sc_k[perm],
        "sod": sod_k[perm], "pot": pot_k[perm], "hemo": hemo_k[perm],
        "pcv": pcv_k[perm], "wc": wc_k[perm], "rc": rc_k[perm],
        "htn": htn_k[perm], "dm": dm_k[perm], "cad": cad_k[perm],
        "appet": appet_k[perm], "pe": pe_k[perm], "ane": ane_k[perm],
        "classification": classification_k[perm]
    })

    # Introduce realistic missingness (5–10 % per numeric column)
    for col in ["age", "bp", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc"]:
        mask = np.random.rand(n_ckd) < 0.06
        df_ckd.loc[mask, col] = np.nan

    cohorts["kidney"] = df_ckd

    # 4. Liver Patient (ILPD)
    n_liver = 583
    age_l = np.random.normal(44, 16, n_liver).astype(int)
    gender_l = np.random.choice(["Male", "Female"], size=n_liver, p=[0.75, 0.25])
    tb_l = np.random.exponential(3.3, n_liver) + 0.1
    db_l = tb_l * np.random.uniform(0.3, 0.5, n_liver)
    alp_l = np.random.normal(290, 240, n_liver)
    alt_l = np.random.exponential(80, n_liver) + 10
    ast_l = alt_l * np.random.uniform(0.8, 1.5, n_liver)
    tp_l = np.random.normal(6.4, 1.0, n_liver)
    alb_l = np.random.normal(3.1, 0.8, n_liver)
    ag_ratio_l = alb_l / (tp_l - alb_l)

    tb_l = np.clip(tb_l, 0.4, 75.0)
    db_l = np.clip(db_l, 0.1, 19.7)
    alp_l = np.clip(alp_l, 63, 2110)
    alt_l = np.clip(alt_l, 10, 2000)
    ast_l = np.clip(ast_l, 10, 4929)
    tp_l = np.clip(tp_l, 2.7, 9.6)
    alb_l = np.clip(alb_l, 0.9, 5.5)
    ag_ratio_l = np.clip(ag_ratio_l, 0.3, 2.8)

    ag_ratio_l[np.random.choice(n_liver, 4, replace=False)] = np.nan

    liv_prob = 1.0 / (1.0 + np.exp(-(-2.0 + 0.5 * tb_l + 0.005 * alt_l - 0.2 * alb_l)))
    dataset_l = np.where(np.random.rand(n_liver) < liv_prob, 1, 2)

    df_liver = pd.DataFrame({
        "Age": age_l,
        "Gender": gender_l,
        "Total_Bilirubin": tb_l,
        "Direct_Bilirubin": db_l,
        "Alkaline_Phosphotase": alp_l,
        "Alamine_Aminotransferase": alt_l,
        "Aspartate_Aminotransferase": ast_l,
        "Total_Protiens": tp_l,
        "Albumin": alb_l,
        "Albumin_and_Globulin_Ratio": ag_ratio_l,
        "Dataset": dataset_l
    })
    cohorts["liver"] = df_liver

    # 5. Breast Cancer (WBCD)
    n_cancer = 569
    diag_c = np.random.choice(["M", "B"], size=n_cancer, p=[0.37, 0.63])
    
    feat_data = {}
    for col_name in DATASET_CONFIGS["cancer"]["numerical_cols"]:
        if "worst" in col_name or "mean" in col_name:
            mean_m = 18.0 if "radius" in col_name or "texture" in col_name else (100.0 if "perimeter" in col_name else (800.0 if "area" in col_name else 0.5))
            mean_b = 12.0 if "radius" in col_name or "texture" in col_name else (70.0 if "perimeter" in col_name else (400.0 if "area" in col_name else 0.2))
            std_m = mean_m * 0.2
            std_b = mean_b * 0.15
            vals = np.where(diag_c == "M", np.random.normal(mean_m, std_m, n_cancer), np.random.normal(mean_b, std_b, n_cancer))
        else:
            vals = np.random.exponential(0.1, n_cancer)
        feat_data[col_name] = np.clip(vals, 0.001, None)

    df_cancer = pd.DataFrame(feat_data)
    df_cancer.insert(0, "id", np.random.choice(range(100000, 999999), n_cancer, replace=False))
    df_cancer.insert(1, "diagnosis", diag_c)
    cohorts["cancer"] = df_cancer

    return cohorts

def ingest_datasets(force_rebuild: bool = True) -> Dict[str, pd.DataFrame]:
    """
    Orchestrates the data ingestion pipeline.
    If force_rebuild is True, deletes existing files and reconstructs them cleanly.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    datasets = {}
    synthetic_fallback = None

    for key, config in DATASET_CONFIGS.items():
        raw_file_path = Path(config["raw_file"])
        url = config["url"]
        
        print(f"\nIngesting {config['name']} dataset...")
        
        # Force delete if it's already there to enforce correct headers
        if force_rebuild and raw_file_path.exists():
            try:
                os.remove(raw_file_path)
                print(f" -> Removed old cache: {raw_file_path.name}")
            except Exception:
                pass

        # Check if already present locally (should not be present if force_rebuild is True)
        if not force_rebuild and raw_file_path.exists():
            print(f" -> Found cached raw file: {raw_file_path.name}")
            try:
                df = pd.read_csv(raw_file_path)
                datasets[key] = df
                print(f" -> Loaded successfully. Shape: {df.shape}")
                continue
            except Exception:
                pass

        # Attempt download
        download_success = False
        try:
            print(f" -> Downloading from: {url}")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                content = response.read()

            if url.endswith(".zip"):
                zip_file = zipfile.ZipFile(io.BytesIO(content))
                if key == "heart":
                    # Extract processed.cleveland.data
                    data_content = zip_file.read("processed.cleveland.data").decode("utf-8", errors="ignore")
                    col_names = RAW_COLUMNS[key]
                    df = pd.read_csv(io.StringIO(data_content), header=None, names=col_names, na_values=["?", "NaN", "nan"])
                    df.to_csv(raw_file_path, index=False)
                    download_success = True
                elif key == "kidney":
                    arff_file_name = [name for name in zip_file.namelist() if name.endswith(".arff")]
                    if arff_file_name:
                        print(f"   -> Found ARFF in zip: {arff_file_name[0]}")
                        arff_text = zip_file.read(arff_file_name[0]).decode("utf-8", errors="ignore")
                        df = parse_arff_to_dataframe(arff_text)
                        df.to_csv(raw_file_path, index=False)
                        download_success = True
                    else:
                        raise ValueError("No .arff file found in downloaded Chronic Kidney Disease zip.")
            else:
                # Text/CSV downloads
                text_content = content.decode("utf-8", errors="ignore")
                col_names = RAW_COLUMNS[key]
                df = pd.read_csv(io.StringIO(text_content), header=None, names=col_names, na_values=["?", "NaN", "nan"])
                df.to_csv(raw_file_path, index=False)
                download_success = True

            if download_success:
                datasets[key] = df
                print(f" -> Downloaded and saved successfully. Shape: {df.shape}")

        except Exception as e:
            print(f" -> Download failed: {e}")
            print(" -> Falling back to generating a statistically representative synthetic dataset cohort.")
            if synthetic_fallback is None:
                synthetic_fallback = generate_synthetic_cohorts()
            
            df = synthetic_fallback[key]
            df.to_csv(raw_file_path, index=False)
            datasets[key] = df
            print(f" -> Generated and saved synthetic {key} cohort. Shape: {df.shape}")

    return datasets

if __name__ == "__main__":
    ingest_datasets()
