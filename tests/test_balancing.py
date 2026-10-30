import numpy as np
import pytest
from src.data.balancing import AdaptiveDataBalancer

def test_data_balancer():
    np.random.seed(42)
    # Construct a highly imbalanced dataset
    # 90 samples of Class 0, 10 samples of Class 1
    X = np.random.randn(100, 3)
    y = np.array([0] * 90 + [1] * 10)
    
    balancer = AdaptiveDataBalancer(sampling_strategy=0.8, random_state=42)
    X_res, y_res = balancer.fit_resample(X, y)
    
    # Check that target distribution is improved
    unique, counts = np.unique(y_res, return_counts=True)
    count_dict = dict(zip(unique, counts))
    
    # The minority class count should increase (closer to 80% of majority)
    assert count_dict[1] > 10
    assert len(y_res) > 100
    
    # Verify no balancing occurs if the dataset is already balanced (e.g. 50-50)
    X_bal = np.random.randn(100, 3)
    y_bal = np.array([0] * 50 + [1] * 50)
    
    balancer_nobal = AdaptiveDataBalancer(sampling_strategy=0.8, random_state=42)
    X_res_no, y_res_no = balancer_nobal.fit_resample(X_bal, y_bal)
    
    assert X_res_no.shape == X_bal.shape
    assert len(y_res_no) == 100

def test_smote_enn_balancer():
    np.random.seed(42)
    X = np.random.randn(150, 4)
    y = np.array([0] * 130 + [1] * 20)
    
    balancer = AdaptiveDataBalancer(sampling_strategy=0.8, method="smote_enn", random_state=42)
    X_res, y_res = balancer.fit_resample(X, y)
    
    # Should balance, and ENN clean-up might drop some samples, resulting in clean margins
    unique, counts = np.unique(y_res, return_counts=True)
    count_dict = dict(zip(unique, counts))
    assert count_dict[1] > 20

def test_borderline_smote_balancer():
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = np.array([0] * 80 + [1] * 20)
    
    balancer = AdaptiveDataBalancer(sampling_strategy=0.8, method="borderline_smote", random_state=42)
    X_res, y_res = balancer.fit_resample(X, y)
    
    unique, counts = np.unique(y_res, return_counts=True)
    count_dict = dict(zip(unique, counts))
    assert count_dict[1] > 20
