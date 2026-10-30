import numpy as np
import pytest
from src.features.afo import AdaptiveFeatureOptimizer

def test_afo_optimizer():
    np.random.seed(42)
    N = 100
    P = 10
    X = np.random.randn(N, P)
    # Make columns 0 and 1 strong predictors
    y = (X[:, 0] * 2.0 + X[:, 1] * 1.5 + np.random.randn(N) * 0.1 > 0).astype(int)
    
    # Instantiate AFO selecting top 4 features
    afo = AdaptiveFeatureOptimizer(k_top=4, random_state=42)
    
    X_sel = afo.fit_transform(X, y)
    
    # Check dimensions
    assert X_sel.shape == (N, 4)
    
    # Strong predictors should be selected (index 0 and 1)
    assert 0 in afo.selected_indices_
    assert 1 in afo.selected_indices_
    
    # Test transform
    X_test = np.random.randn(10, P)
    X_test_sel = afo.transform(X_test)
    assert X_test_sel.shape == (10, 4)
