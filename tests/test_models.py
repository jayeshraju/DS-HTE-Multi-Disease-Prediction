"""
Tests 7, 8: Tabular Transformer architecture and DS-HTE ensemble mechanisms.
"""

import numpy as np
import torch
import pytest

from src.models.transformer import TabularTransformerClassifier, TabularTransformerNet
from src.models.dshte import DSHTEClassifier


def test_transformer_forward_backward():
    """
    Test 7a: Transformer shapes, forward pass, reconstruction pass, and backpropagation.
    """
    net = TabularTransformerNet(n_features=8, d_token=32, n_heads=2, n_layers=1)
    x = torch.randn(16, 8)
    
    # Classification forward pass
    logits = net(x)
    assert logits.shape == (16,)
    loss_cls = logits.sum()
    loss_cls.backward()

    # Reconstruction pass
    recon = net.reconstruct(x)
    assert recon.shape == (16, 8)
    loss_recon = recon.sum()
    loss_recon.backward()

    for p in net.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_transformer_deterministic_and_api():
    """
    Test 7b: TabularTransformerClassifier reproducibility and sklearn API.
    """
    X = np.random.RandomState(42).randn(100, 6)
    y = np.random.RandomState(42).randint(0, 2, size=100)

    clf1 = TabularTransformerClassifier(epochs=3, d_model=32, n_heads=2, n_layers=1, random_state=42)
    clf1.fit(X, y)
    p1 = clf1.predict_proba(X)

    clf2 = TabularTransformerClassifier(epochs=3, d_model=32, n_heads=2, n_layers=1, random_state=42)
    clf2.fit(X, y)
    p2 = clf2.predict_proba(X)

    assert np.allclose(p1, p2, atol=1e-5), "Transformer should be deterministic given seed"
    assert p1.shape == (100, 2)
    assert np.all((p1 >= 0.0) & (p1 <= 1.0))
    assert np.allclose(np.sum(p1, axis=1), 1.0)


def test_ensemble_weights_and_bounds():
    """
    Test 8: DS-HTE weights are non-negative, sum to 1, and probabilities in [0, 1].
    """
    X = np.random.RandomState(42).randn(80, 5)
    y = np.random.RandomState(42).randint(0, 2, size=80)

    # Test with subset of models (Transformer + LightGBM)
    ensemble = DSHTEClassifier(
        include_transformer=True,
        include_xgb=False,
        include_lgbm=True,
        include_catboost=False,
        transformer_params={"epochs": 2, "d_model": 16, "n_heads": 2, "n_layers": 1},
        lgbm_params={"n_estimators": 10},
        n_inner_splits=2,
        random_state=42
    )
    ensemble.fit(X, y)

    weights = ensemble.weights_
    assert len(weights) == 2
    for w in weights.values():
        assert w >= -1e-6, f"Weight {w} must be non-negative"
    assert np.isclose(sum(weights.values()), 1.0, atol=1e-4), "Weights must sum to 1.0"

    probs = ensemble.predict_proba(X)
    assert probs.shape == (80, 2)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert np.allclose(np.sum(probs, axis=1), 1.0)
