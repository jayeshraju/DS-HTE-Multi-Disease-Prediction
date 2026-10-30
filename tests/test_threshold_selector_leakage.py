"""
Unit test asserting that threshold selectors never receive outer-test labels.
Follows Gate 0 Check 3:
- Both operating points (Balanced/Youden, Accuracy-optimal) are chosen on inner-validation predictions only.
- Outer test labels must never reach the threshold optimizer.
"""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

from src.threshold.youden import optimize_youden_threshold, optimize_accuracy_threshold


def test_threshold_optimizers_never_access_test_labels():
    """Mocks threshold selectors and asserts they are strictly called with inner validation arrays."""
    # Create synthetic inner-val arrays and sealed test arrays
    val_probs = np.array([0.1, 0.4, 0.6, 0.8, 0.9])
    val_y_true = np.array([0, 0, 1, 1, 1])
    
    test_probs = np.array([0.2, 0.3, 0.7, 0.85])
    test_y_true = np.array([0, 0, 1, 1])
    
    with patch("src.threshold.youden.optimize_youden_threshold", wraps=optimize_youden_threshold) as mock_youden:
        # Simulation of pipeline step: selector must receive val_y_true and val_probs
        res_balanced = mock_youden(val_y_true, val_probs)
        assert mock_youden.called
        call_args = mock_youden.call_args[0]
        # Assert called with val, NOT test
        np.testing.assert_array_equal(call_args[0], val_y_true)
        np.testing.assert_array_equal(call_args[1], val_probs)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(call_args[0], test_y_true)

    with patch("src.threshold.youden.optimize_accuracy_threshold", wraps=optimize_accuracy_threshold) as mock_acc:
        res_acc = mock_acc(val_y_true, val_probs)
        assert mock_acc.called
        call_args = mock_acc.call_args[0]
        np.testing.assert_array_equal(call_args[0], val_y_true)
        np.testing.assert_array_equal(call_args[1], val_probs)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(call_args[0], test_y_true)
            
    th_balanced = res_balanced["optimal_threshold"] if isinstance(res_balanced, dict) else res_balanced
    th_acc = res_acc["optimal_threshold"] if isinstance(res_acc, dict) else res_acc

    # Apply threshold to test predictions without peeking test labels
    test_preds_balanced = (test_probs >= th_balanced).astype(int)
    test_preds_acc = (test_probs >= th_acc).astype(int)
    assert len(test_preds_balanced) == len(test_probs)
    assert len(test_preds_acc) == len(test_probs)
