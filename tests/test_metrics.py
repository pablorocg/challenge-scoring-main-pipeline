"""Tests for metric computation."""

import numpy as np
from src.metrics.classification import compute_auroc
from src.metrics.regression import compute_absolute_error, compute_correlation


def test_auroc():
    """Test AUROC computation."""
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    auroc = compute_auroc(y_true, y_scores)
    assert 0.0 <= auroc <= 1.0


def test_absolute_error():
    """Test absolute error computation."""
    y_true = [20.0, 30.0, 40.0]
    y_pred = [22.0, 28.0, 41.0]
    mae = compute_absolute_error(y_true, y_pred)
    assert mae > 0


def test_correlation():
    """Test correlation computation."""
    y_true = [20.0, 30.0, 40.0, 50.0]
    y_pred = [22.0, 28.0, 41.0, 48.0]
    corr = compute_correlation(y_true, y_pred)
    assert -1.0 <= corr <= 1.0
