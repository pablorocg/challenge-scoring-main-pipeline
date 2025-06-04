"""Regression metrics."""

import numpy as np
from scipy.stats import pearsonr

from config.settings import SETTINGS


def compute_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    """Compute mean absolute error."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return SETTINGS.AE_WORST

    try:
        mae = np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
        return float(mae)
    except Exception:
        return SETTINGS.AE_WORST


def compute_correlation(y_true: list[float], y_pred: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        return SETTINGS.CORR_WORST

    try:
        correlation, _ = pearsonr(y_true, y_pred)
        return float(correlation) if not np.isnan(correlation) else SETTINGS.CORR_WORST
    except Exception:
        return SETTINGS.CORR_WORST
