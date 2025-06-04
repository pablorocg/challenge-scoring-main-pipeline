"""Classification metrics."""

from sklearn.metrics import roc_auc_score

from config.settings import SETTINGS


def compute_auroc(y_true: list[int], y_scores: list[float]) -> float:
    """Compute Area Under ROC Curve."""
    if len(y_true) != len(y_scores) or len(y_true) == 0:
        return SETTINGS.AUROC_WORST

    try:
        # Check if all labels are the same (no positive or negative cases)
        if len(set(y_true)) < 2:
            return SETTINGS.AUROC_WORST

        return float(roc_auc_score(y_true, y_scores))
    except Exception:
        return SETTINGS.AUROC_WORST
