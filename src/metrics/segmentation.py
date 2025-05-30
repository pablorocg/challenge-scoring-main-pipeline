"""Segmentation metrics."""

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from config.settings import SETTINGS


def compute_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Dice Similarity Coefficient."""
    try:
        # Convert to binary
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        
        intersection = np.sum(y_true_bin * y_pred_bin)
        union = np.sum(y_true_bin) + np.sum(y_pred_bin)
        
        if union == 0:
            return 1.0 if intersection == 0 else SETTINGS.DSC_WORST
        
        dice = 2.0 * intersection / union
        return float(dice)
    except Exception:
        return SETTINGS.DSC_WORST


def compute_nsd(y_true: np.ndarray, y_pred: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """Compute Normalized Surface Distance."""
    try:
        # Convert to binary
        y_true_bin = (y_true > 0).astype(int)
        y_pred_bin = (y_pred > 0).astype(int)
        
        # Get surface points
        true_surface = _get_surface_points(y_true_bin)
        pred_surface = _get_surface_points(y_pred_bin)
        
        if len(true_surface) == 0 or len(pred_surface) == 0:
            return SETTINGS.NSD_WORST
        
        # Compute surface distances
        distances = []
        for point in true_surface:
            min_dist = np.min(np.linalg.norm(pred_surface - point, axis=1))
            distances.append(min_dist)
        
        # Normalize by image diagonal
        diagonal = np.sqrt(np.sum([d**2 for d in y_true.shape]))
        nsd = np.mean(distances) / diagonal
        
        return float(nsd)
    except Exception:
        return SETTINGS.NSD_WORST


def _get_surface_points(binary_mask: np.ndarray) -> np.ndarray:
    """Extract surface points from binary mask."""
    from scipy import ndimage
    
    # Simple surface extraction using erosion
    eroded = ndimage.binary_erosion(binary_mask)
    surface = binary_mask & ~eroded
    
    return np.array(np.where(surface)).T
