# Code to compute Normal Surface Distance (NSD) between two binary masks (load, compute and save)
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_nsd(y_true, y_pred, tolerance=2.0):
    # Extract surfaces (edge voxels)
    surface_true = np.logical_xor(y_true, binary_erosion(y_true))
    surface_pred = np.logical_xor(y_pred, binary_erosion(y_pred))
    
    # Compute distance transforms
    dist_map_true = distance_transform_edt(~surface_true)
    dist_map_pred = distance_transform_edt(~surface_pred)
    
    # Calculate NSD: % of predicted surface within tolerance of true surface
    nsd_pred_to_true = np.mean(dist_map_true[surface_pred] <= tolerance)
    nsd_true_to_pred = np.mean(dist_map_pred[surface_true] <= tolerance)
    return (nsd_pred_to_true + nsd_true_to_pred) / 2  # Symmetric NSD