# src/tasks/task2_meningioma.py
"""Task 2: Meningioma Segmentation - Fixed implementation with MONAI metrics."""

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.utils.logging_utils import get_logger

# MONAI imports
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.data import MetaTensor


class MeningiomaSegmentationTask(BaseTask):
    """Meningioma segmentation task with MONAI metrics."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Initialize MONAI metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.nsd_metric = SurfaceDistanceMetric(
            include_background=False, 
            symmetric=True,
            distance_metric="euclidean"
        )
    
    @property
    def name(self) -> str:
        return "Meningioma Segmentation"
    
    @property
    def data_dir(self) -> Path:
        return SETTINGS.TASK2_DATA_DIR
    
    @property
    def output_extension(self) -> str:
        return ".nii.gz"
    
    def evaluate(self, output_path: Path, task_output_dir: Path = None) -> Dict[str, Any]:
        """Evaluate segmentation predictions."""
        self.logger.info("Starting meningioma segmentation evaluation")
        
        # Find prediction files
        prediction_files = self._find_prediction_files(output_path, task_output_dir)
        
        if not prediction_files:
            self.logger.error("No prediction files found")
            return self._get_worst_metrics()
        
        # Get available subjects
        available_subjects = self._get_available_subjects()
        
        if not available_subjects:
            self.logger.error("No ground truth subjects found")
            return self._get_worst_metrics()
        
        # Evaluate each subject
        subject_metrics = []
        valid_evaluations = 0
        
        for pred_path, subject_id in prediction_files:
            if subject_id in available_subjects:
                metrics = self._evaluate_single_subject(pred_path, subject_id)
                if metrics:
                    subject_metrics.append(metrics)
                    valid_evaluations += 1
                    self.logger.info(f"✅ {subject_id}: DSC={metrics['dice']:.4f}, NSD={metrics['nsd']:.4f}")
                else:
                    self.logger.error(f"❌ {subject_id}: Evaluation failed")
            else:
                self.logger.warning(f"⚠️ {subject_id}: No ground truth found")
        
        if valid_evaluations == 0:
            self.logger.error("No valid evaluations completed")
            return self._get_worst_metrics()
        
        # Save individual subject metrics
        if task_output_dir:
            self._save_subject_metrics_csv(subject_metrics, task_output_dir)
        
        # Compute average metrics
        final_metrics = self._compute_average_metrics(subject_metrics)
        
        self.logger.info(f"Final metrics - DSC: {final_metrics['dice']:.4f}, "
                        f"NSD: {final_metrics['nsd']:.4f} ({valid_evaluations} subjects)")
        
        return final_metrics
    
    def _find_prediction_files(self, output_path: Path, task_output_dir: Path) -> List[Tuple[Path, str]]:
        """Find prediction files and extract subject IDs."""
        prediction_files = []
        
        # For segmentation, predictions are individual subject output files
        if task_output_dir and task_output_dir.exists():
            # Look for subject output files: sub_XXX_output.nii.gz
            for output_file in task_output_dir.glob("sub_*_output.nii.gz"):
                subject_id = self._extract_subject_from_filename(output_file.name)
                if subject_id:
                    prediction_files.append((output_file, subject_id))
        
        # Fallback: check if output_path is a single file
        if not prediction_files and output_path.is_file() and output_path.suffix == '.gz':
            subject_id = self._extract_subject_from_filename(output_path.name)
            if subject_id:
                prediction_files.append((output_path, subject_id))
        
        self.logger.info(f"Found {len(prediction_files)} prediction files")
        return prediction_files
    
    def _get_available_subjects(self) -> List[str]:
        """Get list of subjects with ground truth."""
        subjects = []
        
        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "seg.nii.gz"
            if label_file.exists():
                subjects.append(subject_dir.name)
        
        self.logger.info(f"Found {len(subjects)} subjects with ground truth")
        return subjects
    
    def _extract_subject_from_filename(self, filename: str) -> Optional[str]:
        """Extract subject ID from filename (e.g., sub_123_output.nii.gz -> sub_123)."""
        # Remove extensions
        name = filename.replace('.nii.gz', '').replace('.nii', '')
        
        # Look for sub_XXX pattern
        if name.startswith('sub_'):
            # Extract sub_XXX part (handles sub_123, sub_123_output, etc.)
            parts = name.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"  # sub_123
        
        return None
    
    def _evaluate_single_subject(self, pred_path: Path, subject_id: str) -> Optional[Dict[str, Any]]:
        """Evaluate a single subject's segmentation."""
        try:
            # Load prediction
            pred_data = self._load_nifti_file(pred_path)
            if pred_data is None:
                return None
            
            # Load ground truth
            gt_path = self.get_labels_path(subject_id) / "seg.nii.gz"
            gt_data = self._load_nifti_file(gt_path)
            if gt_data is None:
                return None
            
            # Ensure same shape
            if pred_data.shape != gt_data.shape:
                self.logger.error(f"Shape mismatch for {subject_id}: "
                                f"pred={pred_data.shape}, gt={gt_data.shape}")
                return None
            
            # Convert to binary and add batch/channel dimensions
            pred_binary = self._prepare_tensor(pred_data)
            gt_binary = self._prepare_tensor(gt_data)
            
            # Compute metrics
            dice_score = self._compute_dice_metric(pred_binary, gt_binary)
            nsd_score = self._compute_nsd_metric(pred_binary, gt_binary)
            
            return {
                'subject_id': subject_id,
                'dice': float(dice_score),
                'nsd': float(nsd_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error evaluating {subject_id}: {e}")
            return None
    
    def _load_nifti_file(self, file_path: Path) -> Optional[np.ndarray]:
        """Load NIfTI file and return data."""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            return data
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _prepare_tensor(self, data: np.ndarray) -> torch.Tensor:
        """Prepare tensor for MONAI metrics with proper dimensions."""
        # Convert to binary
        binary_data = (data > 0.5).astype(np.float32)
        
        # Add batch and channel dimensions: [B, C, H, W, D]
        tensor = torch.from_numpy(binary_data)[None, None, ...]
        
        return tensor
    
    def _compute_dice_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Dice coefficient using MONAI."""
        try:
            self.dice_metric.reset()
            self.dice_metric(pred, gt)
            dice_score = self.dice_metric.aggregate().item()
            
            # Handle edge cases
            if np.isnan(dice_score) or np.isinf(dice_score):
                # Check if both masks are empty
                if torch.sum(pred) == 0 and torch.sum(gt) == 0:
                    return 1.0  # Perfect match for empty masks
                else:
                    return SETTINGS.DSC_WORST
            
            return dice_score
            
        except Exception as e:
            self.logger.error(f"Error computing Dice: {e}")
            return SETTINGS.DSC_WORST
    
    def _compute_nsd_metric(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Compute Normalized Surface Distance using MONAI."""
        try:
            # Check if both masks have surfaces
            if torch.sum(pred) == 0 or torch.sum(gt) == 0:
                if torch.sum(pred) == 0 and torch.sum(gt) == 0:
                    return 0.0  # No surface distance for empty masks
                else:
                    return SETTINGS.NSD_WORST
            
            self.nsd_metric.reset()
            self.nsd_metric(pred, gt)
            nsd_score = self.nsd_metric.aggregate().item()
            
            # Handle edge cases
            if np.isnan(nsd_score) or np.isinf(nsd_score):
                return SETTINGS.NSD_WORST
            
            return nsd_score
            
        except Exception as e:
            self.logger.error(f"Error computing NSD: {e}")
            return SETTINGS.NSD_WORST
    
    def _save_subject_metrics_csv(self, subject_metrics: List[Dict], task_output_dir: Path) -> None:
        """Save per-subject metrics to CSV."""
        try:
            df = pd.DataFrame(subject_metrics)
            csv_path = task_output_dir / "subject_metrics.csv"
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved subject metrics to {csv_path}")
        except Exception as e:
            self.logger.error(f"Error saving subject metrics: {e}")
    
    def _compute_average_metrics(self, subject_metrics: List[Dict]) -> Dict[str, Any]:
        """Compute average metrics from subject metrics."""
        if not subject_metrics:
            return self._get_worst_metrics()
        
        dice_scores = [m['dice'] for m in subject_metrics]
        nsd_scores = [m['nsd'] for m in subject_metrics]
        
        return {
            "dice": float(np.mean(dice_scores)),
            "nsd": float(np.mean(nsd_scores))
        }
    
    def _get_worst_metrics(self) -> Dict[str, Any]:
        """Return worst possible metrics."""
        return {
            "dice": SETTINGS.DSC_WORST,
            "nsd": SETTINGS.NSD_WORST
        }