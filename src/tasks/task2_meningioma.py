# src/tasks/task2_meningioma.py - VERSION CSV METRICS
"""Task 2: Meningioma Segmentation - Genera CSV con métricas por sujeto."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.segmentation import compute_dice, compute_nsd
from monai.metrics import surface_dice as nsd
from monai.metrics import meandice as dice
from src.utils.logging_utils import get_logger
import nibabel as nib


class MeningiomaSegmentationTask(BaseTask):
    """Meningioma segmentation task class."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "Meningioma Segmentation"
    
    @property
    def data_dir(self) -> Path:
        return SETTINGS.TASK2_DATA_DIR
    
    @property
    def output_extension(self) -> str:
        return ".nii.gz"
    

    def _evaluate_parallel(self, ...):
        """Method still not implemented, use evaluate() in its place."""
        raise NotImplementedError("Parallel evaluation not implemented for this task. Use evaluate() instead.")
    
    def _evaluate_sequential(self, ...):
        """Method still not implemented, use evaluate() in its place."""
        raise NotImplementedError("Sequential evaluation not implemented for this task. Use evaluate() instead.")
    

    def evaluate(self, output_path: Path, task_output_dir: Path = None) -> Dict[str, Any]:
        # Evaluate (sequential or parallel)
        pass
        
    
    def _find_prediction_files(self, output_path: Path, task_output_dir: Path) -> List[tuple]:
        pass
    
    def _get_available_subjects(self) -> List[str]:
        pass
    
    def _extract_subject_from_filename(self, filename: str) -> str:
        pass
    
    def _save_subject_metrics_csv(self, subject_metrics: List[dict], task_output_dir: Path) -> str:
        pass
    
    def _compute_average_metrics(self, subject_metrics: List[dict]) -> Dict[str, Any]:
        """
        Compute avgerage metrics from subject metrics.
        
        Returns something like:
        {
            "dice": float(mean_dice),
            "nsd": float(mean_nsd)
        }
        
        """
        
        
        return 
    
    def _load_segmentation(self, seg_path: Path):
        """Cargar segmentación desde archivo NIfTI."""
        try:
            nii = nib.load(seg_path)
            data = nii.get_fdata()
            return data
            
        except Exception as e:
            self.logger.error(f"   ❌ Error loading segmentation {seg_path}: {e}")
            return None
    

    def _load_nifti_file(self, file_path: Path):
        """Cargar archivo NIfTI y retornar datos."""
        try:
            nii = nib.load(file_path)
            data = nii.get_fdata()
            return data
        except Exception as e:
            self.logger.error(f"   ❌ Error loading NIfTI file {file_path}: {e}")
            return None
    