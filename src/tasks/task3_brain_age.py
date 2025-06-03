# src/tasks/task3_brain_age.py
"""Task 3: Brain Age Prediction - Simplified version."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.regression import compute_absolute_error, compute_correlation
from src.utils.logging_utils import get_logger


class BrainAgePredictionTask(BaseTask):
    """Brain age prediction task - one prediction per subject."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @property
    def name(self) -> str:
        return "Brain Age Prediction"
    
    @property
    def data_dir(self) -> Path:
        return SETTINGS.TASK3_DATA_DIR
    
    @property
    def output_extension(self) -> str:
        return ".csv"
    
    def evaluate(self, output_path: Path, task_output_dir: Path = None) -> Dict[str, Any]:
        """Evaluate age predictions - one per subject."""
        if not output_path.exists():
            self.logger.error(f"Output file not found: {output_path}")
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()
        
        # Align and compute metrics
        y_true, y_pred = self._align_data(predictions, ground_truth)
        
        if not y_true or not y_pred:
            self.logger.error("No valid predictions to evaluate")
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        abs_error = compute_absolute_error(y_true, y_pred)
        correlation = compute_correlation(y_true, y_pred)
        
        self.logger.info(f"MAE: {abs_error:.2f}, Correlation: {correlation:.4f} ({len(y_true)} subjects)")
        
        return {
            "absolute_error": abs_error,
            "correlation": correlation
        }
    
    def _load_predictions(self, output_path: Path) -> Dict[str, float]:
        """Load age predictions from CSV file."""
        predictions = {}
        
        df = pd.read_csv(output_path)
        
        if 'header' not in df.columns or 'value' not in df.columns:
            self.logger.error(f"CSV missing required columns: {list(df.columns)}")
            return predictions
        
        for _, row in df.iterrows():
            subject_id = str(row['header'])
            age = float(row['value'])
            
            # Validate age range
            if 0 <= age <= 150:
                predictions[subject_id] = age
        
        self.logger.info(f"Loaded {len(predictions)} age predictions")
        return predictions
    
    def _load_ground_truth(self) -> Dict[str, float]:
        """Load ground truth ages."""
        ground_truth = {}
        
        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "label.txt"
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    age = float(f.read().strip())
                    if 0 <= age <= 150:
                        ground_truth[subject_dir.name] = age
        
        self.logger.info(f"Loaded {len(ground_truth)} ground truth ages")
        return ground_truth
    
    def _align_data(self, predictions: Dict[str, float], ground_truth: Dict[str, float]) -> tuple[list, list]:
        """Align predictions with ground truth."""
        y_true, y_pred = [], []
        
        for subject_id, pred_age in predictions.items():
            if subject_id in ground_truth:
                y_true.append(ground_truth[subject_id])
                y_pred.append(pred_age)
        
        return y_true, y_pred