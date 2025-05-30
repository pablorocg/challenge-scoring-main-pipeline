"""Task 3: Brain Age Prediction."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .base_task import BaseTask
from config.settings import SETTINGS
from metrics.regression import compute_absolute_error, compute_correlation


class BrainAgePredictionTask(BaseTask):
    """Brain age prediction task."""
    
    @property
    def name(self) -> str:
        return "Brain Age Prediction"
    
    @property
    def data_dir(self) -> Path:
        return SETTINGS.TASK3_DATA_DIR
    
    @property
    def output_extension(self) -> str:
        return ".csv"
    
    @property
    def image_modalities(self) -> list[str]:
        return ["modality"]
    
    def evaluate(self, output_path: Path) -> Dict[str, Any]:
        """Evaluate age predictions."""
        if not output_path.exists():
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()
        
        if not predictions or not ground_truth:
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        # Compute metrics
        abs_error = compute_absolute_error(ground_truth, predictions)
        correlation = compute_correlation(ground_truth, predictions)
        
        return {
            "absolute_error": abs_error,
            "correlation": correlation,
            "num_cases": len(predictions)
        }
    
    def _load_predictions(self, output_path: Path) -> list[float]:
        """Load age predictions from CSV file."""
        try:
            df = pd.read_csv(output_path)
            # Expected format: header,value
            if 'value' in df.columns:
                return df['value'].tolist()
            elif len(df.columns) >= 2:
                # Use second column if value not found
                return df.iloc[:, 1].tolist()
            else:
                return []
        except Exception:
            return []
    
    def _load_ground_truth(self) -> list[float]:
        """Load ground truth ages."""
        ground_truth = []
        
        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "label.txt"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        # Round to nearest integer as specified
                        age = round(float(f.read().strip()))
                        ground_truth.append(float(age))
                except (ValueError, IOError):
                    continue
        
        return ground_truth
