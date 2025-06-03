"""Task 1: Infarct Classification."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.classification import compute_auroc


class InfarctClassificationTask(BaseTask):
    """Infarct classification task."""
    
    @property
    def name(self) -> str:
        return "Infarct Classification"
    
    @property
    def data_dir(self) -> Path:
        return SETTINGS.TASK1_DATA_DIR
    
    @property
    def output_extension(self) -> str:
        return ".csv"
    
    @property
    def image_modalities(self) -> list[str]:
        return ["flair", "adc", "dwi_b1000", "t2s", "swi"]
    
    def evaluate(self, output_path: Path) -> Dict[str, Any]:
        """Evaluate classification predictions."""
        if not output_path.exists():
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()
        
        if not predictions or not ground_truth:
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Align predictions with ground truth by subject
        aligned_predictions, aligned_ground_truth = self._align_data(predictions, ground_truth)
        
        if not aligned_predictions or not aligned_ground_truth:
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Compute AUROC
        auroc = compute_auroc(aligned_ground_truth, aligned_predictions)
        
        return {
            "auroc": auroc,
            "num_cases": len(aligned_predictions)
        }
    
    def _load_predictions(self, output_path: Path) -> dict:
        """Load prediction scores from CSV file."""
        predictions = {}
        try:
            df = pd.read_csv(output_path)
            # Expected format: header,prob_class_1
            if 'prob_class_1' in df.columns and 'header' in df.columns:
                for _, row in df.iterrows():
                    # Extract subject ID from header
                    subject_id = self._extract_subject_id(row['header'])
                    if subject_id:
                        predictions[subject_id] = float(row['prob_class_1'])
        except Exception as e:
            self.logger.error(f"Error loading predictions: {e}")
        
        return predictions
    
    def _load_ground_truth(self) -> dict:
        """Load ground truth labels."""
        ground_truth = {}
        
        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "label.txt"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        label = int(f.read().strip())
                        ground_truth[subject_dir.name] = label
                except (ValueError, IOError):
                    continue
        
        return ground_truth
    
    def _extract_subject_id(self, header: str) -> str:
        """Extract subject ID from header string."""
        # Handle various header formats
        if 'sub_' in header:
            parts = header.split('sub_')
            if len(parts) > 1:
                subject_num = parts[1].split('_')[0].split('.')[0]
                return f"sub_{subject_num}"
        return None
    
    def _align_data(self, predictions: dict, ground_truth: dict) -> tuple[list, list]:
        """Align predictions with ground truth by subject ID."""
        aligned_preds = []
        aligned_gt = []
        
        for subject_id in ground_truth:
            if subject_id in predictions:
                aligned_preds.append(predictions[subject_id])
                aligned_gt.append(ground_truth[subject_id])
        
        return aligned_preds, aligned_gt