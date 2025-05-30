"""Task 1: Infarct Classification."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from .base_task import BaseTask
from config.settings import SETTINGS
from metrics.classification import compute_auroc


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
        return ["modality"]
    
    def evaluate(self, output_path: Path) -> Dict[str, Any]:
        """Evaluate classification predictions."""
        if not output_path.exists():
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()
        
        if not predictions or not ground_truth:
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Compute AUROC
        auroc = compute_auroc(ground_truth, predictions)
        
        return {
            "auroc": auroc,
            "num_cases": len(predictions)
        }
    
    def _load_predictions(self, output_path: Path) -> list[float]:
        """Load prediction scores from CSV file."""
        try:
            df = pd.read_csv(output_path)
            # Expected format: header,prob_class_1
            if 'prob_class_1' in df.columns:
                return df['prob_class_1'].tolist()
            elif len(df.columns) >= 2:
                # Use second column if prob_class_1 not found
                return df.iloc[:, 1].tolist()
            else:
                return []
        except Exception:
            return []
    
    def _load_ground_truth(self) -> list[int]:
        """Load ground truth labels."""
        ground_truth = []
        
        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "label.txt"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        label = int(f.read().strip())
                        ground_truth.append(label)
                except (ValueError, IOError):
                    continue
        
        return ground_truth
