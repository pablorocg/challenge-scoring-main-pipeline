"""Task 1: Infarct Classification."""

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
        return ".txt"
    
    @property
    def image_modalities(self) -> list[str]:
        return ["flair", "adc", "dwi_b1000", "t2s_or_swi"]
    
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
        """Load prediction scores from output file."""
        predictions = []
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        predictions.append(float(line))
        except (ValueError, IOError):
            return []
        
        return predictions
    
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
