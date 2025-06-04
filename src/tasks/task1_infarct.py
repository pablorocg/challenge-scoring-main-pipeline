"""Task 1: Infarct Classification Task"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.config.settings import SETTINGS
from src.metrics.classification import compute_auroc
from src.tasks.base_task import BaseTask
from src.utils.logging_utils import get_logger


class InfarctClassificationTask(BaseTask):
    """Infarct classification task - one prediction per subject."""

    def __init__(self):
        self.logger = get_logger(__name__)

    @property
    def name(self) -> str:
        return "Infarct Classification"

    @property
    def data_dir(self) -> Path:
        return SETTINGS.TASK1_DATA_DIR

    @property
    def output_extension(self) -> str:
        return ".csv"

    def evaluate(
        self, output_path: Path, task_output_dir: Path = None
    ) -> Dict[str, Any]:
        """Evaluate classification predictions - one per subject."""
        if not output_path.exists():
            self.logger.error(f"Output file not found: {output_path}")
            return {"auroc": SETTINGS.AUROC_WORST}

        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()

        # Align and compute AUROC
        y_true, y_scores = self._align_data(predictions, ground_truth)

        if not y_true or not y_scores:
            self.logger.error("No valid predictions to evaluate")
            return {"auroc": SETTINGS.AUROC_WORST}

        auroc = compute_auroc(y_true, y_scores)
        self.logger.info(f"AUROC: {auroc:.4f} ({len(y_true)} subjects)")

        return {"auroc": auroc}

    def _load_predictions(self, output_path: Path) -> Dict[str, float]:
        """Load predictions from CSV file."""
        predictions = {}

        df = pd.read_csv(output_path)

        if "header" not in df.columns or "prob_class_1" not in df.columns:
            self.logger.error(f"CSV missing required columns: {list(df.columns)}")
            return predictions

        for _, row in df.iterrows():
            subject_id = str(row["header"])
            prob = float(row["prob_class_1"])
            predictions[subject_id] = prob

        self.logger.info(f"Loaded {len(predictions)} predictions")
        return predictions

    def _load_ground_truth(self) -> Dict[str, int]:
        """Load ground truth labels."""
        ground_truth = {}

        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "label.txt"

            if label_file.exists():
                with open(label_file, "r") as f:
                    label = int(f.read().strip())
                    ground_truth[subject_dir.name] = label

        self.logger.info(f"Loaded {len(ground_truth)} ground truth labels")
        return ground_truth

    def _align_data(
        self, predictions: Dict[str, float], ground_truth: Dict[str, int]
    ) -> tuple[list, list]:
        """Align predictions with ground truth."""
        y_true, y_scores = [], []

        for subject_id, pred_score in predictions.items():
            if subject_id in ground_truth:
                y_true.append(ground_truth[subject_id])
                y_scores.append(pred_score)

        return y_true, y_scores
