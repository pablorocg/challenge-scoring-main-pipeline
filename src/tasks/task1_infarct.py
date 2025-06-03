"""Task 1: Infarct Classification - FIXED VERSION."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.classification import compute_auroc
from src.utils.logging_utils import get_logger


class InfarctClassificationTask(BaseTask):
    """Infarct classification task with proper T2*/SWI handling."""
    
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
    
    @property
    def image_modalities(self) -> list[str]:
        # Core modalities that should always be present
        return ["flair", "adc", "dwi_b1000", "t2s", "swi"]
    
    def get_subject_modalities(self, subject_dir: Path) -> list[str]:
        """Get actual available modalities for a specific subject."""
        ses_dir = subject_dir / "ses_1"
        if not ses_dir.exists():
            return []
        
        available_modalities = []
        
        # Core modalities
        for modality in ["flair", "adc", "dwi_b1000"]:
            modality_file = ses_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                available_modalities.append(modality)
        
        # T2*/SWI either/or logic
        t2s_file = ses_dir / "t2s.nii.gz"
        swi_file = ses_dir / "swi.nii.gz"
        
        if t2s_file.exists():
            available_modalities.append("t2s")
        elif swi_file.exists():
            available_modalities.append("swi")
        
        return available_modalities
    
    def evaluate(self, output_path: Path) -> Dict[str, Any]:
        """Evaluate classification predictions."""
        if not output_path.exists():
            self.logger.error(f"Output file does not exist: {output_path}")
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()
        
        if not predictions or not ground_truth:
            self.logger.error("Failed to load predictions or ground truth")
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Align predictions with ground truth by subject
        aligned_predictions, aligned_ground_truth = self._align_data(predictions, ground_truth)
        
        if not aligned_predictions or not aligned_ground_truth:
            self.logger.error("Failed to align predictions with ground truth")
            return {"auroc": SETTINGS.AUROC_WORST}
        
        # Compute AUROC
        auroc = compute_auroc(aligned_ground_truth, aligned_predictions)
        
        self.logger.info(f"Evaluation complete: AUROC={auroc:.4f}, Cases={len(aligned_predictions)}")
        
        return {
            "auroc": auroc
        }
    
    def _load_predictions(self, output_path: Path) -> dict:
        """Load prediction scores from CSV file."""
        predictions = {}
        try:
            df = pd.read_csv(output_path)
            self.logger.info(f"Loaded CSV with columns: {list(df.columns)}")
            self.logger.info(f"CSV shape: {df.shape}")
            
            # Expected format: header,prob_class_1
            if 'prob_class_1' in df.columns and 'header' in df.columns:
                self.logger.info("Sample headers from CSV:")
                for i, header in enumerate(df['header'].head()):
                    self.logger.info(f"  {i+1}: {header}")
                
                for _, row in df.iterrows():
                    header = row['header']
                    prob = float(row['prob_class_1'])
                    
                    # Headers now include modality: "sub_1_flair", "sub_1_adc", etc.
                    # Keep them as-is for individual modality evaluation
                    predictions[header] = prob
                    self.logger.debug(f"Loaded prediction: {header} -> {prob}")
                        
                self.logger.info(f"Successfully loaded {len(predictions)} modality predictions")
                
                # Debug: show some loaded predictions
                sample_preds = dict(list(predictions.items())[:3])
                self.logger.info(f"Sample predictions: {sample_preds}")
                
            else:
                self.logger.error(f"CSV missing required columns. Expected 'header' and 'prob_class_1', found: {list(df.columns)}")
                
        except Exception as e:
            self.logger.error(f"Error loading predictions: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
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
                except (ValueError, IOError) as e:
                    self.logger.warning(f"Could not read label for {subject_dir.name}: {e}")
                    continue
        
        self.logger.info(f"Loaded ground truth for {len(ground_truth)} subjects")
        return ground_truth
    
    def _extract_subject_id(self, header: str) -> str:
        """Extract subject ID from header string."""
        # Handle various header formats
        if 'sub_' in header:
            parts = header.split('sub_')
            if len(parts) > 1:
                subject_num = parts[1].split('_')[0].split('.')[0]
                return f"sub_{subject_num}"
        
        # Fallback: try to extract any subject pattern
        import re
        match = re.search(r'sub_(\d+)', header)
        if match:
            return f"sub_{match.group(1)}"
        
        self.logger.warning(f"Could not extract subject ID from header: {header}")
        return None
    
    def _align_data(self, predictions: dict, ground_truth: dict) -> tuple[list, list]:
        """Align predictions with ground truth - expand GT for each modality."""
        aligned_preds = []
        aligned_gt = []
        
        self.logger.info(f"Aligning data: {len(predictions)} modality predictions, {len(ground_truth)} subject ground truth labels")
        
        # Debug: show some samples
        pred_headers = list(predictions.keys())
        gt_subjects = list(ground_truth.keys())
        
        self.logger.info(f"Prediction headers sample: {pred_headers[:5]}")
        self.logger.info(f"Ground truth subjects sample: {gt_subjects[:5]}")
        
        # Process each prediction (which is now per modality)
        matched_cases = []
        for header, pred_value in predictions.items():
            # Extract subject ID from header: "sub_1_flair" -> "sub_1"
            subject_id = self._extract_subject_id(header)
            
            if subject_id and subject_id in ground_truth:
                # Use the same ground truth label for this modality
                gt_value = ground_truth[subject_id]
                
                aligned_preds.append(pred_value)
                aligned_gt.append(gt_value)
                matched_cases.append(header)
                
                self.logger.debug(f"Matched: {header} (pred={pred_value}) -> {subject_id} (gt={gt_value})")
            else:
                if subject_id:
                    self.logger.warning(f"No ground truth found for subject {subject_id} (header: {header})")
                else:
                    self.logger.warning(f"Could not extract subject ID from header: {header}")
        
        self.logger.info(f"Successfully aligned {len(aligned_preds)} modality predictions with ground truth")
        self.logger.info(f"Matched cases: {matched_cases[:5]}{'...' if len(matched_cases) > 5 else ''}")
        
        # Show alignment summary
        unique_subjects = set(self._extract_subject_id(header) for header in matched_cases if self._extract_subject_id(header))
        self.logger.info(f"Evaluation covers {len(unique_subjects)} subjects with {len(aligned_preds)} total modality cases")
        
        return aligned_preds, aligned_gt