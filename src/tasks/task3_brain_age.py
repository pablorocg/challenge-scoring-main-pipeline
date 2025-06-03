"""Task 3: Brain Age Prediction - FIXED VERSION."""

import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.regression import compute_absolute_error, compute_correlation
from src.utils.logging_utils import get_logger


class BrainAgePredictionTask(BaseTask):
    """Brain age prediction task with proper T1/T2 handling."""
    
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
    
    @property
    def image_modalities(self) -> list[str]:
        # Brain age prediction uses T1 and T2 structural images
        return ["t1", "t2"]
    
    def get_subject_modalities(self, subject_dir: Path) -> list[str]:
        """Get actual available modalities for a specific subject."""
        ses_dir = subject_dir / "ses_1"
        if not ses_dir.exists():
            return []
        
        available_modalities = []
        
        # Check T1 and T2
        for modality in ["t1", "t2"]:
            modality_file = ses_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                available_modalities.append(modality)
        
        return available_modalities
    
    def evaluate(self, output_path: Path) -> Dict[str, Any]:
        """Evaluate age predictions."""
        if not output_path.exists():
            self.logger.error(f"Output file does not exist: {output_path}")
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        # Load predictions and ground truth
        predictions = self._load_predictions(output_path)
        ground_truth = self._load_ground_truth()
        
        if not predictions:
            self.logger.error("Failed to load predictions")
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        if not ground_truth:
            self.logger.error("Failed to load ground truth")
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        # Align predictions with ground truth by subject
        aligned_predictions, aligned_ground_truth = self._align_data(predictions, ground_truth)
        
        if not aligned_predictions or not aligned_ground_truth:
            self.logger.error("Failed to align predictions with ground truth")
            return {
                "absolute_error": SETTINGS.AE_WORST,
                "correlation": SETTINGS.CORR_WORST
            }
        
        # Compute metrics
        abs_error = compute_absolute_error(aligned_ground_truth, aligned_predictions)
        correlation = compute_correlation(aligned_ground_truth, aligned_predictions)
        
        self.logger.info(f"Evaluation complete: MAE={abs_error:.2f}, Corr={correlation:.4f}, Cases={len(aligned_predictions)}")
        
        # Log some statistics
        import numpy as np
        pred_ages = np.array(aligned_predictions)
        true_ages = np.array(aligned_ground_truth)
        
        self.logger.info(f"Age statistics:")
        self.logger.info(f"  Predicted: mean={pred_ages.mean():.1f}, std={pred_ages.std():.1f}, range=[{pred_ages.min():.1f}, {pred_ages.max():.1f}]")
        self.logger.info(f"  True: mean={true_ages.mean():.1f}, std={true_ages.std():.1f}, range=[{true_ages.min():.1f}, {true_ages.max():.1f}]")
        
        return {
            "absolute_error": abs_error,
            "correlation": correlation
        }
    
    def _load_predictions(self, output_path: Path) -> dict:
        """Load age predictions from CSV file."""
        predictions = {}
        try:
            df = pd.read_csv(output_path)
            self.logger.info(f"Loaded CSV with columns: {list(df.columns)}")
            self.logger.info(f"CSV shape: {df.shape}")
            
            # Expected format: header,value
            if 'value' in df.columns and 'header' in df.columns:
                self.logger.info("Sample headers from CSV:")
                for i, header in enumerate(df['header'].head()):
                    self.logger.info(f"  {i+1}: {header}")
                
                for _, row in df.iterrows():
                    header = row['header']
                    age = float(row['value'])
                    
                    # Validate age range
                    if 0 <= age <= 150:  # Reasonable age range
                        # Headers now include modality: "sub_1_t1", "sub_1_t2", etc.
                        # Keep them as-is for individual modality evaluation
                        predictions[header] = age
                        self.logger.debug(f"Loaded prediction: {header} -> {age:.1f} years")
                    else:
                        self.logger.warning(f"Unrealistic age prediction for {header}: {age}")
                        predictions[header] = max(0, min(150, age))  # Clamp to reasonable range
                        
                self.logger.info(f"Successfully loaded {len(predictions)} modality age predictions")
                
                # Debug: show some loaded predictions
                sample_preds = dict(list(predictions.items())[:3])
                self.logger.info(f"Sample predictions: {sample_preds}")
                
            else:
                self.logger.error(f"CSV missing required columns. Expected 'header' and 'value', found: {list(df.columns)}")
                
        except Exception as e:
            self.logger.error(f"Error loading predictions: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
        return predictions
    
    def _load_ground_truth(self) -> dict:
        """Load ground truth ages."""
        ground_truth = {}
        
        for subject_dir in self.get_subject_dirs():
            label_file = self.get_labels_path(subject_dir.name) / "label.txt"
            if label_file.exists():
                try:
                    with open(label_file, 'r') as f:
                        age_str = f.read().strip()
                        # Parse age (could be float in file)
                        age_float = float(age_str)
                        # Round to nearest integer as specified in requirements
                        age = round(age_float)
                        
                        # Validate age range
                        if 0 <= age <= 150:
                            ground_truth[subject_dir.name] = float(age)
                        else:
                            self.logger.warning(f"Unrealistic ground truth age for {subject_dir.name}: {age}")
                            
                except (ValueError, IOError) as e:
                    self.logger.warning(f"Could not read age label for {subject_dir.name}: {e}")
                    continue
        
        self.logger.info(f"Loaded ground truth ages for {len(ground_truth)} subjects")
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
            # Extract subject ID from header: "sub_1_t1" -> "sub_1"
            subject_id = self._extract_subject_id(header)
            
            if subject_id and subject_id in ground_truth:
                # Use the same ground truth age for this modality
                gt_value = ground_truth[subject_id]
                
                aligned_preds.append(pred_value)
                aligned_gt.append(gt_value)
                matched_cases.append(header)
                
                self.logger.debug(f"Matched: {header} (pred={pred_value:.1f}) -> {subject_id} (gt={gt_value:.1f})")
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
    
    def validate_age_predictions(self, output_path: Path) -> bool:
        """Validate that age predictions are properly formatted."""
        try:
            predictions = self._load_predictions(output_path)
            if not predictions:
                return False
            
            # Check age ranges
            ages = list(predictions.values())
            min_age, max_age = min(ages), max(ages)
            
            self.logger.info("Age prediction validation:")
            self.logger.info(f"  Number of predictions: {len(predictions)}")
            self.logger.info(f"  Age range: [{min_age:.1f}, {max_age:.1f}]")
            
            # Check for reasonable age range
            if min_age < 0 or max_age > 150:
                self.logger.warning("Some ages outside reasonable range [0, 150]")
            
            # Check for missing subjects
            expected_subjects = set(sub.name for sub in self.get_subject_dirs())
            predicted_subjects = set(self._extract_subject_id(header) for header in predictions.keys() if self._extract_subject_id(header))
            
            missing = expected_subjects - predicted_subjects
            extra = predicted_subjects - expected_subjects
            
            if missing:
                self.logger.warning(f"Missing predictions for subjects: {missing}")
            if extra:
                self.logger.warning(f"Extra predictions for unknown subjects: {extra}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating age predictions: {e}")
            return False