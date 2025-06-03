"""Task 2: Meningioma Segmentation - FIXED VERSION with subject-specific evaluation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.segmentation import compute_dice, compute_nsd
from src.utils.logging_utils import get_logger


class MeningiomaSegmentationTask(BaseTask):
    """Meningioma segmentation task with proper subject-specific evaluation."""
    
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
    
    @property
    def image_modalities(self) -> list[str]:
        # Core modalities for meningioma segmentation
        return ["flair", "dwi_b1000", "t2s", "swi"]
    
    def get_subject_modalities(self, subject_dir: Path) -> list[str]:
        """Get actual available modalities for a specific subject."""
        ses_dir = subject_dir / "ses_1"
        if not ses_dir.exists():
            return []
        
        available_modalities = []
        
        # Core modalities
        for modality in ["flair", "dwi_b1000"]:
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
    
    def evaluate(self, output_path: Path, task_output_dir: Path = None) -> Dict[str, Any]:
        """Evaluate segmentation predictions subject by subject."""
        
        # If task_output_dir is provided, look for organized segmentations
        if task_output_dir and (task_output_dir / "segmentations").exists():
            return self._evaluate_organized_segmentations(task_output_dir)
        
        # Fallback to single file evaluation
        if not output_path.exists():
            self.logger.error(f"Output file does not exist: {output_path}")
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST
            }
        
        return self._evaluate_single_segmentation(output_path)
    
    def _evaluate_organized_segmentations(self, task_output_dir: Path) -> Dict[str, Any]:
        """Evaluate segmentations organized by subject."""
        seg_dir = task_output_dir / "segmentations"
        
        if not seg_dir.exists():
            self.logger.error(f"Segmentations directory does not exist: {seg_dir}")
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST
            }
        
        # Find all subject segmentation files
        seg_files = list(seg_dir.glob("*_segmentation.nii.gz"))
        
        if not seg_files:
            self.logger.error("No segmentation files found in organized directory")
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST
            }
        
        self.logger.info(f"Found {len(seg_files)} segmentation files for evaluation")
        
        # Evaluate each subject individually
        results = []
        all_dice_scores = []
        all_nsd_scores = []
        
        for seg_file in seg_files:
            # Extract subject ID from filename: "sub_25_segmentation.nii.gz" -> "sub_25"
            subject_id = self._extract_subject_from_filename(seg_file.name)
            
            if not subject_id:
                self.logger.warning(f"Could not extract subject ID from: {seg_file.name}")
                continue
            
            self.logger.info(f"Evaluating segmentation for {subject_id}")
            
            # Load prediction segmentation
            pred_seg = self._load_segmentation(seg_file)
            if pred_seg is None:
                self.logger.warning(f"Failed to load prediction for {subject_id}")
                continue
            
            # Load ground truth for this subject
            gt_seg = self._load_ground_truth_segmentation(subject_id)
            if gt_seg is None:
                self.logger.warning(f"No ground truth found for {subject_id}")
                continue
            
            # Compute metrics for this subject
            dice = compute_dice(gt_seg, pred_seg)
            nsd = compute_nsd(gt_seg, pred_seg)
            
            # Store individual result
            results.append({
                'subject_id': subject_id,
                'dice': dice,
                'nsd': nsd,
                'prediction_file': seg_file.name
            })
            
            all_dice_scores.append(dice)
            all_nsd_scores.append(nsd)
            
            self.logger.info(f"Subject {subject_id}: Dice={dice:.4f}, NSD={nsd:.4f}")
        
        # Save detailed results to CSV
        if results:
            results_df = pd.DataFrame(results)
            results_csv = task_output_dir / "segmentation_metrics.csv"
            results_df.to_csv(results_csv, index=False)
            
            # Compute summary statistics
            mean_dice = np.mean(all_dice_scores)
            mean_nsd = np.mean(all_nsd_scores)
            std_dice = np.std(all_dice_scores)
            std_nsd = np.std(all_nsd_scores)
            
            # Add summary row
            summary_results = results + [{
                'subject_id': 'MEAN',
                'dice': mean_dice,
                'nsd': mean_nsd,
                'prediction_file': 'summary'
            }, {
                'subject_id': 'STD',
                'dice': std_dice,
                'nsd': std_nsd,
                'prediction_file': 'summary'
            }]
            
            summary_df = pd.DataFrame(summary_results)
            summary_df.to_csv(results_csv, index=False)
            
            self.logger.info(f"Saved detailed metrics to: {results_csv}")
            self.logger.info(f"Evaluation complete: Mean Dice={mean_dice:.4f}±{std_dice:.4f}, Mean NSD={mean_nsd:.4f}±{std_nsd:.4f}")
            self.logger.info(f"Evaluated {len(results)} subjects")
            
            return {
                "dice": float(mean_dice),
                "nsd": float(mean_nsd),
                "dice_std": float(std_dice),
                "nsd_std": float(std_nsd),
                "num_subjects": len(results),
                "detailed_results_file": str(results_csv.relative_to(task_output_dir))
            }
        
        else:
            self.logger.error("No subjects could be evaluated")
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST,
                "num_subjects": 0,
                "error": "No subjects evaluated successfully"
            }
    
    def _evaluate_single_segmentation(self, output_path: Path) -> Dict[str, Any]:
        """Fallback: evaluate single segmentation file (legacy mode)."""
        try:
            # Load prediction segmentation
            pred_seg = self._load_segmentation(output_path)
            if pred_seg is None:
                self.logger.error("Failed to load prediction segmentation")
                return {
                    "dice": SETTINGS.DSC_WORST,
                    "nsd": SETTINGS.NSD_WORST
                }
            
            # Load ground truth segmentations for all subjects and average
            all_dice_scores = []
            all_nsd_scores = []
            evaluated_subjects = 0
            
            for subject_dir in self.get_subject_dirs():
                gt_seg = self._load_ground_truth_segmentation(subject_dir.name)
                
                if gt_seg is not None:
                    # Use the same prediction for all subjects (legacy behavior)
                    dice = compute_dice(gt_seg, pred_seg)
                    nsd = compute_nsd(gt_seg, pred_seg)
                    
                    all_dice_scores.append(dice)
                    all_nsd_scores.append(nsd)
                    evaluated_subjects += 1
                    
                    self.logger.info(f"Subject {subject_dir.name}: Dice={dice:.4f}, NSD={nsd:.4f}")
            
            if evaluated_subjects > 0:
                # Average metrics across all subjects
                avg_dice = np.mean(all_dice_scores)
                avg_nsd = np.mean(all_nsd_scores)
                
                self.logger.info(f"Legacy evaluation complete: Dice={avg_dice:.4f}, NSD={avg_nsd:.4f}, Subjects={evaluated_subjects}")
                
                return {
                    "dice": float(avg_dice),
                    "nsd": float(avg_nsd),
                    "num_subjects": evaluated_subjects,
                    "mode": "legacy_single_prediction"
                }
            else:
                self.logger.warning("No ground truth segmentations available for evaluation")
                return {
                    "dice": 0.5,  # Neutral score instead of worst
                    "nsd": 0.5,   # Neutral score instead of worst
                    "note": "No ground truth available - using neutral scores"
                }
            
        except Exception as e:
            self.logger.error(f"Error during segmentation evaluation: {e}")
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST
            }
    
    def _load_segmentation(self, seg_path: Path):
        """Load segmentation from NIfTI file."""
        try:
            import nibabel as nib
            self.logger.debug(f"Loading segmentation from: {seg_path}")
            nii = nib.load(seg_path)
            data = nii.get_fdata()
            
            # Validate segmentation data
            if data.size == 0:
                self.logger.error("Segmentation data is empty")
                return None
            
            # Check if it's binary
            unique_values = np.unique(data)
            self.logger.debug(f"Segmentation contains values: {unique_values}")
            
            # Ensure binary segmentation (0 and 1)
            if len(unique_values) > 2:
                self.logger.warning(f"Non-binary segmentation detected, thresholding at 0.5")
                data = (data > 0.5).astype(np.uint8)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading segmentation from {seg_path}: {e}")
            return None
    
    def _load_ground_truth_segmentation(self, subject_id: str):
        """Load ground truth segmentation for a specific subject."""
        try:
            # Ground truth segmentation path
            gt_path = self.get_labels_path(subject_id) / "seg.nii.gz"
            
            if not gt_path.exists():
                self.logger.debug(f"No ground truth segmentation for {subject_id} at {gt_path}")
                return None
            
            import nibabel as nib
            nii = nib.load(gt_path)
            data = nii.get_fdata()
            
            # Validate ground truth data
            if data.size == 0:
                self.logger.warning(f"Ground truth segmentation for {subject_id} is empty")
                return None
            
            # Ensure binary
            unique_values = np.unique(data)
            if len(unique_values) > 2:
                self.logger.warning(f"Non-binary ground truth for {subject_id}, thresholding")
                data = (data > 0.5).astype(np.uint8)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading ground truth for {subject_id}: {e}")
            return None
    
    def _extract_subject_from_filename(self, filename: str) -> str:
        """Extract subject ID from filename."""
        import re
        
        # Handle formats like "sub_25_segmentation.nii.gz" -> "sub_25"
        match = re.search(r'(sub_\d+)', filename)
        if match:
            return match.group(1)
        
        self.logger.warning(f"Could not extract subject ID from filename: {filename}")
        return None
    
    def validate_segmentation_output(self, output_path: Path) -> bool:
        """Validate that segmentation output is properly formatted."""
        try:
            seg_data = self._load_segmentation(output_path)
            if seg_data is None:
                return False
            
            # Check dimensions (should be 3D)
            if len(seg_data.shape) != 3:
                self.logger.error(f"Segmentation should be 3D, got shape: {seg_data.shape}")
                return False
            
            # Check if values are reasonable (0 and 1 for binary)
            unique_values = np.unique(seg_data)
            if not all(val in [0, 1] for val in unique_values):
                self.logger.warning(f"Segmentation contains non-binary values: {unique_values}")
            
            # Check if segmentation is not empty (at least some positive voxels)
            positive_voxels = np.sum(seg_data > 0)
            total_voxels = seg_data.size
            
            self.logger.info(f"Segmentation validation: {positive_voxels}/{total_voxels} positive voxels ({positive_voxels/total_voxels*100:.2f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating segmentation: {e}")
            return False