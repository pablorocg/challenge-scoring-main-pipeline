"""Task 2: Meningioma Segmentation."""

from pathlib import Path
from typing import Dict, Any

from .base_task import BaseTask
from config.settings import SETTINGS
from metrics.segmentation import compute_dice, compute_nsd


class MeningiomaSegmentationTask(BaseTask):
    """Meningioma segmentation task."""
    
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
        return ["modality"]
    
    def evaluate(self, output_path: Path) -> Dict[str, Any]:
        """Evaluate segmentation predictions."""
        if not output_path.exists():
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST
            }
        
        # Note: Ground truth segmentations are currently empty
        # This is a placeholder for when labels become available
        
        try:
            # Load prediction and ground truth segmentations
            pred_seg = self._load_segmentation(output_path)
            gt_seg = self._load_ground_truth_segmentation()
            
            if pred_seg is None or gt_seg is None:
                return {
                    "dice": SETTINGS.DSC_WORST,
                    "nsd": SETTINGS.NSD_WORST
                }
            
            # Compute metrics
            dice = compute_dice(gt_seg, pred_seg)
            nsd = compute_nsd(gt_seg, pred_seg)
            
            return {
                "dice": dice,
                "nsd": nsd,
                "num_cases": 1
            }
            
        except Exception:
            return {
                "dice": SETTINGS.DSC_WORST,
                "nsd": SETTINGS.NSD_WORST
            }
    
    def _load_segmentation(self, seg_path: Path):
        """Load segmentation from NIfTI file."""
        try:
            import nibabel as nib
            nii = nib.load(seg_path)
            return nii.get_fdata()
        except Exception:
            return None
    
    def _load_ground_truth_segmentation(self):
        """Load ground truth segmentation."""
        # Currently empty labels - return None for now
        # When available, segmentations will be in:
        # self.data_dir / "labels" / subject_id / "seg.nii.gz"
        return None
