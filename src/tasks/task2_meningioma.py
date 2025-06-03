# src/tasks/task2_meningioma.py - VERSION CSV METRICS
"""Task 2: Meningioma Segmentation - Genera CSV con métricas por sujeto."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

from src.tasks.base_task import BaseTask
from src.config.settings import SETTINGS
from src.metrics.segmentation import compute_dice, compute_nsd
from src.utils.logging_utils import get_logger


class MeningiomaSegmentationTask(BaseTask):
    """Meningioma segmentation con evaluación CSV por sujeto."""
    
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
    
    def evaluate(self, output_path: Path, task_output_dir: Path = None) -> Dict[str, Any]:
        """
        Evaluación que genera CSV con métricas por sujeto.
        
        Flujo:
        1. Busca todas las máscaras predichas (.nii.gz)
        2. Para cada máscara, encuentra su ground truth correspondiente
        3. Computa métricas (Dice, NSD) para cada par
        4. Guarda CSV con métricas por sujeto
        5. Calcula promedios para el JSON final
        """
        self.logger.info(f"🔍 TASK 2 EVALUATION - CSV Metrics per Subject")
        self.logger.info(f"   Output path: {output_path}")
        self.logger.info(f"   Task output dir: {task_output_dir}")
        
        # Buscar segmentaciones predichas
        prediction_files = self._find_prediction_files(output_path, task_output_dir)
        
        if not prediction_files:
            self.logger.error("❌ No prediction files found")
            return {"dice": SETTINGS.DSC_WORST, "nsd": SETTINGS.NSD_WORST}
        
        self.logger.info(f"📁 Found {len(prediction_files)} prediction files")
        
        # Evaluar cada predicción contra su ground truth
        subject_metrics = []
        
        for pred_file, subject_id in prediction_files:
            self.logger.info(f"🎯 Evaluating {subject_id}: {pred_file.name}")
            
            # Cargar predicción
            pred_seg = self._load_segmentation(pred_file)
            if pred_seg is None:
                self.logger.warning(f"   ❌ Failed to load prediction for {subject_id}")
                continue
            
            # Cargar ground truth
            gt_seg = self._load_ground_truth_segmentation(subject_id)
            if gt_seg is None:
                self.logger.warning(f"   ❌ No ground truth found for {subject_id}")
                continue
            
            # Computar métricas
            dice = compute_dice(gt_seg, pred_seg)
            nsd = compute_nsd(gt_seg, pred_seg)
            
            # Guardar métricas del sujeto
            subject_metrics.append({
                'subject_id': subject_id,
                'dice': dice,
                'nsd': nsd,
                'prediction_file': pred_file.name,
                'gt_positive_voxels': int(np.sum(gt_seg)),
                'pred_positive_voxels': int(np.sum(pred_seg)),
                'intersection_voxels': int(np.sum(gt_seg * pred_seg))
            })
            
            self.logger.info(f"   📊 Dice={dice:.4f}, NSD={nsd:.4f}")
        
        if not subject_metrics:
            self.logger.error("❌ No subjects could be evaluated")
            return {"dice": SETTINGS.DSC_WORST, "nsd": SETTINGS.NSD_WORST}
        
        # Guardar CSV con métricas por sujeto
        csv_results = self._save_subject_metrics_csv(subject_metrics, task_output_dir)
        
        # Calcular métricas promedio para JSON
        final_metrics = self._compute_average_metrics(subject_metrics)
        
        self.logger.info(f"📈 FINAL RESULTS (Simple Format):")
        self.logger.info(f"   Evaluated subjects: {len(subject_metrics)}")
        self.logger.info(f"   Average Dice: {final_metrics['dice']:.6f}")
        self.logger.info(f"   Average NSD: {final_metrics['nsd']:.6f}")
        self.logger.info(f"   CSV saved: {csv_results}")
        
        return final_metrics
    
    def _find_prediction_files(self, output_path: Path, task_output_dir: Path) -> List[tuple]:
        """
        Encuentra todos los archivos de predicción y extrae subject_ids.
        
        Returns:
            List of (file_path, subject_id) tuples
        """
        prediction_files = []
        
        # Opción 1: Buscar en directorio organizado (segmentations/)
        if task_output_dir and (task_output_dir / "segmentations").exists():
            seg_dir = task_output_dir / "segmentations"
            for seg_file in seg_dir.glob("*.nii.gz"):
                subject_id = self._extract_subject_from_filename(seg_file.name)
                if subject_id:
                    prediction_files.append((seg_file, subject_id))
            self.logger.info(f"   Found {len(prediction_files)} files in segmentations/ directory")
        
        # Opción 2: Buscar archivos individuales en task_output_dir
        if task_output_dir and len(prediction_files) == 0:
            for nii_file in task_output_dir.glob("*.nii.gz"):
                subject_id = self._extract_subject_from_filename(nii_file.name)
                if subject_id:
                    prediction_files.append((nii_file, subject_id))
            self.logger.info(f"   Found {len(prediction_files)} files in task output directory")
        
        # Opción 3: Usar el archivo único (legacy)
        if len(prediction_files) == 0 and output_path.exists():
            # Si no hay subject_id específico, evaluar contra todos los subjects disponibles
            self.logger.info("   Using single prediction file against all subjects")
            available_subjects = self._get_available_subjects()
            for subject_id in available_subjects:
                prediction_files.append((output_path, subject_id))
        
        return prediction_files
    
    def _get_available_subjects(self) -> List[str]:
        """Obtener lista de subjects disponibles en el dataset."""
        preprocessed_dir = self.data_dir / "preprocessed"
        if not preprocessed_dir.exists():
            return []
        
        subjects = []
        for subject_dir in preprocessed_dir.iterdir():
            if subject_dir.is_dir() and subject_dir.name.startswith('sub_'):
                subjects.append(subject_dir.name)
        
        subjects.sort()
        return subjects
    
    def _extract_subject_from_filename(self, filename: str) -> str:
        """Extraer subject_id del nombre del archivo."""
        import re
        
        # Buscar patrón sub_X
        match = re.search(r'(sub_\d+)', filename)
        if match:
            return match.group(1)
        
        # Si no encuentra sub_X, retornar None para que use evaluación legacy
        return None
    
    def _save_subject_metrics_csv(self, subject_metrics: List[dict], task_output_dir: Path) -> str:
        """
        Guardar métricas por sujeto en CSV.
        
        Formato CSV:
        subject_id,dice,nsd,prediction_file,gt_positive_voxels,pred_positive_voxels,intersection_voxels
        sub_1,0.8456,0.1234,sub_1_segmentation.nii.gz,1247,1456,1089
        sub_2,0.7234,0.2345,sub_2_segmentation.nii.gz,892,1123,734
        """
        # Crear DataFrame
        df = pd.DataFrame(subject_metrics)
        
        # Guardar CSV
        if task_output_dir:
            csv_path = task_output_dir / "subject_metrics.csv"
        else:
            csv_path = Path("subject_metrics.csv")
        
        df.to_csv(csv_path, index=False, float_format='%.6f')
        
        self.logger.info(f"💾 Saved subject metrics CSV: {csv_path}")
        self.logger.info(f"   Columns: {list(df.columns)}")
        self.logger.info(f"   Rows: {len(df)}")
        
        return str(csv_path)
    
    def _compute_average_metrics(self, subject_metrics: List[dict]) -> Dict[str, Any]:
        """Calcular métricas promedio para el JSON final - formato simple."""
        if not subject_metrics:
            return {"dice": SETTINGS.DSC_WORST, "nsd": SETTINGS.NSD_WORST}
        
        dice_scores = [m['dice'] for m in subject_metrics]
        nsd_scores = [m['nsd'] for m in subject_metrics]
        
        # Calcular solo los promedios (formato simple requerido)
        mean_dice = np.mean(dice_scores)
        mean_nsd = np.mean(nsd_scores)
        
        self.logger.info(f"📊 Individual Dice scores: {[f'{d:.6f}' for d in dice_scores]}")
        self.logger.info(f"📊 Individual NSD scores: {[f'{n:.6f}' for n in nsd_scores]}")
        
        return {
            "dice": float(mean_dice),
            "nsd": float(mean_nsd)
        }
    
    def _load_segmentation(self, seg_path: Path):
        """Cargar segmentación desde archivo NIfTI."""
        try:
            import nibabel as nib
            nii = nib.load(seg_path)
            data = nii.get_fdata()
            
            # Asegurar que sea binaria
            binary_data = (data > 0.5).astype(np.uint8)
            
            self.logger.debug(f"   ✅ Loaded prediction: shape={binary_data.shape}, positives={np.sum(binary_data)}")
            return binary_data
            
        except Exception as e:
            self.logger.error(f"   ❌ Error loading segmentation {seg_path}: {e}")
            return None
    
    def _load_ground_truth_segmentation(self, subject_id: str):
        """Cargar ground truth segmentation para un sujeto específico."""
        try:
            gt_path = self.get_labels_path(subject_id) / "seg.nii.gz"
            
            if not gt_path.exists():
                self.logger.debug(f"   ❌ GT file not found: {gt_path}")
                return None
            
            import nibabel as nib
            nii = nib.load(gt_path)
            data = nii.get_fdata()
            
            # Asegurar que sea binaria
            binary_data = (data > 0.5).astype(np.uint8)
            
            self.logger.debug(f"   ✅ Loaded GT: shape={binary_data.shape}, positives={np.sum(binary_data)}")
            return binary_data
            
        except Exception as e:
            self.logger.error(f"   ❌ Error loading GT for {subject_id}: {e}")
            return None