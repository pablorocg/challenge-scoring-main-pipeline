# src/runners/apptainer_runner.py
"""Optimized Apptainer container runner - simplified and efficient."""

import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import contextmanager

from src.config.settings import SETTINGS
from src.config.task_config import get_task_config
from src.utils.logging_utils import get_logger


class ApptainerRunner:
    """Simplified container runner for medical imaging inference."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def run_inference(self, container_path: Path, task, output_path: Path, task_output_dir: Path) -> bool:
        """Run inference on all subjects."""
        subject_dirs = task.get_subject_dirs()
        
        if not subject_dirs:
            self.logger.error("No subject directories found")
            return False
        
        self.logger.info(f"Processing {len(subject_dirs)} subjects")
        
        subject_outputs = []
        for subject_dir in subject_dirs:
            output_file = self._process_subject(container_path, subject_dir, task, task_output_dir)
            if output_file:
                subject_outputs.append(output_file)
                self.logger.info(f"✅ {subject_dir.name}")
            else:
                self.logger.error(f"❌ {subject_dir.name}")
        
        if not subject_outputs:
            return False
        
        # Handle final output based on task type
        if task.name == "Meningioma Segmentation":
            # For segmentation, individual subject files are sufficient
            self.logger.info(f"Completed segmentation for {len(subject_outputs)} subjects")
            return True
        else:
            # For CSV tasks (classification/regression), combine outputs
            return self._combine_csv_outputs(subject_outputs, output_path)
    
    def _process_subject(self, container_path: Path, subject_dir: Path, task, task_output_dir: Path) -> Optional[Path]:
        """Process a single subject."""
        modalities = self._get_modalities(subject_dir, task.name)
        if not modalities:
            return None
        
        output_file = task_output_dir / f"{subject_dir.name}_output{task.output_extension}"
        instance_name = f"fomo_{container_path.stem}_{subject_dir.name}_{int(time.time())}"
        
        with self._container_instance(container_path, subject_dir, instance_name, task_output_dir) as instance:
            if instance and self._run_inference(instance, modalities, output_file):
                return output_file
        return None
    
    def _get_modalities(self, subject_dir: Path, task_name: str) -> List[Tuple[Path, str]]:
        """Get available modalities for a subject."""
        ses_dir = subject_dir / "ses_1"
        if not ses_dir.exists():
            return []
        
        config = get_task_config(task_name)
        modalities = []
        
        # Add required modalities
        for modality in config.required:
            modality_file = ses_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                modalities.append((modality_file, modality))
        
        # Add first available optional modality
        for modality in config.optional:
            modality_file = ses_dir / f"{modality}.nii.gz"
            if modality_file.exists():
                modalities.append((modality_file, modality))
                break
        
        return modalities
    
    @contextmanager
    def _container_instance(self, container_path: Path, subject_dir: Path, instance_name: str, task_output_dir: Path):
        """Manage container instance."""
        try:
            if self._start_instance(container_path, subject_dir, instance_name, task_output_dir):
                yield instance_name
            else:
                yield None
        finally:
            self._stop_instance(instance_name)
    
    def _start_instance(self, container_path: Path, subject_dir: Path, instance_name: str, task_output_dir: Path) -> bool:
        """Start container instance."""
        ses_dir = subject_dir / "ses_1"
        task_output_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            SETTINGS.APPTAINER_EXECUTABLE, "instance", "start",
            "--bind", f"{ses_dir}:/input:ro",
            "--bind", f"{task_output_dir}:/output:rw"
        ]
        
        if SETTINGS.ENABLE_GPU:
            cmd.append("--nv")
        
        cmd.extend([str(container_path), instance_name])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SETTINGS.INSTANCE_START_TIMEOUT)
        return result.returncode == 0
    
    def _stop_instance(self, instance_name: str):
        """Stop container instance."""
        subprocess.run([SETTINGS.APPTAINER_EXECUTABLE, "instance", "stop", instance_name],
                      capture_output=True, timeout=SETTINGS.INSTANCE_STOP_TIMEOUT)
    
    def _run_inference(self, instance_name: str, modalities: List[Tuple[Path, str]], output_file: Path) -> bool:
        """Execute inference command."""
        cmd = [
            SETTINGS.APPTAINER_EXECUTABLE, "exec", f"instance://{instance_name}",
            "python", SETTINGS.PYTHON_SCRIPT
        ]
        
        # Add modality arguments
        for modality_file, modality in modalities:
            cmd.extend([f"--{modality}", f"/input/{modality_file.name}"])
        
        cmd.extend(["--output", f"/output/{output_file.name}"])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=SETTINGS.CONTAINER_TIMEOUT)
        
        success = result.returncode == 0 and output_file.exists()
        if not success:
            self.logger.error(f"Inference failed: {result.stderr or 'Output not created'}")
        
        return success
    
    def _combine_csv_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Combine CSV outputs from multiple subjects."""
        import pandas as pd
        
        valid_outputs = [f for f in outputs if f.exists()]
        if not valid_outputs:
            return False
        
        combined_data = []
        for output_file in valid_outputs:
            df = pd.read_csv(output_file)
            subject_id = output_file.stem.replace('_output', '')
            df['header'] = subject_id
            combined_data.append(df)
        
        final_df = pd.concat(combined_data, ignore_index=True)
        final_df.to_csv(final_output, index=False)
        self.logger.info(f"Combined {len(valid_outputs)} predictions")
        return True