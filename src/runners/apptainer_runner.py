# src/runners/apptainer_runner.py
"""Simplified Apptainer container runner - one prediction per subject using all modalities."""

import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict
from contextlib import contextmanager

from src.config.settings import SETTINGS
from src.utils.logging_utils import get_logger


class ApptainerRunner:
    """Runs inference using all available modalities per subject."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def run_inference(self, container_path: Path, task, output_path: Path, task_output_dir: Path) -> bool:
        """Run inference processing each subject with all their modalities at once."""
        subject_dirs = task.get_subject_dirs()
        
        if not subject_dirs:
            self.logger.error("No subject directories found")
            return False
        
        self.logger.info(f"Processing {len(subject_dirs)} subjects with {container_path.name}")
        
        success_count = 0
        all_outputs = []
        
        for subject_dir in subject_dirs:
            subject_output = self._process_subject_with_all_modalities(
                container_path, subject_dir, task, task_output_dir
            )
            
            if subject_output:
                all_outputs.append(subject_output)
                success_count += 1
                self.logger.info(f"✅ {subject_dir.name} processed successfully")
            else:
                self.logger.error(f"❌ {subject_dir.name} failed")
        
        if success_count > 0:
            return self._create_final_output(all_outputs, output_path, task)
        
        return False
    
    def _process_subject_with_all_modalities(self, container_path: Path, subject_dir: Path, 
                                           task, task_output_dir: Path) -> Path:
        """Process one subject using ALL available modalities in a single prediction."""
        # Get all available modalities for this subject
        modalities = self._get_subject_modalities(subject_dir, task)
        
        if not modalities:
            self.logger.warning(f"No modalities found for {subject_dir.name}")
            return None
        
        self.logger.info(f"Processing {subject_dir.name} with modalities: {[m[1] for m in modalities]}")
        
        # Create output file for this subject
        output_file = task_output_dir / f"{subject_dir.name}_output{task.output_extension}"
        
        # Use container instance to process this subject
        instance_name = f"fomo_{container_path.stem}_{subject_dir.name}_{int(time.time())}"
        
        with self._container_instance(container_path, subject_dir, instance_name, task_output_dir) as instance:
            if not instance:
                return None
            
            # Build command with ALL modalities
            if self._run_multi_modality_inference(instance_name, modalities, output_file):
                return output_file
        
        return None
    
    def _get_subject_modalities(self, subject_dir: Path, task) -> List[Tuple[Path, str]]:
        """Get all available modalities for a specific subject."""
        ses_dir = subject_dir / "ses_1"
        if not ses_dir.exists():
            return []
        
        modalities = []
        
        # Handle each task's specific modality requirements
        if task.name == "Infarct Classification":
            # Core modalities: flair, adc, dwi_b1000
            for modality in ["flair", "adc", "dwi_b1000"]:
                modality_file = ses_dir / f"{modality}.nii.gz"
                if modality_file.exists():
                    modalities.append((modality_file, modality))
            
            # Either t2s OR swi
            t2s_file = ses_dir / "t2s.nii.gz"
            swi_file = ses_dir / "swi.nii.gz"
            
            if t2s_file.exists():
                modalities.append((t2s_file, "t2s"))
            elif swi_file.exists():
                modalities.append((swi_file, "swi"))
                
        elif task.name == "Meningioma Segmentation":
            # Core modalities: flair, dwi_b1000
            for modality in ["flair", "dwi_b1000"]:
                modality_file = ses_dir / f"{modality}.nii.gz"
                if modality_file.exists():
                    modalities.append((modality_file, modality))
            
            # Either t2s OR swi
            t2s_file = ses_dir / "t2s.nii.gz"
            swi_file = ses_dir / "swi.nii.gz"
            
            if t2s_file.exists():
                modalities.append((t2s_file, "t2s"))
            elif swi_file.exists():
                modalities.append((swi_file, "swi"))
                
        elif task.name == "Brain Age Prediction":
            # T1 and T2
            for modality in ["t1", "t2"]:
                modality_file = ses_dir / f"{modality}.nii.gz"
                if modality_file.exists():
                    modalities.append((modality_file, modality))
        
        return modalities
    
    @contextmanager
    def _container_instance(self, container_path: Path, subject_dir: Path, 
                           instance_name: str, task_output_dir: Path):
        """Context manager for container instance."""
        try:
            if self._start_instance(container_path, subject_dir, instance_name, task_output_dir):
                yield instance_name
            else:
                yield None
        finally:
            self._stop_instance(instance_name)
    
    def _start_instance(self, container_path: Path, subject_dir: Path, 
                       instance_name: str, task_output_dir: Path) -> bool:
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
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=SETTINGS.INSTANCE_START_TIMEOUT)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, Exception):
            return False
    
    def _stop_instance(self, instance_name: str):
        """Stop container instance."""
        try:
            subprocess.run([SETTINGS.APPTAINER_EXECUTABLE, "instance", "stop", instance_name],
                         capture_output=True, timeout=SETTINGS.INSTANCE_STOP_TIMEOUT)
        except:
            pass  # Best effort cleanup
    
    def _run_multi_modality_inference(self, instance_name: str, modalities: List[Tuple[Path, str]], 
                                    output_file: Path) -> bool:
        """Run inference using ALL modalities at once."""
        cmd = [
            SETTINGS.APPTAINER_EXECUTABLE, "exec", f"instance://{instance_name}",
            "python", SETTINGS.PYTHON_SCRIPT
        ]
        
        # Add each modality as a flag with its file path
        for modality_file, modality in modalities:
            cmd.extend([f"--{modality}", f"/input/{modality_file.name}"])
        
        # Add output
        cmd.extend(["--output", f"/output/{output_file.name}"])
        
        self.logger.info(f"Running: {' '.join(cmd[3:])}")  # Log only the python command part
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=SETTINGS.CONTAINER_TIMEOUT)
            
            if result.returncode == 0 and output_file.exists():
                return True
            else:
                self.logger.error(f"Inference failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Inference timed out")
            return False
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return False
    
    def _create_final_output(self, subject_outputs: List[Path], final_output: Path, task) -> bool:
        """Create final aggregated output from all subjects."""
        try:
            if task.name in ["Infarct Classification", "Brain Age Prediction"]:
                return self._combine_csv_outputs(subject_outputs, final_output)
            elif task.name == "Meningioma Segmentation":
                return self._organize_segmentation_outputs(subject_outputs, final_output)
            else:
                # Default: copy first output
                if subject_outputs:
                    import shutil
                    shutil.copy(subject_outputs[0], final_output)
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error creating final output: {e}")
            return False
    
    def _combine_csv_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Combine CSV outputs from all subjects."""
        import pandas as pd
        
        all_data = []
        for output_file in outputs:
            df = pd.read_csv(output_file)
            # Extract subject ID from filename
            subject_id = output_file.stem.replace('_output', '')
            df['header'] = subject_id  # Overwrite or set header to subject ID
            all_data.append(df)
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            final_df.to_csv(final_output, index=False)
            self.logger.info(f"Combined {len(outputs)} subject predictions into {final_output}")
            return True
        
        return False
    
    def _organize_segmentation_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Organize segmentation outputs."""
        # For segmentation, we can just use the first output as representative
        # or create a summary. This depends on evaluation requirements.
        if outputs:
            import shutil
            shutil.copy(outputs[0], final_output)
            
            # Also create a summary of all segmentations
            seg_dir = final_output.parent / "segmentations"
            seg_dir.mkdir(exist_ok=True)
            
            for output_file in outputs:
                subject_id = output_file.stem.replace('_output', '')
                dest_file = seg_dir / f"{subject_id}_segmentation.nii.gz"
                shutil.copy(output_file, dest_file)
            
            self.logger.info(f"Organized {len(outputs)} segmentations")
            return True
        
        return False