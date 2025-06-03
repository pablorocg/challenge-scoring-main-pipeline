# src/runners/apptainer_runner.py
"""Apptainer container runner with instance support."""

import subprocess
import time
from pathlib import Path
from typing import List, Tuple
from contextlib import contextmanager

from src.config.settings import SETTINGS
from src.utils.logging_utils import get_logger


class ApptainerRunner:
    """Runs inference using Apptainer container instances."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def run_inference(self, container_path: Path, task, output_path: Path) -> bool:
        """Run inference using an Apptainer container instance for all modalities."""
        self.logger.info(f"Running inference with {container_path}")
        
        # Get all input files for this task
        input_files = self._get_input_files(task)
        
        if not input_files:
            self.logger.error("No input files found for task")
            return False
        
        # Create instance name based on container and timestamp
        instance_name = f"fomo_{container_path.stem}_{int(time.time())}"
        
        # Use context manager for instance lifecycle
        try:
            with self._container_instance(container_path, input_files, output_path, instance_name) as instance:
                if not instance:
                    return False
                
                # Run inference for each modality/file combination
                all_outputs = []
                success_count = 0
                
                for input_file, modality in input_files:
                    self.logger.info(f"Processing {modality}: {input_file}")
                    
                    # Create individual output file for this modality
                    modality_output = output_path.parent / f"{output_path.stem}_{modality}{output_path.suffix}"
                    
                    # Run inference on this file using the instance
                    if self._run_instance_inference(instance_name, modality, input_file, modality_output):
                        all_outputs.append(modality_output)
                        success_count += 1
                        self.logger.info(f"✅ Successfully processed {modality}: {input_file}")
                    else:
                        self.logger.error(f"❌ Failed to process {modality}: {input_file}")
                
                # Aggregate results if needed
                if success_count > 0:
                    result = self._aggregate_outputs(all_outputs, output_path, task)
                    self.logger.info(f"Processing Summary: {success_count}/{len(input_files)} files successful")
                    return result
                
                self.logger.error("All inference attempts failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            return False
    
    @contextmanager
    def _container_instance(self, container_path: Path, input_files: List[Tuple[Path, str]], 
                           output_path: Path, instance_name: str):
        """Context manager for container instance lifecycle."""
        instance = None
        try:
            # Determine unique input and output directories for binding
            input_dirs = set(input_file.parent for input_file, _ in input_files)
            output_dir = output_path.parent
            
            # Start container instance
            self.logger.info(f"🚀 Starting container instance: {instance_name}")
            
            if self._start_instance(container_path, input_dirs, output_dir, instance_name):
                self.logger.info(f"✅ Container instance started successfully")
                instance = instance_name
                yield instance
            else:
                self.logger.error(f"❌ Failed to start container instance")
                yield None
                
        finally:
            # Cleanup: stop instance
            if instance:
                self._stop_instance(instance_name)
    
    def _start_instance(self, container_path: Path, input_dirs: set, output_dir: Path, 
                       instance_name: str) -> bool:
        """Start container instance with appropriate binds."""
        try:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build command with all necessary binds
            cmd = [
                SETTINGS.APPTAINER_EXECUTABLE,
                "instance", "start"
            ]
            
            # Add GPU support if enabled
            if SETTINGS.ENABLE_GPU:
                cmd.append("--nv")
            
            # Add bind mounts for all input directories (read-only)
            for input_dir in input_dirs:
                cmd.extend(["--bind", f"{input_dir}:/input/{input_dir.name}:ro"])
            
            # Add bind mount for output directory (read-write)
            cmd.extend(["--bind", f"{output_dir}:/output:rw"])
            
            # Add container path and instance name
            cmd.extend([str(container_path), instance_name])
            
            self.logger.info(f"Starting instance command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.INSTANCE_START_TIMEOUT  # Use configured timeout
            )
            
            if result.returncode == 0:
                # Verify instance is running
                return self._verify_instance_running(instance_name)
            else:
                self.logger.error(f"Failed to start instance: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Instance start timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error starting instance: {e}")
            return False
    
    def _verify_instance_running(self, instance_name: str) -> bool:
        """Verify that the instance is running."""
        try:
            cmd = [SETTINGS.APPTAINER_EXECUTABLE, "instance", "list", instance_name]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and instance_name in result.stdout
        except Exception:
            return False
    
    def _stop_instance(self, instance_name: str):
        """Stop container instance."""
        try:
            self.logger.info(f"🛑 Stopping container instance: {instance_name}")
            cmd = [SETTINGS.APPTAINER_EXECUTABLE, "instance", "stop", instance_name]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.INSTANCE_STOP_TIMEOUT
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Container instance stopped successfully")
            else:
                self.logger.warning(f"⚠️ Failed to stop instance cleanly: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error stopping instance: {e}")
    
    def _run_instance_inference(self, instance_name: str, modality: str, 
                               input_file: Path, output_file: Path) -> bool:
        """Run inference using container instance."""
        try:
            # Build command for this specific inference
            cmd = [
                SETTINGS.APPTAINER_EXECUTABLE,
                "exec",
                f"instance://{instance_name}",
                "python", SETTINGS.PYTHON_SCRIPT,
                f"--{modality}",  # Use specific modality flag
                "--input", f"/input/{input_file.parent.name}/{input_file.name}",
                "--output", f"/output/{output_file.name}"
            ]
            
            self.logger.info(f"⏳ Running inference command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.CONTAINER_TIMEOUT
            )
            
            if result.returncode == 0:
                # Verify output was created
                if output_file.exists():
                    return True
                else:
                    self.logger.error(f"Inference completed but no output file created: {output_file}")
                    return False
            else:
                self.logger.error(f"Inference failed: {result.stderr}")
                if result.stdout:
                    self.logger.error(f"Stdout: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"Inference timed out for {modality}: {input_file}")
            return False
        except Exception as e:
            self.logger.error(f"Error running inference: {e}")
            return False
    
    def _get_input_files(self, task) -> List[Tuple[Path, str]]:
        """Get list of (file_path, modality) tuples for the task."""
        input_files = []
        
        # Get all subject directories for this task
        subject_dirs = task.get_subject_dirs()
        
        for subject_dir in subject_dirs:
            ses_dir = subject_dir / "ses_1"
            if not ses_dir.exists():
                continue
            
            # Find files for each modality
            for modality in task.image_modalities:
                if modality == "t2s" or modality == "swi":  # Handle optional modalities
                    # Try t2s first, then swi
                    t2s_file = ses_dir / "t2s.nii.gz"
                    swi_file = ses_dir / "swi.nii.gz"
                    
                    if t2s_file.exists():
                        input_files.append((t2s_file, "t2s"))
                    elif swi_file.exists():
                        input_files.append((swi_file, "swi"))
                else:
                    # Standard modality files
                    modality_file = ses_dir / f"{modality}.nii.gz"
                    if modality_file.exists():
                        input_files.append((modality_file, modality))
        
        return input_files
    
    def _aggregate_outputs(self, modality_outputs: List[Path], final_output: Path, task) -> bool:
        """Aggregate multiple modality outputs into final result."""
        try:
            if task.name == "Infarct Classification":
                # For classification: average probabilities across modalities
                return self._aggregate_classification_outputs(modality_outputs, final_output)
            
            elif task.name == "Brain Age Prediction":
                # For age prediction: average ages across modalities
                return self._aggregate_regression_outputs(modality_outputs, final_output)
            
            elif task.name == "Meningioma Segmentation":
                # For segmentation: use the best available output
                return self._aggregate_segmentation_outputs(modality_outputs, final_output)
            
            else:
                # Unknown task: just use first output
                if modality_outputs:
                    import shutil
                    shutil.copy(modality_outputs[0], final_output)
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Error aggregating outputs: {e}")
            return False
    
    def _aggregate_classification_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Aggregate classification probabilities."""
        import pandas as pd
        import numpy as np
        
        all_probs = []
        headers = []
        
        for output_file in outputs:
            try:
                df = pd.read_csv(output_file)
                if 'prob_class_1' in df.columns:
                    all_probs.extend(df['prob_class_1'].tolist())
                    headers.extend(df['header'].tolist())
            except Exception as e:
                self.logger.warning(f"Could not read {output_file}: {e}")
                continue
        
        if all_probs:
            # Average probabilities per subject
            unique_headers = list(set(headers))
            averaged_probs = []
            
            for header in unique_headers:
                header_probs = [prob for h, prob in zip(headers, all_probs) if h == header]
                avg_prob = np.mean(header_probs)
                averaged_probs.append({'header': header, 'prob_class_1': avg_prob})
            
            # Save aggregated results
            result_df = pd.DataFrame(averaged_probs)
            result_df.to_csv(final_output, index=False)
            return True
        
        return False
    
    def _aggregate_regression_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Aggregate age predictions."""
        import pandas as pd
        import numpy as np
        
        all_ages = []
        headers = []
        
        for output_file in outputs:
            try:
                df = pd.read_csv(output_file)
                if 'value' in df.columns:
                    all_ages.extend(df['value'].tolist())
                    headers.extend(df['header'].tolist())
            except Exception as e:
                self.logger.warning(f"Could not read {output_file}: {e}")
                continue
        
        if all_ages:
            # Average ages per subject
            unique_headers = list(set(headers))
            averaged_ages = []
            
            for header in unique_headers:
                header_ages = [age for h, age in zip(headers, all_ages) if h == header]
                avg_age = np.mean(header_ages)
                averaged_ages.append({'header': header, 'value': avg_age})
            
            # Save aggregated results
            result_df = pd.DataFrame(averaged_ages)
            result_df.to_csv(final_output, index=False)
            return True
        
        return False
    
    def _aggregate_segmentation_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Aggregate segmentation outputs (use first available)."""
        import shutil
        
        # For segmentation, just use the first successful output
        # In a real system, you might want to combine segmentations
        if outputs:
            shutil.copy(outputs[0], final_output)
            return True
        
        return False