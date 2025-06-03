# src/runners/apptainer_runner.py
"""Apptainer container runner with subject-by-subject processing."""

import subprocess
import time
import re
from pathlib import Path
from typing import List, Tuple
from contextlib import contextmanager

from src.config.settings import SETTINGS
from src.utils.logging_utils import get_logger


class ApptainerRunner:
    """Runs inference using Apptainer container instances - one per subject."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def run_inference(self, container_path: Path, task, output_path: Path) -> bool:
        """Run inference by processing each subject individually."""
        self.logger.info(f"Running inference with {container_path}")
        
        # Get all subjects for this task
        subject_dirs = task.get_subject_dirs()
        
        if not subject_dirs:
            self.logger.error("No subject directories found for task")
            return False
        
        all_outputs = []
        success_count = 0
        
        # Process each subject individually
        for subject_dir in subject_dirs:
            self.logger.info(f"🔄 Processing subject: {subject_dir.name}")
            
            # Get modalities for this specific subject
            subject_modalities = self._get_subject_modalities(subject_dir, task)
            
            if not subject_modalities:
                self.logger.warning(f"⚠️ No valid modalities found for {subject_dir.name}")
                continue
            
            # Process this subject
            subject_outputs = []
            if self._process_subject(container_path, subject_dir, subject_modalities, 
                                   output_path, subject_outputs):
                all_outputs.extend(subject_outputs)
                success_count += 1
                self.logger.info(f"✅ Successfully processed {subject_dir.name}")
            else:
                self.logger.error(f"❌ Failed to process {subject_dir.name}")
        
        # Aggregate all results
        if success_count > 0:
            result = self._aggregate_all_outputs(all_outputs, output_path, task)
            self.logger.info(f"📊 Processing Summary: {success_count}/{len(subject_dirs)} subjects successful")
            return result
        
        self.logger.error("All subject processing failed")
        return False
    
    def _get_subject_modalities(self, subject_dir: Path, task) -> List[Tuple[Path, str]]:
        """Get available modalities for a specific subject."""
        ses_dir = subject_dir / "ses_1"
        if not ses_dir.exists():
            return []
        
        modalities = []
        
        # Handle standard modalities
        for modality in task.image_modalities:
            if modality in ["t2s", "swi"]:
                # Handle t2s/swi either/or logic
                t2s_file = ses_dir / "t2s.nii.gz"
                swi_file = ses_dir / "swi.nii.gz"
                
                if t2s_file.exists():
                    modalities.append((t2s_file, "t2s"))
                elif swi_file.exists():
                    modalities.append((swi_file, "swi"))
                # If neither exists, skip (don't add anything)
                
            elif modality not in ["t2s", "swi"]:
                # Standard modalities (flair, adc, dwi_b1000, t1, t2)
                modality_file = ses_dir / f"{modality}.nii.gz"
                if modality_file.exists():
                    modalities.append((modality_file, modality))
        
        return modalities
    
    def _process_subject(self, container_path: Path, subject_dir: Path, 
                        modalities: List[Tuple[Path, str]], output_path: Path,
                        subject_outputs: List[Path]) -> bool:
        """Process a single subject with its own container instance."""
        
        # Create unique instance name for this subject
        instance_name = f"fomo_{container_path.stem}_{subject_dir.name}_{int(time.time())}"
        
        try:
            with self._container_instance(container_path, subject_dir, instance_name) as instance:
                if not instance:
                    return False
                
                # Process each modality for this subject
                for modality_file, modality in modalities:
                    self.logger.info(f"  🔍 Processing {modality}: {modality_file.name}")
                    
                    # Create output file for this modality
                    modality_output = output_path.parent / f"{output_path.stem}_{subject_dir.name}_{modality}{output_path.suffix}"
                    
                    # Run inference on this file using the instance
                    if self._run_instance_inference(instance_name, modality, modality_file, modality_output):
                        subject_outputs.append(modality_output)
                        self.logger.info(f"    ✅ {modality} processed successfully")
                    else:
                        self.logger.error(f"    ❌ {modality} processing failed")
                        return False
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error processing subject {subject_dir.name}: {e}")
            return False
    
    @contextmanager
    def _container_instance(self, container_path: Path, subject_dir: Path, instance_name: str):
        """Context manager for single-subject container instance."""
        instance = None
        try:
            # Start container instance for this subject
            self.logger.info(f"🚀 Starting instance for {subject_dir.name}: {instance_name}")
            
            if self._start_subject_instance(container_path, subject_dir, instance_name):
                self.logger.info(f"✅ Instance started for {subject_dir.name}")
                instance = instance_name
                yield instance
            else:
                self.logger.error(f"❌ Failed to start instance for {subject_dir.name}")
                yield None
                
        finally:
            if instance:
                self._stop_instance(instance_name)
    
    def _start_subject_instance(self, container_path: Path, subject_dir: Path, instance_name: str) -> bool:
        """Start container instance for a single subject."""
        try:
            ses_dir = subject_dir / "ses_1"
            output_dir = SETTINGS.OUTPUT_DIR
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Build command with subject-specific binds
            cmd = [
                SETTINGS.APPTAINER_EXECUTABLE,
                "instance", "start"
            ]
            
            # Add GPU support if enabled
            if SETTINGS.ENABLE_GPU:
                cmd.append("--nv")
            
            # Bind THIS subject's session directory to /input
            cmd.extend(["--bind", f"{ses_dir}:/input:ro"])
            
            # Bind output directory
            cmd.extend(["--bind", f"{output_dir}:/output:rw"])
            
            # Add container and instance name
            cmd.extend([str(container_path), instance_name])
            
            self.logger.info(f"Instance command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.INSTANCE_START_TIMEOUT
            )
            
            if result.returncode == 0:
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
            self.logger.info(f"🛑 Stopping instance: {instance_name}")
            cmd = [SETTINGS.APPTAINER_EXECUTABLE, "instance", "stop", instance_name]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.INSTANCE_STOP_TIMEOUT
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Instance stopped successfully")
            else:
                self.logger.warning(f"⚠️ Failed to stop instance cleanly: {result.stderr}")
                
        except Exception as e:
            self.logger.error(f"Error stopping instance: {e}")
    
    def _run_instance_inference(self, instance_name: str, modality: str, 
                               input_file: Path, output_file: Path) -> bool:
        """Run inference using container instance."""
        try:
            # Build command - input file is now directly in /input/
            cmd = [
                SETTINGS.APPTAINER_EXECUTABLE,
                "exec",
                f"instance://{instance_name}",
                "python", SETTINGS.PYTHON_SCRIPT,
                f"--{modality}",
                "--input", f"/input/{input_file.name}",  # File is directly in /input/
                "--output", f"/output/{output_file.name}"
            ]
            
            self.logger.info(f"⏳ Inference command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.CONTAINER_TIMEOUT
            )
            
            if result.returncode == 0:
                if output_file.exists():
                    return True
                else:
                    self.logger.error(f"Inference completed but no output file: {output_file}")
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
    
    def _aggregate_all_outputs(self, all_outputs: List[Path], final_output: Path, task) -> bool:
        """Combine outputs from all subjects and modalities (no aggregation - keep all cases)."""
        try:
            if task.name == "Infarct Classification":
                return self._combine_classification_outputs(all_outputs, final_output)
            elif task.name == "Brain Age Prediction":
                return self._combine_regression_outputs(all_outputs, final_output)
            elif task.name == "Meningioma Segmentation":
                return self._aggregate_segmentation_outputs(all_outputs, final_output)
            else:
                # Default: use first output
                if all_outputs:
                    import shutil
                    shutil.copy(all_outputs[0], final_output)
                    return True
                return False
                
        except Exception as e:
            self.logger.error(f"Error combining outputs: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _combine_classification_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Combine classification outputs - keep all modalities as separate cases."""
        import pandas as pd
        
        all_predictions = []
        
        for output_file in outputs:
            try:
                # Extract subject ID from the output filename
                subject_id = self._extract_subject_from_filename(output_file.name)
                # Extract modality from filename
                modality = self._extract_modality_from_filename(output_file.name)
                
                if not subject_id:
                    self.logger.warning(f"Could not extract subject ID from filename: {output_file.name}")
                    continue
                
                # Load the CSV
                df = pd.read_csv(output_file)
                if 'prob_class_1' not in df.columns:
                    self.logger.warning(f"No prob_class_1 column in {output_file.name}")
                    continue
                
                # Get the probability (assuming single row per file)
                if len(df) > 0:
                    prob = float(df['prob_class_1'].iloc[0])
                    
                    # Create unique header for this modality case
                    header = f"{subject_id}_{modality}" if modality else subject_id
                    
                    all_predictions.append({
                        'header': header,
                        'prob_class_1': prob
                    })
                    
                    self.logger.debug(f"Added prediction: {header} -> {prob}")
                
            except Exception as e:
                self.logger.warning(f"Could not read {output_file}: {e}")
                continue
        
        if all_predictions:
            # Create final DataFrame with ALL modalities as separate cases
            final_df = pd.DataFrame(all_predictions)
            
            self.logger.info(f"Created {len(final_df)} individual modality predictions from {len(outputs)} output files")
            
            # Save all predictions (no aggregation!)
            final_df.to_csv(final_output, index=False)
            return True
        
        self.logger.error("No valid predictions found")
        return False
    
    def _combine_regression_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Combine age predictions - keep all modalities as separate cases."""
        import pandas as pd
        
        all_predictions = []
        
        for output_file in outputs:
            try:
                # Extract subject ID and modality from the output filename
                subject_id = self._extract_subject_from_filename(output_file.name)
                modality = self._extract_modality_from_filename(output_file.name)
                
                if not subject_id:
                    self.logger.warning(f"Could not extract subject ID from filename: {output_file.name}")
                    continue
                
                # Load the CSV
                df = pd.read_csv(output_file)
                if 'value' not in df.columns:
                    self.logger.warning(f"No value column in {output_file.name}")
                    continue
                
                # Get the age prediction (assuming single row per file)
                if len(df) > 0:
                    age = float(df['value'].iloc[0])
                    
                    # Create unique header for this modality case
                    header = f"{subject_id}_{modality}" if modality else subject_id
                    
                    all_predictions.append({
                        'header': header,
                        'value': age
                    })
                    
                    self.logger.debug(f"Added prediction: {header} -> {age}")
                
            except Exception as e:
                self.logger.warning(f"Could not read {output_file}: {e}")
                continue
        
        if all_predictions:
            # Create final DataFrame with ALL modalities as separate cases
            final_df = pd.DataFrame(all_predictions)
            
            self.logger.info(f"Created {len(final_df)} individual modality predictions from {len(outputs)} output files")
            
            # Save all predictions (no aggregation!)
            final_df.to_csv(final_output, index=False)
            return True
        
        self.logger.error("No valid age predictions found")
        return False
    
    def _aggregate_segmentation_outputs(self, outputs: List[Path], final_output: Path) -> bool:
        """Aggregate segmentation outputs."""
        import shutil
        
        # For segmentation, use the first successful output
        # In a real system, you might want to combine segmentations
        if outputs:
            shutil.copy(outputs[0], final_output)
            return True
        
        return False
    
    def _extract_subject_from_header(self, header: str) -> str:
        """Extract subject ID from header, removing modality suffix."""
        # Handle various header formats like:
        # "sub_1_flair" -> "sub_1"
        # "sub_25_adc" -> "sub_25" 
        # "sub_123_dwi_b1000" -> "sub_123"
        # "/path/to/sub_45_t2s.nii.gz" -> "sub_45"
        
        # First try to find sub_X pattern
        match = re.search(r'(sub_\d+)', header)
        if match:
            return match.group(1)
        
        # Fallback: if header doesn't contain sub_ pattern, return as-is
        self.logger.warning(f"Could not extract subject ID from header: {header}")
        return header
    
    def _extract_subject_from_filename(self, filename: str) -> str:
        """Extract subject ID from output filename."""
        # Handle filename formats like:
        # "180201_task1_output_sub_1_flair.csv" -> "sub_1"
        # "180201_task3_output_sub_25_t1.csv" -> "sub_25"
        # "entity_task2_output_sub_123_dwi_b1000.nii.gz" -> "sub_123"
        
        # Look for pattern: _sub_X_ or _sub_X. (before file extension)
        match = re.search(r'_?(sub_\d+)_', filename)
        if match:
            return match.group(1)
        
        # Try pattern at end: _sub_X.extension
        match = re.search(r'_?(sub_\d+)\.', filename)
        if match:
            return match.group(1)
        
        # Try any sub_X pattern in the filename
        match = re.search(r'(sub_\d+)', filename)
        if match:
            return match.group(1)
        
        self.logger.warning(f"Could not extract subject ID from filename: {filename}")
        return None
    
    def _extract_modality_from_filename(self, filename: str) -> str:
        """Extract modality from output filename."""
        # Handle filename formats like:
        # "180201_task1_output_sub_1_flair.csv" -> "flair"
        # "entity_task3_output_sub_25_t1.csv" -> "t1"
        # "test_task2_output_sub_123_dwi_b1000.nii.gz" -> "dwi_b1000"
        
        # Known modalities
        modalities = ['flair', 'adc', 'dwi_b1000', 't2s', 'swi', 't1', 't2']
        
        for modality in modalities:
            if modality in filename:
                return modality
        
        # Fallback: try to extract last part before extension
        # "sub_1_flair.csv" -> "flair"
        match = re.search(r'_([a-zA-Z0-9_]+)\.(csv|nii\.gz)', filename)
        if match:
            potential_modality = match.group(1)
            # Only return if it looks like a modality (not subject ID)
            if not potential_modality.startswith('sub_'):
                return potential_modality
        
        self.logger.warning(f"Could not extract modality from filename: {filename}")
        return "unknown"