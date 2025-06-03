# src/runners/apptainer_runner.py
"""Apptainer container runner."""

import subprocess
from pathlib import Path
from typing import List, Tuple

from src.config.settings import SETTINGS
from src.utils.logging_utils import get_logger


class ApptainerRunner:
    """Runs inference using Apptainer containers."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def run_inference(self, container_path: Path, task, output_path: Path) -> bool:
        """Run inference using an Apptainer container for all modalities."""
        self.logger.info(f"Running inference with {container_path}")
        
        # Get all input files for this task
        input_files = self._get_input_files(task)
        
        if not input_files:
            self.logger.error("No input files found for task")
            return False
        
        # Run container for each modality/file combination
        all_outputs = []
        success_count = 0
        
        for input_file, modality in input_files:
            self.logger.info(f"Processing {modality}: {input_file}")
            
            # Create individual output file for this modality
            modality_output = output_path.parent / f"{output_path.stem}_{modality}{output_path.suffix}"
            
            # Build and run command for this specific modality
            cmd = self._build_command(container_path, modality, input_file, modality_output)
            
            if self._run_single_inference(cmd, input_file, modality_output):
                all_outputs.append(modality_output)
                success_count += 1
            else:
                self.logger.error(f"Failed to process {modality}: {input_file}")
        
        # Aggregate results if needed (for tasks that need multiple modalities)
        if success_count > 0:
            return self._aggregate_outputs(all_outputs, output_path, task)
        
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
                if modality in ["t2s_or_swi"]:  # Handle optional modalities
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
    
    def _build_command(self, container_path: Path, modality: str, input_file: Path, output_file: Path) -> list[str]:
        """Build Apptainer command for a specific modality and file."""
        cmd = [
            SETTINGS.APPTAINER_EXECUTABLE,
            "exec",
            "--bind", f"{input_file.parent}:/input:ro",
            "--bind", f"{output_file.parent}:/output:rw",
            str(container_path),
            "python", SETTINGS.PYTHON_SCRIPT,
            f"--{modality}",  # Use specific modality flag
            "--input", f"/input/{input_file.name}",
            "--output", f"/output/{output_file.name}"
        ]

        self.logger.info(f"Running command: {' '.join(cmd)}")
        
        return cmd
    
    def _run_single_inference(self, cmd: list[str], input_file: Path, output_file: Path) -> bool:
        """Run a single inference command."""
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=SETTINGS.CONTAINER_TIMEOUT
            )
            
            if result.returncode == 0:
                self.logger.info(f"Inference completed for {input_file}")
                return output_file.exists()  # Verify output was created
            else:
                self.logger.error(f"Inference failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Inference timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running container: {e}")
            return False
    
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


# src/tasks/task1_infarct.py (Updated)



# src/tasks/task2_meningioma.py (Updated)



# src/tasks/task3_brain_age.py (Updated)



# # Updated README.md
# ## Container Command Format

# The system runs containers multiple times per task, once for each image file:

# ```bash
# # Task 1 - Multiple runs per subject:
# python /app/predict.py --flair --input /input/flair.nii.gz --output /output/result_flair.csv
# python /app/predict.py --adc --input /input/adc.nii.gz --output /output/result_adc.csv
# python /app/predict.py --dwi_b1000 --input /input/dwi_b1000.nii.gz --output /output/result_dwi.csv
# python /app/predict.py --t2s --input /input/t2s.nii.gz --output /output/result_t2s.csv

# # Task 2 - Multiple runs per subject:
# python /app/predict.py --flair --input /input/flair.nii.gz --output /output/seg_flair.nii.gz
# python /app/predict.py --dwi_b1000 --input /input/dwi_b1000.nii.gz --output /output/seg_dwi.nii.gz
# python /app/predict.py --swi --input /input/swi.nii.gz --output /output/seg_swi.nii.gz

# # Task 3 - Multiple runs per subject:
# python /app/predict.py --t1 --input /input/t1.nii.gz --output /output/age_t1.csv
# python /app/predict.py --t2 --input /input/t2.nii.gz --output /output/age_t2.csv
# ```

# Results from multiple modalities are automatically aggregated into the final output.