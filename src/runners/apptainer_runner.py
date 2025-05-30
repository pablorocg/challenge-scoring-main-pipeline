"""Singularity container runner."""

import subprocess
from pathlib import Path

from config.settings import SETTINGS
from utils.logging_utils import get_logger


class ApptainerRunner:
    """Runs inference using Apptainer containers."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def run_inference(self, container_path: Path, task, output_path: Path) -> bool:
        """Run inference using a Apptainer container."""
        self.logger.info(f"Running inference with {container_path}")
        
        # Prepare command
        cmd = self._build_command(container_path, task, output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run container
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.logger.info("Inference completed successfully")
                return True
            else:
                self.logger.error(f"Inference failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("Inference timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error running container: {e}")
            return False
    
    def _build_command(self, container_path: Path, task, output_path: Path) -> list[str]:
        """Build Singularity command."""
        cmd = [
            SETTINGS.APPTAINER_EXECUTABLE,
            "exec",
            "--bind", f"{SETTINGS.INPUT_DIR}:/input:ro",
            "--bind", f"{SETTINGS.OUTPUT_DIR}:/output:rw",
            str(container_path),
            "python", SETTINGS.PYTHON_SCRIPT
        ]
        
        # Add modality arguments
        for modality in task.image_modalities:
            cmd.extend([f"--{modality}"])
        
        # Add input and output paths
        cmd.extend([
            "--input", "/input",
            "--output", f"/output/{output_path.name}"
        ])
        
        return cmd
