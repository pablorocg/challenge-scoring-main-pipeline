"""
Configuration settings for FOMO evaluation system.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class Settings:
    """Application settings."""
    
    # Directory paths
    BASE_DIR: Path = Path("/projects/users/data/fomo25")
    INCOMING_DIR: Path = Path("/submissions/incoming")
    EVALUATED_DIR: Path = Path("/submissions/evaluated")
    RESULTS_DIR: Path = Path("/results")
    INPUT_DIR: Path = Path("/input")
    OUTPUT_DIR: Path = Path("/output")
    LOGS_DIR: Path = Path("logs")
    
    # Task data paths
    TASK1_DATA_DIR: Path = BASE_DIR / "fomo-task1-val"
    TASK2_DATA_DIR: Path = BASE_DIR / "fomo-task2-val"
    TASK3_DATA_DIR: Path = BASE_DIR / "fomo-task3-val"
    
    # Metric worst values
    AUROC_WORST: float = 0.0
    DSC_WORST: float = 0.0
    NSD_WORST: float = 0.0
    AE_WORST: float = 100.0
    CORR_WORST: float = -100.0
    
    # Singularity settings
    APPTAINER_EXECUTABLE: str = "apptainer"
    PYTHON_SCRIPT: str = "/app/predict.py"


# Global settings instance
SETTINGS = Settings()
