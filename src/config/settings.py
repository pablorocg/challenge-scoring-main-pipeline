"""
Configuration settings for FOMO evaluation system.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # Directory paths
    BASE_DIR: Path = Path(os.getenv("FOMO_BASE_DIR", "/projects/users/data/fomo25"))
    INCOMING_DIR: Path = Path(os.getenv("FOMO_INCOMING_DIR", "/submissions/incoming"))
    EVALUATED_DIR: Path = Path(os.getenv("FOMO_EVALUATED_DIR", "/submissions/evaluated"))
    RESULTS_DIR: Path = Path(os.getenv("FOMO_RESULTS_DIR", "/results"))
    INPUT_DIR: Path = Path(os.getenv("FOMO_INPUT_DIR", "/input"))
    OUTPUT_DIR: Path = Path(os.getenv("FOMO_OUTPUT_DIR", "/output"))
    LOGS_DIR: Path = Path(os.getenv("FOMO_LOGS_DIR", "logs"))
    
    # Task data paths (computed from BASE_DIR)
    TASK1_DATA_DIR: Path = None
    TASK2_DATA_DIR: Path = None
    TASK3_DATA_DIR: Path = None
    
    # Metric worst values
    AUROC_WORST: float = float(os.getenv("FOMO_AUROC_WORST", "0.0"))
    DSC_WORST: float = float(os.getenv("FOMO_DSC_WORST", "0.0"))
    NSD_WORST: float = float(os.getenv("FOMO_NSD_WORST", "0.0"))
    AE_WORST: float = float(os.getenv("FOMO_AE_WORST", "100.0"))
    CORR_WORST: float = float(os.getenv("FOMO_CORR_WORST", "-100.0"))
    
    # Apptainer settings
    APPTAINER_EXECUTABLE: str = os.getenv("FOMO_APPTAINER_EXECUTABLE", "apptainer")
    PYTHON_SCRIPT: str = os.getenv("FOMO_PYTHON_SCRIPT", "/app/predict.py")
    
    # Execution settings
    CONTAINER_TIMEOUT: int = int(os.getenv("FOMO_CONTAINER_TIMEOUT", "3600"))
    INSTANCE_START_TIMEOUT: int = int(os.getenv("FOMO_INSTANCE_START_TIMEOUT", "60"))
    INSTANCE_STOP_TIMEOUT: int = int(os.getenv("FOMO_INSTANCE_STOP_TIMEOUT", "60"))
    
    # GPU settings
    ENABLE_GPU: bool = os.getenv("FOMO_ENABLE_GPU", "false").lower() == "true"
    
    # Debug settings
    DEBUG_MODE: bool = os.getenv("FOMO_DEBUG_MODE", "false").lower() == "true"
    KEEP_INTERMEDIATE_FILES: bool = os.getenv("FOMO_KEEP_INTERMEDIATE_FILES", "false").lower() == "true"
    
    def __post_init__(self):
        """Set computed paths after initialization."""
        self.TASK1_DATA_DIR = self.BASE_DIR / "fomo-task1-val"
        self.TASK2_DATA_DIR = self.BASE_DIR / "fomo-task2-val"
        self.TASK3_DATA_DIR = self.BASE_DIR / "fomo-task3-val"
    
    @property
    def gpu_flag(self) -> str:
        """Get GPU flag for container commands."""
        return "--nv" if self.ENABLE_GPU else ""


# Global settings instance
SETTINGS = Settings()