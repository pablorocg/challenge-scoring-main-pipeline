"""Configuration settings for evaluation system."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    # Directory paths
    BASE_DIR: Path = Path(os.getenv("FOMO_BASE_DIR"))
    INCOMING_DIR: Path = Path(os.getenv("FOMO_INCOMING_DIR"))
    EVALUATED_DIR: Path = Path(os.getenv("FOMO_EVALUATED_DIR"))
    RESULTS_DIR: Path = Path(os.getenv("FOMO_RESULTS_DIR"))
    INPUT_DIR: Path = Path(os.getenv("FOMO_INPUT_DIR"))
    OUTPUT_DIR: Path = Path(os.getenv("FOMO_OUTPUT_DIR"))
    LOGS_DIR: Path = Path(os.getenv("FOMO_LOGS_DIR"))

    # Task data paths (computed from BASE_DIR)
    TASK1_DATA_DIR: Path = None
    TASK2_DATA_DIR: Path = None
    TASK3_DATA_DIR: Path = None

    # Metric worst values
    AUROC_WORST: float = float(os.getenv("FOMO_AUROC_WORST"))
    DSC_WORST: float = float(os.getenv("FOMO_DSC_WORST"))
    NSD_WORST: float = float(os.getenv("FOMO_NSD_WORST"))
    AE_WORST: float = float(os.getenv("FOMO_AE_WORST"))
    CORR_WORST: float = float(os.getenv("FOMO_CORR_WORST"))

    # Apptainer settings
    APPTAINER_EXECUTABLE: str = os.getenv("FOMO_APPTAINER_EXECUTABLE")
    PYTHON_SCRIPT: str = os.getenv("FOMO_PYTHON_SCRIPT")

    # Execution settings
    CONTAINER_TIMEOUT: int = int(os.getenv("FOMO_CONTAINER_TIMEOUT"))
    INSTANCE_START_TIMEOUT: int = int(os.getenv("FOMO_INSTANCE_START_TIMEOUT"))
    INSTANCE_STOP_TIMEOUT: int = int(os.getenv("FOMO_INSTANCE_STOP_TIMEOUT"))

    # GPU settings
    ENABLE_GPU: bool = os.getenv("FOMO_ENABLE_GPU").lower() == "true"

    # Debug settings
    DEBUG_MODE: bool = os.getenv("FOMO_DEBUG_MODE").lower() == "true"
    KEEP_INTERMEDIATE_FILES: bool = (
        os.getenv("FOMO_KEEP_INTERMEDIATE_FILES").lower() == "true"
    )

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
