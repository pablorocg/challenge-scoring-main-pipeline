"""File handling utilities."""

import shutil
from pathlib import Path
from typing import List

from src.utils.logging_utils import get_logger


def find_containers(directory: Path) -> List[Path]:
    """Find all .sif container files in directory."""
    if not directory.exists():
        return []
    
    return list(directory.glob("*.sif"))


def move_container(container_path: Path, destination_dir: Path):
    """Move container to destination directory."""
    logger = get_logger(__name__)
    
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / container_path.name
    
    try:
        shutil.move(str(container_path), str(destination_path))
        logger.info(f"Moved {container_path} to {destination_path}")
    except Exception as e:
        logger.error(f"Failed to move container: {e}")
        raise


def ensure_directories():
    """Ensure all required directories exist."""
    from src.config.settings import SETTINGS
    
    directories = [
        SETTINGS.INCOMING_DIR,
        SETTINGS.EVALUATED_DIR,
        SETTINGS.RESULTS_DIR,
        SETTINGS.INPUT_DIR,
        SETTINGS.OUTPUT_DIR,
        SETTINGS.LOGS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
