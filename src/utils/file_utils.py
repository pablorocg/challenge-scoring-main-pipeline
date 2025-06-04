import shutil
from pathlib import Path
from typing import List

from config.settings import SETTINGS


def find_containers(directory: Path) -> List[Path]:
    """Find all .sif container files in directory."""
    if not directory.exists():
        return []

    return list(directory.glob("*.sif"))


def move_container(container_path: Path, destination_dir: Path):
    """Move container to destination directory."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / container_path.name
    shutil.move(str(container_path), str(destination_path))


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        SETTINGS.INCOMING_DIR,
        SETTINGS.EVALUATED_DIR,
        SETTINGS.RESULTS_DIR,
        SETTINGS.INPUT_DIR,
        SETTINGS.OUTPUT_DIR,
        SETTINGS.LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
