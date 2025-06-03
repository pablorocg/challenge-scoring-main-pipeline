# src/tasks/base_task.py
"""Simplified base class for evaluation tasks."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any


class BaseTask(ABC):
    """Abstract base class for evaluation tasks."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Task name."""
        pass
    
    @property
    @abstractmethod
    def data_dir(self) -> Path:
        """Task data directory."""
        pass
    
    @property
    @abstractmethod
    def output_extension(self) -> str:
        """Expected output file extension."""
        pass
    
    @abstractmethod
    def evaluate(self, output_path: Path, task_output_dir: Path = None) -> Dict[str, Any]:
        """Evaluate task output against ground truth."""
        pass
    
    def get_subject_dirs(self) -> list[Path]:
        """Get list of subject directories."""
        preprocessed_dir = self.data_dir / "preprocessed"
        if not preprocessed_dir.exists():
            return []
        
        return [d for d in preprocessed_dir.iterdir() 
                if d.is_dir() and d.name.startswith('sub_')]
    
    def get_labels_path(self, subject_id: str) -> Path:
        """Get path to labels for a subject."""
        return self.data_dir / "labels" / subject_id