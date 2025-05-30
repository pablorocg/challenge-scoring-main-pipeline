"""Task management for FOMO evaluation."""

from .task1_infarct import InfarctClassificationTask
from .task2_meningioma import MeningiomaSegmentationTask
from .task3_brain_age import BrainAgePredictionTask


class TaskFactory:
    """Factory for creating task instances."""
    
    _TASKS = {
        'task1': InfarctClassificationTask,
        'task2': MeningiomaSegmentationTask,
        'task3': BrainAgePredictionTask,
    }
    
    @classmethod
    def create_task(cls, task_id: str):
        """Create a task instance by ID."""
        if task_id not in cls._TASKS:
            raise ValueError(f"Unknown task ID: {task_id}")
        return cls._TASKS[task_id]()
