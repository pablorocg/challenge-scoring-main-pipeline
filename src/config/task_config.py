"""Task-specific configuration for evaluation system."""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TaskModalityConfig:
    """Configuration for task modalities."""

    required: List[str]
    optional: List[str]
    output_type: str  # 'csv' or 'nifti'

    def get_all_modalities(self) -> List[str]:
        """Get all possible modalities for this task."""
        return self.required + self.optional


# Task configurations
TASK_CONFIGS = {
    "Infarct Classification": TaskModalityConfig(
        required=["flair", "adc", "dwi_b1000"],
        optional=["t2s", "swi"],  # Either t2s OR swi
        output_type="csv",
    ),
    "Meningioma Segmentation": TaskModalityConfig(
        required=["flair", "dwi_b1000"],
        optional=["t2s", "swi"],  # Either t2s OR swi
        output_type="nifti",
    ),
    "Brain Age Prediction": TaskModalityConfig(
        required=["t1", "t2"], optional=[], output_type="csv"
    ),
}


def get_task_config(task_name: str) -> TaskModalityConfig:
    """Get configuration for a specific task."""
    return TASK_CONFIGS.get(task_name, TaskModalityConfig([], [], "csv"))


def validate_task_modalities(
    task_name: str, available_modalities: List[str]
) -> Dict[str, Any]:
    """Validate if available modalities meet task requirements."""
    config = get_task_config(task_name)

    missing_required = [
        mod for mod in config.required if mod not in available_modalities
    ]
    has_optional = any(mod in available_modalities for mod in config.optional)

    is_valid = len(missing_required) == 0 and (not config.optional or has_optional)

    return {
        "valid": is_valid,
        "missing_required": missing_required,
        "has_optional": has_optional,
        "available_optional": [
            mod for mod in config.optional if mod in available_modalities
        ],
    }
