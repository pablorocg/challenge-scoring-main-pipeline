"""FOMO Evaluation System - Optimized main entry point."""

import json
from pathlib import Path
from datetime import datetime

from config.settings import SETTINGS
from tasks.task1_infarct import InfarctClassificationTask
from tasks.task2_meningioma import MeningiomaSegmentationTask
from tasks.task3_brain_age import BrainAgePredictionTask
from runners.apptainer_runner import ApptainerRunner
from utils.file_utils import find_containers, move_container, ensure_directories
from utils.logging_utils import setup_logging


# Inline TaskFactory - no separate file needed
TASK_MAP = {
    'task1': InfarctClassificationTask,
    'task2': MeningiomaSegmentationTask,
    'task3': BrainAgePredictionTask,
}


def create_task(task_id: str):
    """Create task instance."""
    if task_id not in TASK_MAP:
        raise ValueError(f"Unknown task: {task_id}")
    return TASK_MAP[task_id]()


def main():
    """Main evaluation entry point."""
    logger = setup_logging()
    logger.info("Starting FOMO evaluation")
    
    ensure_directories()
    containers = find_containers(SETTINGS.INCOMING_DIR)
    
    if not containers:
        logger.info("No containers found")
        return
    
    logger.info(f"Processing {len(containers)} containers")
    
    runner = ApptainerRunner()
    for container_path in containers:
        process_container(container_path, runner, logger)
    
    logger.info("Evaluation completed")


def process_container(container_path: Path, runner: ApptainerRunner, logger):
    """Process a single container."""
    entity_id, task_id = parse_container_name(container_path.name)
    logger.info(f"Processing {entity_id}_{task_id}")
    
    task = create_task(task_id)
    task_output_dir = SETTINGS.OUTPUT_DIR / task_id / entity_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = task_output_dir / f"{entity_id}_{task_id}_output{task.output_extension}"
    
    if runner.run_inference(container_path, task, output_path, task_output_dir):
        results = task.evaluate(output_path, task_output_dir)
        save_results(entity_id, task_id, results, task_output_dir)
        move_container(container_path, SETTINGS.EVALUATED_DIR)
        logger.info(f"✅ {entity_id}_{task_id}")
    else:
        logger.error(f"❌ {entity_id}_{task_id}")


def parse_container_name(filename: str) -> tuple[str, str]:
    """Parse entity_id and task from filename."""
    name = filename.replace('.sif', '')
    parts = name.split('_')
    
    if len(parts) < 2 or not parts[-1].startswith('task'):
        raise ValueError(f"Invalid container filename: {filename}")
    
    return '_'.join(parts[:-1]), parts[-1]


def save_results(entity_id: str, task_id: str, results: dict, task_output_dir: Path):
    """Save evaluation results."""
    result_data = {
        'entity_id': entity_id,
        'task_id': task_id,
        'timestamp': datetime.now().isoformat(),
        'results': results
    }
    
    # Save in task output directory
    result_file = task_output_dir / f"{entity_id}_{task_id}_results.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Also save in main results directory
    main_result_file = SETTINGS.RESULTS_DIR / f"{entity_id}_{task_id}.json"
    main_result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(main_result_file, 'w') as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    main()