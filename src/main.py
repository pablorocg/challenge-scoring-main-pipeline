"""FOMO Evaluation System - Simplified main entry point."""

import json
from pathlib import Path
from datetime import datetime

from config.settings import SETTINGS
from tasks import TaskFactory
from runners.apptainer_runner import ApptainerRunner
from utils.file_utils import find_containers, move_container, ensure_directories
from utils.logging_utils import setup_logging


def main():
    """Main evaluation orchestration - simplified."""
    logger = setup_logging()
    logger.info("Starting FOMO evaluation system")
    
    # Ensure directories exist
    ensure_directories()
    
    # Find containers to process
    containers = find_containers(SETTINGS.INCOMING_DIR)
    
    if not containers:
        logger.info("No containers found for evaluation")
        return
    
    logger.info(f"Found {len(containers)} containers to evaluate")
    
    # Process each container
    runner = ApptainerRunner()
    
    for container_path in containers:
        process_container(container_path, runner, logger)
    
    logger.info("Evaluation completed")


def process_container(container_path: Path, runner: ApptainerRunner, logger):
    """Process a single container."""
    entity_id, task_id = parse_container_name(container_path.name)
    logger.info(f"Processing {entity_id} for {task_id}")
    
    # Get task and create output directory
    task = TaskFactory.create_task(task_id)
    task_output_dir = SETTINGS.OUTPUT_DIR / task_id / entity_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set final output path
    output_path = task_output_dir / f"{entity_id}_{task_id}_output{task.output_extension}"
    
    # Run inference
    if runner.run_inference(container_path, task, output_path, task_output_dir):
        # Evaluate results
        results = task.evaluate(output_path, task_output_dir)
        
        # Save results
        save_results(entity_id, task_id, results, task_output_dir)
        
        # Move container to evaluated
        move_container(container_path, SETTINGS.EVALUATED_DIR)
        
        logger.info(f"Completed {entity_id}_{task_id}")
    else:
        logger.error(f"Failed to process {entity_id}_{task_id}")


def parse_container_name(filename: str) -> tuple[str, str]:
    """Parse entity_id and task from container filename."""
    name = filename.replace('.sif', '')
    parts = name.split('_')
    
    if len(parts) < 2 or not parts[-1].startswith('task'):
        raise ValueError(f"Invalid container filename: {filename}")
    
    task_id = parts[-1]
    entity_id = '_'.join(parts[:-1])
    
    return entity_id, task_id


def save_results(entity_id: str, task_id: str, results: dict, task_output_dir: Path):
    """Save evaluation results."""
    result_data = {
        'entity_id': entity_id,
        'task_id': task_id,
        'timestamp': datetime.now().isoformat(),
        'container_name': f"{entity_id}_{task_id}.sif",
        'results': results
    }
    
    # Save in task output directory
    result_file = task_output_dir / f"{entity_id}_{task_id}_results.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    # Also save in main results directory (backwards compatibility)
    main_result_file = SETTINGS.RESULTS_DIR / f"{entity_id}_{task_id}.json"
    main_result_file.parent.mkdir(parents=True, exist_ok=True)
    with open(main_result_file, 'w') as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    main()