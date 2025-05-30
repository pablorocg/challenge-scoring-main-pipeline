#!/usr/bin/env python3
"""
FOMO Evaluation System - Main Entry Point
Orchestrates the evaluation of medical imaging tasks using Apptainer containers.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

from config.settings import SETTINGS
from tasks import TaskFactory
from runners.apptainer_runner import ApptainerRunner
from utils.file_utils import find_containers, move_container
from utils.logging_utils import setup_logging, get_logger


def main():
    """Main evaluation orchestration."""
    logger = setup_logging()
    logger.info("Starting FOMO evaluation system")
    
    runner = ApptainerRunner()
    containers = find_containers(SETTINGS.INCOMING_DIR)
    
    if not containers:
        logger.info("No containers found for evaluation")
        return
    
    logger.info(f"Found {len(containers)} containers to evaluate")
    
    for container_path in containers:
        try:
            evaluate_container(container_path, runner, logger)
        except Exception as e:
            logger.error(f"Failed to evaluate {container_path}: {e}")
            continue
    
    logger.info("Evaluation completed")


def evaluate_container(container_path: Path, runner: ApptainerRunner, logger):
    """Evaluate a single container."""
    entity_id, task_id = parse_container_name(container_path.name)
    logger.info(f"Evaluating {entity_id} for {task_id}")
    
    # Get task configuration
    task = TaskFactory.create_task(task_id)
    
    # Run inference
    output_path = SETTINGS.OUTPUT_DIR / f"{entity_id}_{task_id}_output{task.output_extension}"
    success = runner.run_inference(container_path, task, output_path)
    
    if not success:
        logger.error(f"Inference failed for {container_path}")
        return
    
    # Compute metrics
    results = task.evaluate(output_path)
    
    # Save results
    save_results(entity_id, task_id, results)
    
    # Move container to evaluated
    move_container(container_path, SETTINGS.EVALUATED_DIR)
    
    logger.info(f"Completed evaluation for {entity_id}_{task_id}")


def parse_container_name(filename: str) -> tuple[str, str]:
    """Parse entity_id and task from container filename."""
    name = filename.replace('.sif', '')
    parts = name.split('_')
    
    if len(parts) < 2 or not parts[-1].startswith('task'):
        raise ValueError(f"Invalid container filename format: {filename}")
    
    task_id = parts[-1]
    entity_id = '_'.join(parts[:-1])
    
    return entity_id, task_id


def save_results(entity_id: str, task_id: str, results: dict):
    """Save evaluation results to JSON file with same name as container."""
    timestamp = datetime.now().isoformat()
    result_data = {
        'entity_id': entity_id,
        'task_id': task_id,
        'timestamp': timestamp,
        'container_name': f"{entity_id}_{task_id}.sif",
        'results': results
    }
    
    # JSON file has same name as container (without .sif extension)
    result_file = SETTINGS.RESULTS_DIR / f"{entity_id}_{task_id}.json"
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)


if __name__ == "__main__":
    main()
