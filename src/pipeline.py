# Finetuning





# Task 1: Infarct Detection

# Image-level binary classification of the presence of infarct(s) in the brain.

# Cohort: 18-year-old or older patients with possible brain infarct(s). Subjects underwent an MRI scan and were subsequently diagnosed with infarcts and a control group in 2019 in multiple hospitals in Denmark (evaluation data) and India (finetuning data). Finetuning data is acquired from GE, Siemens scanners. Evaluation data is acquired from a diverse set of scanners from GE, Siemens and Philips. All scanners are 1.5T or 3T.

# Sequences: Sequences always include T2 FLAIR, DWI (b-value 1000), ADC, and either T2* or SWI images.

# Available additional information: A binary mask of the infarcts is provided for the finetuning data. This mask is not provided for the evaluation data. Note that the masks have not been verified by expert radiologists and are meant to serve as approximate annotations only.

# Dataset sizes: Finetune cases: 21. Validation cases: 80. Test cases: 320. Each case represent different subjects.

# Assessment Metric: AUROC (Area Under the Receiver Operator Curve (ROC))





# Task 2: Meningioma Segmentation

# Binary segmentation of brain meningiomas on MRI scans.

# Cohort: 18-year-old or older preoperative meningioma patients. Subjects who underwent an MRI diagnosed with preoperative meningioma in 2019 in multiple hospitals in Denmark (evaluation data) and India (finetuning data). Finetuning data is acquired from GE, Siemens scanners. Evaluation data is acquired from a diverse set of scanners from GE, Siemens and Philips. All scanners are 1.5T or 3T.

# Sequences: Sequences always include T2 FLAIR, DWI (b-value 1000), and either T2* or SWI images.

# Available additional information: For the finetuning data, a binary mask of meningiomas will be provided.

# Dataset sizes: Finetuning cases: 23. Validation cases: 40. Test cases: 160. Each case represent different subjects.

# Assessment Metric: Overlap-based metric: Dice Similarity Coefficient (DSC). Boundary-based metric: Normal Surface Distance (NSD)






# Task 3: Brain Age Regression

# Accurate prediction of the age of the patient based on MRI scans.

# Cohort: Patients 18 years or older with no underlying brain conditions defined as patients with no neurological conditions and not on brain-related medication. Subjects who underwent an MRI with no underlying brain conditions in 2019 in multiple hospitals in Denmark (evaluation data) and Boston (finetuning data).

# Sequences: Sequences always includes T1w and T2w MRI scans.

# Available additional information: For each case, age (represented by an integer) at the time of the MRI visit is provided.

# Dataset sizes: Finetuning cases: 100. Validation Cases: 200. Test cases: 800 Each case represent different subjects.

# Assessment Metric: Absolute Error (AE), Correlation Coefficitent



#!/usr/bin/env python3
"""
Script de evaluación para competición de IA con contenedores Apptainer
Evalúa modelos en tareas específicas (task_1, task_2, task_3)
"""

import os
import sys
import json
import subprocess
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvaluationResult:
    """Clase para almacenar resultados de evaluación"""
    container_name: str
    task: str
    score: float
    execution_time: float
    status: str
    error_message: Optional[str] = None
    metrics: Optional[Dict] = None






class AICompetitionEvaluator:
    def __init__(self, submissions_dir: str = "submissions/incoming", 
                 data_dir: str = "data", results_dir: str = "results"):
        self.submissions_dir = Path(submissions_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Configurar logging
        self.setup_logging()
        
        # Configuración de tareas
        self.task_configs = {
            "task_1": {
                "input_file": "task_1_input.txt",
                "ground_truth": "task_1_ground_truth.json",
                "timeout": 300,  # 5 minutos
                "max_score": 100
            },
            "task_2": {
                "input_file": "task_2_input.csv",
                "ground_truth": "task_2_ground_truth.json",
                "timeout": 600,  # 10 minutos
                "max_score": 100
            },
            "task_3": {
                "input_file": "task_3_input.json",
                "ground_truth": "task_3_ground_truth.json",
                "timeout": 900,  # 15 minutos
                "max_score": 100
            }
        }
    
    def setup_logging(self):
        """Configurar sistema de logging"""
        log_file = self.results_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def find_containers(self) -> List[Path]:
        """Encontrar todos los contenedores en el directorio de submissions"""
        if not self.submissions_dir.exists():
            self.logger.error(f"Directorio de submissions no existe: {self.submissions_dir}")
            return []
        
        containers = []
        # Buscar archivos con extensiones comunes de contenedores Apptainer
        for ext in ['*.sif', '*.img']:
            containers.extend(self.submissions_dir.glob(ext))
        
        self.logger.info(f"Encontrados {len(containers)} contenedores")
        return containers
    
    def prepare_task_environment(self, task: str) -> Tuple[bool, str]:
        """Preparar el entorno para una tarea específica"""
        config = self.task_configs.get(task)
        if not config:
            return False, f"Tarea desconocida: {task}"
        
        # Verificar que existen los archivos necesarios
        input_file = self.data_dir / config["input_file"]
        ground_truth_file = self.data_dir / config["ground_truth"]
        
        if not input_file.exists():
            return False, f"Archivo de entrada no encontrado: {input_file}"
        
        if not ground_truth_file.exists():
            return False, f"Ground truth no encontrado: {ground_truth_file}"
        
        return True, "Entorno preparado correctamente"
    
    def run_container(self, container_path: Path, task: str) -> Tuple[bool, str, float]:
        """Ejecutar un contenedor para una tarea específica"""
        config = self.task_configs[task]
        input_file = self.data_dir / config["input_file"]
        
        # Crear directorio temporal para la salida
        output_dir = self.results_dir / f"temp_{container_path.stem}_{task}"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"output_{task}.json"
        
        # Comando para ejecutar el contenedor
        cmd = [
            "apptainer", "exec",
            "--bind", f"{self.data_dir}:/data:ro",  # Montar datos como solo lectura
            "--bind", f"{output_dir}:/output",      # Montar directorio de salida
            str(container_path),
            "python", "/app/main.py",  # Asumimos que el script principal está en /app/main.py
            "--task", task,
            "--input", f"/data/{config['input_file']}",
            "--output", f"/output/output_{task}.json"
        ]
        
        self.logger.info(f"Ejecutando: {' '.join(cmd)}")
        
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                timeout=config["timeout"],
                capture_output=True,
                text=True
            )
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                if output_file.exists():
                    return True, str(output_file), execution_time
                else:
                    return False, "Archivo de salida no generado", execution_time
            else:
                error_msg = f"Error de ejecución: {result.stderr}"
                self.logger.error(error_msg)
                return False, error_msg, execution_time
                
        except subprocess.TimeoutExpired:
            execution_time = time.time() - start_time
            error_msg = f"Timeout después de {config['timeout']} segundos"
            self.logger.error(error_msg)
            return False, error_msg, execution_time
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error inesperado: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg, execution_time
    
    def evaluate_output(self, output_file: str, task: str) -> Tuple[float, Dict]:
        """Evaluar la salida del modelo contra el ground truth"""
        config = self.task_configs[task]
        ground_truth_file = self.data_dir / config["ground_truth"]
        
        try:
            # Cargar ground truth
            with open(ground_truth_file, 'r') as f:
                ground_truth = json.load(f)
            
            # Cargar output del modelo
            with open(output_file, 'r') as f:
                model_output = json.load(f)
            
            # Evaluar según el tipo de tarea
            if task == "task_1":
                return self.evaluate_task_1(model_output, ground_truth)
            elif task == "task_2":
                return self.evaluate_task_2(model_output, ground_truth)
            elif task == "task_3":
                return self.evaluate_task_3(model_output, ground_truth)
            else:
                return 0.0, {"error": "Tarea no implementada"}
                
        except Exception as e:
            self.logger.error(f"Error en evaluación: {str(e)}")
            return 0.0, {"error": str(e)}
    
    def evaluate_task_1(self, model_output: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
        """Evaluación específica para task_1 (personalizar según necesidades)"""
        # Ejemplo: evaluación de clasificación
        correct = 0
        total = len(ground_truth.get("labels", []))
        
        if "predictions" in model_output and "labels" in ground_truth:
            predictions = model_output["predictions"]
            labels = ground_truth["labels"]
            
            for pred, true_label in zip(predictions, labels):
                if pred == true_label:
                    correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
        
        return accuracy, metrics
    
    def evaluate_task_2(self, model_output: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
        """Evaluación específica para task_2 (personalizar según necesidades)"""
        # Ejemplo: evaluación de regresión (MSE)
        if "predictions" in model_output and "values" in ground_truth:
            predictions = model_output["predictions"]
            true_values = ground_truth["values"]
            
            mse = sum((p - t) ** 2 for p, t in zip(predictions, true_values)) / len(predictions)
            # Convertir MSE a score (menor MSE = mayor score)
            score = max(0, 100 - mse)
            
            metrics = {
                "mse": mse,
                "score": score,
                "n_samples": len(predictions)
            }
            
            return score, metrics
        
        return 0.0, {"error": "Formato de salida incorrecto"}
    
    def evaluate_task_3(self, model_output: Dict, ground_truth: Dict) -> Tuple[float, Dict]:
        """Evaluación específica para task_3 (personalizar según necesidades)"""
        # Ejemplo: evaluación F1-score
        if "predictions" in model_output and "labels" in ground_truth:
            predictions = model_output["predictions"]
            labels = ground_truth["labels"]
            
            # Calcular precision, recall, F1 (implementación simplificada)
            tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
            fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
            fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            score = f1 * 100
            
            metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "score": score
            }
            
            return score, metrics
        
        return 0.0, {"error": "Formato de salida incorrecto"}
    
    def evaluate_single_container(self, container_path: Path, task: str) -> EvaluationResult:
        """Evaluar un único contenedor"""
        self.logger.info(f"Evaluando {container_path.name} en {task}")
        
        # Preparar entorno
        success, message = self.prepare_task_environment(task)
        if not success:
            return EvaluationResult(
                container_name=container_path.name,
                task=task,
                score=0.0,
                execution_time=0.0,
                status="FAILED",
                error_message=message
            )
        
        # Ejecutar contenedor
        success, output_or_error, execution_time = self.run_container(container_path, task)
        
        if not success:
            return EvaluationResult(
                container_name=container_path.name,
                task=task,
                score=0.0,
                execution_time=execution_time,
                status="FAILED",
                error_message=output_or_error
            )
        
        # Evaluar salida
        score, metrics = self.evaluate_output(output_or_error, task)
        
        return EvaluationResult(
            container_name=container_path.name,
            task=task,
            score=score,
            execution_time=execution_time,
            status="SUCCESS",
            metrics=metrics
        )
    
    def run_evaluation(self, task: str, specific_container: Optional[str] = None) -> List[EvaluationResult]:
        """Ejecutar evaluación completa"""
        self.logger.info(f"Iniciando evaluación para {task}")
        
        containers = self.find_containers()
        if not containers:
            self.logger.error("No se encontraron contenedores")
            return []
        
        # Filtrar contenedor específico si se especifica
        if specific_container:
            containers = [c for c in containers if c.name == specific_container]
            if not containers:
                self.logger.error(f"Contenedor {specific_container} no encontrado")
                return []
        
        results = []
        for container in containers:
            result = self.evaluate_single_container(container, task)
            results.append(result)
            
            # Log resultado
            if result.status == "SUCCESS":
                self.logger.info(f"{container.name}: {result.score:.2f} puntos ({result.execution_time:.2f}s)")
            else:
                self.logger.error(f"{container.name}: FAILED - {result.error_message}")
        
        return results
    
    def generate_report(self, results: List[EvaluationResult], task: str):
        """Generar reporte de resultados"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.results_dir / f"report_{task}_{timestamp}.json"
        
        # Preparar datos del reporte
        report_data = {
            "task": task,
            "timestamp": timestamp,
            "total_containers": len(results),
            "successful_evaluations": len([r for r in results if r.status == "SUCCESS"]),
            "results": []
        }
        
        for result in results:
            report_data["results"].append({
                "container_name": result.container_name,
                "task": result.task,
                "score": result.score,
                "execution_time": result.execution_time,
                "status": result.status,
                "error_message": result.error_message,
                "metrics": result.metrics
            })
        
        # Ordenar por score (descendente)
        report_data["results"].sort(key=lambda x: x["score"], reverse=True)
        
        # Guardar reporte
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Reporte guardado en: {report_file}")
        
        # Mostrar ranking
        print(f"\n=== RANKING {task.upper()} ===")
        print(f"{'Posición':<10} {'Contenedor':<30} {'Score':<10} {'Tiempo':<10} {'Estado'}")
        print("-" * 70)
        
        for i, result in enumerate(report_data["results"], 1):
            print(f"{i:<10} {result['container_name']:<30} {result['score']:<10.2f} "
                  f"{result['execution_time']:<10.2f} {result['status']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluador de competición de IA")
    parser.add_argument("task", choices=["task_1", "task_2", "task_3"], 
                       help="Tarea a evaluar")
    parser.add_argument("--container", help="Evaluar solo un contenedor específico")
    parser.add_argument("--submissions-dir", default="submissions/incoming",
                       help="Directorio de contenedores (default: submissions/incoming)")
    parser.add_argument("--data-dir", default="data",
                       help="Directorio de datos (default: data)")
    parser.add_argument("--results-dir", default="results",
                       help="Directorio de resultados (default: results)")
    
    args = parser.parse_args()
    
    # Crear evaluador
    evaluator = AICompetitionEvaluator(
        submissions_dir=args.submissions_dir,
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    # Ejecutar evaluación
    results = evaluator.run_evaluation(args.task, args.container)
    
    if results:
        evaluator.generate_report(results, args.task)
    else:
        print("No se obtuvieron resultados de la evaluación")
        sys.exit(1)


if __name__ == "__main__":
    main()