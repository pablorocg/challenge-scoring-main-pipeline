# FOMO Evaluation System

A robust evaluation system for medical imaging tasks using Apptainer containers.

## Features

- **Task 1**: Infarct classification (AUROC metric) - CSV output
- **Task 2**: Meningioma segmentation (DSC, NSD metrics) - NIfTI output  
- **Task 3**: Brain age prediction (Absolute Error, Correlation metrics) - CSV output

## Architecture

```
src/
├── main.py              # Main orchestration script
├── config/              # Configuration management
├── tasks/               # Task-specific implementations
├── metrics/             # Metric computation modules
├── runners/             # Container execution
└── utils/               # Utility functions
```

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Configure environment: `cp .env.template .env` and modify values
3. Place containers in incoming directory with format: `entity_id_taskX.sif`
4. Run evaluation: `python src/main.py`

## Configuration

The system uses environment variables for configuration. Copy `.env.template` to `.env` and modify:

```bash
# Essential paths
FOMO_BASE_DIR=/projects/users/data/fomo25
FOMO_INCOMING_DIR=/submissions/incoming
FOMO_EVALUATED_DIR=/submissions/evaluated

# Container settings
FOMO_APPTAINER_EXECUTABLE=apptainer
FOMO_CONTAINER_TIMEOUT=3600

# Metric worst values (for failed evaluations)
FOMO_AUROC_WORST=0.0
FOMO_AE_WORST=100.0
```

## Container Requirements

Containers must:
- Accept `--modality`, `--input`, and `--output` parameters
- Implement `/app/predict.py` script
- Output format:
  - **Task 1**: CSV with `header,prob_class_1` columns
  - **Task 2**: Binary NIfTI segmentation mask
  - **Task 3**: CSV with `header,value` columns

## Container Command Format

```bash
python /app/predict.py --modality --input /input --output /output/filename.ext
```

## Output

- Results saved as JSON in results directory
- Containers moved to evaluated directory
- Logs available in logs directory

## Security

- Configuration stored in `.env` file (not committed to git)
- Sensitive paths and settings externalized
- Container execution isolated with read-only input mounts
