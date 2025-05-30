# FOMO Evaluation System

A robust evaluation system for medical imaging tasks using Singularity containers.

## Features

- **Task 1**: Infarct classification (AUROC metric)
- **Task 2**: Meningioma segmentation (DSC, NSD metrics)
- **Task 3**: Brain age prediction (Absolute Error, Correlation metrics)

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

## Usage

1. Install dependencies: `pip install -r requirements.txt`
2. Place containers in `/submissions/incoming/` with format: `entity_id_taskX.sif`
3. Run evaluation: `python src/main.py`

## Container Requirements

Containers must:
- Accept `--input` and `--output` parameters
- Implement `/app/predict.py` script
- Support required modality flags (--flair, --adc, etc.)

## Output

- Results saved as JSON in `/results/`
- Containers moved to `/submissions/evaluated/`
- Logs available in `logs/` directory
