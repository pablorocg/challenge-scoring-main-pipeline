#!/bin/bash

# FOMO Evaluation Runner Script
cd "$(dirname "$0")/.."

# Ensure directories exist
python -c "from src.utils.file_utils import ensure_directories; ensure_directories()"

# Run evaluation
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/main.py "$@"
