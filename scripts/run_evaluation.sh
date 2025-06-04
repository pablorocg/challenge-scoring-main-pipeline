#!/bin/bash
# FOMO Evaluation Runner - Simplified

set -e

cd "$(dirname "$0")/.."

echo "🧠 FOMO Evaluation System"

# Check .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from template..."
    cp .env.template .env
    echo "Please edit .env file and run again"
    exit 1
fi

# Load environment
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check apptainer
if ! command -v "${FOMO_APPTAINER_EXECUTABLE:-apptainer}" &> /dev/null; then
    echo "❌ Apptainer not found"
    exit 1
fi

# Check for containers
INCOMING_DIR="${FOMO_INCOMING_DIR:-/submissions/incoming}"
CONTAINER_COUNT=$(find "$INCOMING_DIR" -name "*.sif" 2>/dev/null | wc -l)

if [ "$CONTAINER_COUNT" -eq 0 ]; then
    echo "⚠️ No containers found in $INCOMING_DIR"
    exit 1
fi

echo "Found $CONTAINER_COUNT container(s)"

# Ensure directories
python -c "from src.utils.file_utils import ensure_directories; ensure_directories()"

# Run evaluation
echo "🚀 Starting evaluation..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

if [ "${1:-}" = "--debug" ]; then
    export FOMO_DEBUG_MODE=true
fi

python src/main.py "$@"

# Check results
RESULTS_DIR="${FOMO_RESULTS_DIR:-/results}"
RESULT_COUNT=$(find "$RESULTS_DIR" -name "*.json" 2>/dev/null | wc -l)

if [ "$RESULT_COUNT" -gt 0 ]; then
    echo "✅ Generated $RESULT_COUNT result(s)"
else
    echo "⚠️ No results generated"
fi

echo "✅ Completed"