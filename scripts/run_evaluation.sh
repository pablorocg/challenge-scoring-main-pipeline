#!/bin/bash

# FOMO Evaluation Runner Script with Instance Support
# This script demonstrates how to run the updated evaluation system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display colored messages
msg() {
    local color=$1
    local message=$2
    local emoji=$3
    echo -e "${color}${emoji} ${message}${NC}"
}

# Change to script directory
cd "$(dirname "$0")/.."

msg "$BLUE" "FOMO Evaluation System - Container Instance Mode" "🧠"

# Check if .env file exists
if [ ! -f ".env" ]; then
    msg "$YELLOW" "No .env file found. Creating from template..." "⚠️"
    cp .env.template .env
    msg "$GREEN" "Please edit .env file with your settings before running again" "📝"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Display configuration
msg "$BLUE" "Configuration:" "⚙️"
echo "  Base Directory: ${FOMO_BASE_DIR:-/projects/users/data/fomo25}"
echo "  Incoming Directory: ${FOMO_INCOMING_DIR:-/submissions/incoming}"
echo "  Output Directory: ${FOMO_OUTPUT_DIR:-/output}"
echo "  GPU Enabled: ${FOMO_ENABLE_GPU:-false}"
echo "  Container Timeout: ${FOMO_CONTAINER_TIMEOUT:-3600}s"

# Check if apptainer is available
if ! command -v "${FOMO_APPTAINER_EXECUTABLE:-apptainer}" &> /dev/null; then
    msg "$RED" "Apptainer not found. Please install apptainer first." "❌"
    exit 1
fi

msg "$GREEN" "Apptainer found: $(${FOMO_APPTAINER_EXECUTABLE:-apptainer} --version)" "✅"

# Check for containers
INCOMING_DIR="${FOMO_INCOMING_DIR:-/submissions/incoming}"
CONTAINER_COUNT=$(find "$INCOMING_DIR" -name "*.sif" 2>/dev/null | wc -l)

if [ "$CONTAINER_COUNT" -eq 0 ]; then
    msg "$YELLOW" "No containers found in $INCOMING_DIR" "⚠️"
    msg "$BLUE" "Please place your .sif containers in the incoming directory" "📂"
    exit 1
fi

msg "$GREEN" "Found $CONTAINER_COUNT container(s) to evaluate" "📦"

# List containers
msg "$BLUE" "Containers to process:" "📋"
find "$INCOMING_DIR" -name "*.sif" -exec basename {} \; | sed 's/^/  - /'

# Ask for confirmation if in interactive mode
if [ -t 0 ]; then
    echo ""
    read -p "Continue with evaluation? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        msg "$YELLOW" "Evaluation cancelled" "🚫"
        exit 0
    fi
fi

# Ensure directories exist
msg "$BLUE" "Ensuring directories exist..." "📁"
python -c "from src.utils.file_utils import ensure_directories; ensure_directories()"

# Run evaluation with instance support
msg "$BLUE" "Starting FOMO evaluation with container instances..." "🚀"
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Option to run in debug mode
if [ "${1:-}" = "--debug" ]; then
    msg "$BLUE" "Running in debug mode..." "🔍"
    export FOMO_DEBUG_MODE=true
    python src/main.py --debug
elif [ "${1:-}" = "--test" ] && [ -n "${2:-}" ]; then
    msg "$BLUE" "Running test on specific container..." "🧪"
    python debug_instance.py test "$2" "${3:-task1}"
else
    # Regular evaluation
    python src/main.py "$@"
fi

# Check results
RESULTS_DIR="${FOMO_RESULTS_DIR:-/results}"
RESULT_COUNT=$(find "$RESULTS_DIR" -name "*.json" 2>/dev/null | wc -l)

if [ "$RESULT_COUNT" -gt 0 ]; then
    msg "$GREEN" "Evaluation completed! Generated $RESULT_COUNT result file(s)" "🎉"
    msg "$BLUE" "Results saved in: $RESULTS_DIR" "📊"
    
    # Show recent results
    msg "$BLUE" "Recent results:" "📋"
    find "$RESULTS_DIR" -name "*.json" -newer "$0" 2>/dev/null | head -5 | sed 's/^/  - /'
else
    msg "$YELLOW" "No results generated. Check logs for errors." "⚠️"
fi

# Show logs location
LOGS_DIR="${FOMO_LOGS_DIR:-logs}"
LATEST_LOG=$(find "$LOGS_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)

if [ -n "$LATEST_LOG" ]; then
    msg "$BLUE" "Latest log file: $LATEST_LOG" "📝"
    
    # Show any errors from the log
    if grep -q "ERROR" "$LATEST_LOG" 2>/dev/null; then
        msg "$YELLOW" "Errors found in log:" "⚠️"
        grep "ERROR" "$LATEST_LOG" | tail -3 | sed 's/^/  /'
    fi
fi

msg "$GREEN" "Evaluation script completed" "✅"
