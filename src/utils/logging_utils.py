# src/utils/logging_utils.py
"""Simplified logging utilities."""

import logging
import sys
from datetime import datetime

from src.config.settings import SETTINGS


def setup_logging() -> logging.Logger:
    """Setup simplified application logging."""
    SETTINGS.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = SETTINGS.LOGS_DIR / f"fomo_evaluation_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger("fomo_evaluation")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)