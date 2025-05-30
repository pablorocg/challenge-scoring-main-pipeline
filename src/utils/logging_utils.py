"""Logging utilities."""

import logging
import sys
from pathlib import Path
from datetime import datetime

from config.settings import SETTINGS


def setup_logging() -> logging.Logger:
    """Setup application logging."""
    # Create logs directory
    SETTINGS.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = SETTINGS.LOGS_DIR / f"fomo_evaluation_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("fomo_evaluation")
    logger.info("Logging initialized")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
