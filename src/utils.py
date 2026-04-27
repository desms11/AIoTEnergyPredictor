"""Utility functions for AIoT Energy Predictor pipeline."""

import logging
import os
from pathlib import Path

# Configure logging
def setup_logging(name=__name__, level=logging.INFO):
    """
    Configure logging for the project.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create console handler with formatting
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


# Path resolution
def get_project_root():
    """
    Get the project root directory.
    
    Returns:
        Path: Path to project root
    """
    return Path(__file__).parent.parent


def get_data_dir():
    """Get path to data directory."""
    return get_project_root() / "data"


def get_models_dir():
    """Get path to models directory."""
    return get_project_root() / "models"


def get_results_dir():
    """Get path to results directory."""
    return get_project_root() / "results"


def ensure_directories_exist():
    """Ensure all required directories exist."""
    dirs = [get_data_dir(), get_models_dir(), get_results_dir()]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


# Configuration
class Config:
    """Configuration for the AIoT Energy Predictor pipeline."""
    
    # Data simulation parameters
    START_DATE = "2023-01-01"
    END_DATE = "2023-12-31"
    RANDOM_SEED = 42
    
    # Processing parameters
    LOOKBACK_WINDOW = 24  # hours
    MISSING_DATA_RATIO = 0.02  # 2%
    NOISE_LEVEL = 0.03  # 3%
    
    # Model parameters
    LSTM_UNITS_1 = 64
    LSTM_UNITS_2 = 32
    DROPOUT_RATE = 0.2
    BATCH_SIZE = 32
    EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 5
    LEARNING_RATE = 0.001
    
    # Split parameters
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
