"""
Logging utilities for the recommendation system pipeline.
Provides structured logging with file rotation and console output.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorlog


class ColoredFormatter(colorlog.ColoredFormatter):
    """Custom colored formatter for console output."""
    
    def __init__(self, fmt: str, datefmt: Optional[str] = None):
        super().__init__(
            fmt=fmt,
            datefmt=datefmt,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )


def setup_logger(
    name: str,
    log_dir: str = "logs",
    level: str = "INFO",
    console_output: bool = True,
    file_output: bool = True,
    rotation_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        console_output: Enable console output
        file_output: Enable file output
        rotation_bytes: Max file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Format strings
    detailed_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    simple_format = "%(levelname)s - %(message)s"
    
    # Console handler with colors
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_formatter = ColoredFormatter(
            fmt="%(log_color)s" + detailed_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{name}_{timestamp}.log"
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=rotation_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt=detailed_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Also create a latest.log symlink
        latest_log = log_path / "latest.log"
        try:
            if latest_log.exists() or latest_log.is_symlink():
                latest_log.unlink()
            latest_log.symlink_to(log_file)
        except (OSError, NotImplementedError):
            pass  # Symlinks not supported on all systems
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger by name.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class StageLogger:
    """Context manager for stage-specific logging."""
    
    def __init__(self, logger: logging.Logger, stage_name: str):
        self.logger = logger
        self.stage_name = stage_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Starting stage: {self.stage_name}")
        self.logger.info(f"{'='*60}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.info(f"{'='*60}")
            self.logger.info(f"Completed stage: {self.stage_name} in {duration}")
            self.logger.info(f"{'='*60}")
        else:
            self.logger.error(f"{'='*60}")
            self.logger.error(f"Failed stage: {self.stage_name} after {duration}")
            self.logger.error(f"Error: {exc_type.__name__}: {exc_val}")
            self.logger.error(f"{'='*60}")
        return False


def log_stage(logger: logging.Logger, stage_name: str):
    """Decorator for logging function execution as a stage."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with StageLogger(logger, stage_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
