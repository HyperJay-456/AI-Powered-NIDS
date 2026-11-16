"""Standardized logging configuration for the NIDS platform."""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "nids", log_file: Optional[Path] = None) -> logging.Logger:
    """Initialize and return a configured logger instance.

    Args:
        name: Name of the logger.
        log_file: Optional path to write log files.

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console stream handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if log_file is specified
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
