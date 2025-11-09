"""Logging configuration for cmw-vllm."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None) -> None:
    """Setup logging configuration.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). If None, reads from LOG_LEVEL env var.
    """
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO").upper()

    log_level = getattr(logging, level, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        stream=sys.stdout,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
