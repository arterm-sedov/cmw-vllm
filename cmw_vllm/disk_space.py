"""Disk space checking utilities."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def get_free_space_gb(path: Path | str) -> float:
    """Get free disk space in GB for the given path.

    Args:
        path: Path to check disk space for

    Returns:
        Free space in GB, or -1 if unable to determine
    """
    try:
        path_obj = Path(path).resolve()
        if path_obj.exists():
            stat = shutil.disk_usage(path_obj)
        else:
            # If path doesn't exist, check parent directory
            stat = shutil.disk_usage(path_obj.parent if path_obj.parent.exists() else Path.cwd())
        return stat.free / (1024**3)  # Convert to GB
    except Exception as e:
        logger.warning(f"Unable to determine free disk space: {e}")
        return -1.0


def check_disk_space_available(required_gb: float, path: Path | str | None = None) -> tuple[bool, float, str]:
    """Check if sufficient disk space is available.

    Args:
        required_gb: Required space in GB
        path: Path to check (defaults to current working directory)

    Returns:
        Tuple of (is_available, free_gb, message)
    """
    check_path = Path(path) if path else Path.cwd()
    free_gb = get_free_space_gb(check_path)

    if free_gb < 0:
        return True, -1.0, "Unable to check disk space - proceeding with caution"

    buffer_multiplier = 1.2  # 20% buffer for safety
    required_with_buffer = required_gb * buffer_multiplier
    available = free_gb >= required_with_buffer

    if available:
        message = f"Sufficient disk space: {free_gb:.2f} GB available (requires {required_gb:.2f} GB)"
    else:
        needed = required_with_buffer - free_gb
        message = (
            f"Insufficient disk space: {free_gb:.2f} GB available, "
            f"but {required_gb:.2f} GB required (with 20% buffer: {required_with_buffer:.2f} GB). "
            f"Please free up at least {needed:.2f} GB of space."
        )

    return available, free_gb, message
