"""GPU detection and information utilities."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_gpu_info() -> dict:
    """Get GPU information.

    Returns:
        Dictionary with GPU information:
        - available: bool
        - count: int
        - name: str (first GPU name)
        - memory_total_gb: float (total memory in GB)
        - memory_free_gb: float (free memory in GB)
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "available": False,
                "count": 0,
                "name": None,
                "memory_total_gb": 0.0,
                "memory_free_gb": 0.0,
            }

        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"

        # Get memory info
        memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        memory_free = memory_total - memory_allocated

        return {
            "available": True,
            "count": device_count,
            "name": device_name,
            "memory_total_gb": memory_total,
            "memory_free_gb": memory_free,
        }
    except ImportError:
        logger.warning("PyTorch not available. Cannot detect GPU.")
        return {
            "available": False,
            "count": 0,
            "name": None,
            "memory_total_gb": 0.0,
            "memory_free_gb": 0.0,
        }
    except Exception as e:
        logger.warning(f"Error detecting GPU: {e}")
        return {
            "available": False,
            "count": 0,
            "name": None,
            "memory_total_gb": 0.0,
            "memory_free_gb": 0.0,
        }


def check_gpu_memory_available(required_gb: float) -> tuple[bool, dict, str]:
    """Check if sufficient GPU memory is available.

    Args:
        required_gb: Required GPU memory in GB

    Returns:
        Tuple of (is_available, gpu_info, message)
    """
    gpu_info = get_gpu_info()

    if not gpu_info["available"]:
        return False, gpu_info, "No GPU available"

    available_gb = gpu_info["memory_free_gb"]
    is_available = available_gb >= required_gb

    if is_available:
        message = (
            f"Sufficient GPU memory: {available_gb:.2f} GB free "
            f"(requires {required_gb:.2f} GB)"
        )
    else:
        needed = required_gb - available_gb
        message = (
            f"Insufficient GPU memory: {available_gb:.2f} GB free, "
            f"but {required_gb:.2f} GB required. "
            f"Need {needed:.2f} GB more."
        )

    return is_available, gpu_info, message
