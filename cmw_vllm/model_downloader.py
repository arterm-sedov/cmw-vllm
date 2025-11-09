"""Model downloading utilities."""
from __future__ import annotations

import logging
import os
from pathlib import Path

from tqdm import tqdm

from cmw_vllm.disk_space import check_disk_space_available
from cmw_vllm.model_registry import estimate_model_size

logger = logging.getLogger(__name__)


def download_model(
    model_id: str,
    local_dir: Path | str | None = None,
    resume: bool = True,
    check_space: bool = True,
) -> Path:
    """Download model from HuggingFace.

    Args:
        model_id: HuggingFace model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507")
        local_dir: Local directory to download to. If None, uses HuggingFace cache.
        resume: Whether to resume interrupted downloads
        check_space: Whether to check disk space before downloading

    Returns:
        Path to downloaded model

    Raises:
        ValueError: If insufficient disk space
        RuntimeError: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface-hub is required for model downloading. "
            "Install it with: pip install huggingface-hub"
        )

    # Estimate model size
    estimated_size = estimate_model_size(model_id)
    if estimated_size > 0 and check_space:
        available, free_gb, message = check_disk_space_available(estimated_size, local_dir)
        if not available:
            raise ValueError(f"Insufficient disk space: {message}")
        logger.info(message)

    # Determine download location
    if local_dir:
        download_path = Path(local_dir) / model_id.replace("/", "--")
    else:
        # Use HuggingFace cache
        cache_dir = os.getenv("HF_HOME")
        if cache_dir:
            download_path = Path(cache_dir) / "hub" / model_id
        else:
            cache_home = os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"
            download_path = Path(cache_home) / "huggingface" / "hub" / model_id

    logger.info(f"Downloading model {model_id} to {download_path}")

    try:
        # Download with progress bar
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=str(download_path) if local_dir else None,
            resume_download=resume,
            local_dir_use_symlinks=False,
        )

        logger.info(f"Model downloaded successfully to {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise RuntimeError(f"Model download failed: {e}") from e


def check_model_downloaded(model_id: str, local_dir: Path | str | None = None) -> tuple[bool, Path | None]:
    """Check if model is already downloaded.

    Args:
        model_id: HuggingFace model identifier
        local_dir: Local directory to check. If None, checks HuggingFace cache.

    Returns:
        Tuple of (is_downloaded, model_path)
    """
    if local_dir:
        model_path = Path(local_dir) / model_id.replace("/", "--")
    else:
        cache_home = os.getenv("HF_HOME") or os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"
        model_path = Path(cache_home) / "huggingface" / "hub" / model_id

    # Check if config.json exists (indicates model is downloaded)
    config_file = model_path / "config.json"
    if config_file.exists():
        return True, model_path

    return False, None
