"""Model verification utilities."""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def verify_model_integrity(model_path: Path, model_id: str) -> tuple[bool, str]:
    """Verify model files are complete and valid.

    Args:
        model_path: Path to model directory
        model_id: Model identifier for reference

    Returns:
        Tuple of (is_valid, error_message)
    """
    model_path = Path(model_path)

    if not model_path.exists():
        return False, f"Model path does not exist: {model_path}"

    # Check for essential files
    required_files = ["config.json", "tokenizer.json"]
    missing_files = []

    for file_name in required_files:
        file_path = model_path / file_name
        if not file_path.exists():
            missing_files.append(file_name)

    if missing_files:
        return False, f"Missing required files: {', '.join(missing_files)}"

    # Check for model weights (safetensors or pytorch)
    has_safetensors = any((model_path / f).exists() for f in model_path.glob("*.safetensors"))
    has_pytorch = any((model_path / f).exists() for f in model_path.glob("*.bin"))
    has_index = (model_path / "model.safetensors.index.json").exists()

    if not (has_safetensors or has_pytorch):
        return False, "No model weights found (no .safetensors or .bin files)"

    if has_safetensors and not has_index:
        # Check if it's a single file or multiple files
        safetensor_files = list(model_path.glob("*.safetensors"))
        if len(safetensor_files) > 1:
            return False, "Multiple safetensors files found but no index file"

    logger.info(f"Model verification passed for {model_id}")
    return True, "Model files are valid"
