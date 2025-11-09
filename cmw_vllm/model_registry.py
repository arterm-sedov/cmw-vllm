"""Model registry and metadata."""
from __future__ import annotations

MODEL_REGISTRY = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "size_gb": 30.5,  # Approximate size in GB
        "context_window": 262144,
        "architecture": "qwen3_moe",
        "description": "Qwen3 30B A3B Instruct model (Mixture of Experts)",
    },
}


def get_model_info(model_id: str) -> dict | None:
    """Get model information from registry.

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen3-30B-A3B-Instruct-2507")

    Returns:
        Model info dict or None if not found
    """
    return MODEL_REGISTRY.get(model_id)


def estimate_model_size(model_id: str) -> float:
    """Estimate model size in GB.

    Args:
        model_id: Model identifier

    Returns:
        Estimated size in GB, or 0 if unknown
    """
    info = get_model_info(model_id)
    if info:
        return info.get("size_gb", 0.0)

    # Try to get from HuggingFace API if not in registry
    try:
        from huggingface_hub import model_info

        model_info_obj = model_info(model_id)
        if hasattr(model_info_obj, "safetensors"):
            # Estimate from safetensors metadata
            total_size = model_info_obj.safetensors.get("total", {}).get("BF16", 0)
            return total_size / (1024**3)  # Convert bytes to GB
    except Exception:
        pass

    return 0.0
