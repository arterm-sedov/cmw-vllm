"""Model configuration utilities for reading model config values."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_model_config_value(model_id: str, key: str, model_path: Path | None = None) -> str | None:
    """Get a value from model config.json file.
    
    Args:
        model_id: HuggingFace model identifier
        key: Config key to read (supports dot notation for nested keys, e.g., "text_config.model_type")
        model_path: Path to model directory. If None, tries to find in cache.
        
    Returns:
        Config value or None if not found
    """
    # Find config file
    if model_path:
        config_path = Path(model_path) / "config.json"
    else:
        # Try to find in HuggingFace cache
        try:
            from huggingface_hub import hf_hub_download
            config_path = Path(hf_hub_download(repo_id=model_id, filename="config.json"))
        except Exception as e:
            logger.debug(f"Could not find config file for {model_id}: {e}")
            return None
    
    if not config_path.exists():
        logger.debug(f"Config file not found at {config_path}")
        return None
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Navigate to the key
        keys = key.split(".")
        current = config
        for k in keys:
            if not isinstance(current, dict) or k not in current:
                return None
            current = current[k]
        
        return str(current) if current is not None else None
        
    except Exception as e:
        logger.debug(f"Failed to read config for {model_id}: {e}")
        return None
