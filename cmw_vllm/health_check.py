"""Health check utilities for vLLM server."""
from __future__ import annotations

import logging
from typing import Any

import requests

from cmw_vllm.server_config import ServerConfig

logger = logging.getLogger(__name__)


def check_server_status(base_url: str = "http://localhost:8000", timeout: int = 5) -> dict[str, Any]:
    """Check vLLM server status.

    Args:
        base_url: Base URL of the server
        timeout: Request timeout in seconds

    Returns:
        Dictionary with status information:
        - running: bool
        - models: list of available models
        - error: str or None
    """
    try:
        # Check /v1/models endpoint
        response = requests.get(f"{base_url}/v1/models", timeout=timeout)
        response.raise_for_status()

        data = response.json()
        models = [model["id"] for model in data.get("data", [])]

        return {
            "running": True,
            "models": models,
            "error": None,
        }

    except requests.exceptions.ConnectionError:
        return {
            "running": False,
            "models": [],
            "error": "Cannot connect to server. Is it running?",
        }
    except requests.exceptions.Timeout:
        return {
            "running": False,
            "models": [],
            "error": "Server request timed out",
        }
    except Exception as e:
        return {
            "running": False,
            "models": [],
            "error": f"Error checking server status: {e}",
        }


def test_inference(
    base_url: str = "http://localhost:8000",
    model: str = "openai/gpt-oss-20b",
    timeout: int = 30,
) -> dict[str, Any]:
    """Test inference with a simple request.

    Args:
        base_url: Base URL of the server
        model: Model identifier
        timeout: Request timeout in seconds

    Returns:
        Dictionary with test results:
        - success: bool
        - response: str or None
        - error: str or None
    """
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_tokens": 10,
            },
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "success": True,
            "response": content,
            "error": None,
        }

    except Exception as e:
        return {
            "success": False,
            "response": None,
            "error": f"Inference test failed: {e}",
        }


def get_server_info(config: ServerConfig | None = None) -> dict[str, Any]:
    """Get comprehensive server information.

    Args:
        config: Server configuration. If None, loads from environment.

    Returns:
        Dictionary with server information
    """
    if config is None:
        config = ServerConfig.from_env()

    base_url = f"http://{config.host}:{config.port}"
    status = check_server_status(base_url)

    info = {
        "config": {
            "model": config.model,
            "host": config.host,
            "port": config.port,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
        },
        "status": status,
    }

    if status["running"]:
        # Test inference
        test_result = test_inference(base_url, config.model)
        info["inference_test"] = test_result

    return info
