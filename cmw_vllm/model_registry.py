"""Model registry and metadata."""
from __future__ import annotations

MODEL_REGISTRY = {
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        "name": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "size_gb": 30.5,  # Approximate size in GB
        "context_window": 262144,
        "architecture": "qwen3_moe",
        "description": "Qwen3 30B A3B Instruct model (Mixture of Experts)",
        "max_model_len": 40000,  # Reduced from 262144 for GPU memory constraints (48GB GPUs)
        "gpu_memory_utilization": 0.8,  # Default GPU memory utilization
        "cpu_offload_gb": 0,  # Disabled (causes issues with vLLM v1 engine)
        "trust_remote_code": False,  # Not required for Qwen3 models
        "enable_auto_tool_choice": True,  # Enable auto tool choice for function calling
        "tool_call_parser": "hermes",  # Default tool call parser for Qwen models
    },
    "openai/gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "size_gb": 40.0,  # Approximate size in GB for 20B parameter model
        "context_window": 131072,  # Model's native context window (128k-131k tokens)
        "max_model_len": 40000,  # vLLM max sequence length (reduced from 131k for GPU memory constraints)
        "architecture": "gpt",
        "description": "OpenAI GPT OSS 20B model",
        "gpu_memory_utilization": 0.6,  # Model-specific GPU memory utilization
        "cpu_offload_gb": 0,  # Disabled (causes issues with vLLM v1 engine)
        "trust_remote_code": False,  # Not required for GPT-OSS models
        "enable_auto_tool_choice": True,  # Required for function calling with gpt-oss models
        # Note: gpt-oss models require --tool-call-parser openai --enable-auto-tool-choice for function calling
        # See: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#function-calling
        "tool_call_parser": "openai",  # Required for function calling with gpt-oss models
        # Context length: 128,000-131,072 tokens (combined input + output)
    },
    "mistralai/Ministral-3-14B-Instruct-2512": {
        "name": "mistralai/Ministral-3-14B-Instruct-2512",
        "size_gb": 28.0,  # Approximate size in GB for 14B parameter model
        "context_window": 32768,  # Mistral models typically have 32K context window
        "architecture": "mistral",
        "description": "Mistral Ministral-3 14B Instruct model",
        "max_model_len": 32768,  # Match the model's context window
        "gpu_memory_utilization": 0.6,  # Same as Qwen for consistent memory usage
        "cpu_offload_gb": 0,  # Disabled (causes issues with vLLM v1 engine)
        "trust_remote_code": True,  # Required for custom tokenizer
        "enable_auto_tool_choice": True,  # Enable auto tool choice for function calling
        "tool_call_parser": "mistral",  # Mistral models use MistralToolParser for tool calling
        "tokenizer_mode": "mistral",  # Required for Mistral models per official vLLM docs
        "config_format": "mistral",  # Required for Mistral models per official vLLM docs
        "load_format": "mistral",  # Required for Mistral models per official vLLM docs
    },
    "ai-sage/GigaChat3-10B-A1.8B-bf16": {
        "name": "ai-sage/GigaChat3-10B-A1.8B-bf16",
        "size_gb": 24.0,  # Approximate size in GB for 10B + 1.8B MoE model in BF16
        "context_window": 256000,  # Поддерживает контекст до 256 тысяч токенов
        "architecture": "gigachat3_moe",
        "description": "GigaChat3 10B base with 1.8B active MoE experts (BF16)",
        # Recommended vLLM launch options from model card:
        # vllm serve ai-sage/GigaChat3-10B-A1.8B-bf16 \
        #   --dtype auto \
        #   --speculative-config '{"method": "mtp", "num_speculative_tokens": 1, "disable_padded_drafter_batch": false}'
        "max_model_len": 128000,  # Reduced to 128k for better concurrent request capacity
        "gpu_memory_utilization": 0.6,  # Reduced for multi-model GPU sharing
        "cpu_offload_gb": 0,  # Disabled for GigaChat3 (causes issues with vLLM v1 engine)
        "trust_remote_code": False,  # Not required for GigaChat3
        "dtype": "auto",
        "speculative_config": '{"method": "mtp", "num_speculative_tokens": 1, "disable_padded_drafter_batch": false}',
        "tool_call_parser": "gigachat3",  # Custom plugin parser for GigaChat3 models
        "enable_auto_tool_choice": True,  # Enable auto tool choice for function calling
    },
    "cerebras/Qwen3-Coder-REAP-25B-A3B": {
        "name": "cerebras/Qwen3-Coder-REAP-25B-A3B",
        "size_gb": 25.5,  # Approximate size in GB for 25B parameter MoE model
        "context_window": 262144,  # Qwen3 models typically have 256K context window
        "architecture": "qwen3_moe",
        "description": "Cerebras Qwen3 Coder REAP 25B A3B model (Mixture of Experts, code-specialized)",
        "max_model_len": 40000,  # Reduced from 262144 for GPU memory constraints (48GB GPUs)
        "gpu_memory_utilization": 0.8,  # Default GPU memory utilization
        "cpu_offload_gb": 0,  # Disabled (causes issues with vLLM v1 engine)
        "trust_remote_code": False,  # Not required for Qwen3-Coder models
        "enable_auto_tool_choice": True,  # Enable auto tool choice for function calling
        # Qwen3-Coder models use XML format for tool calling and require --tool-call-parser qwen3_xml
        # See: https://docs.vllm.ai/en/stable/features/tool_calling/?h=qwen3+coder#qwen3-coder-models-qwen3_xml
        "tool_call_parser": "qwen3_xml",  # Required for function calling with Qwen3-Coder models (XML format)
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
