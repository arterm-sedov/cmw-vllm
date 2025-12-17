"""Server configuration management."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field

from cmw_vllm.model_config_patcher import get_model_config_value
from cmw_vllm.model_registry import get_model_info


class ServerConfig(BaseModel):
    """vLLM server configuration."""

    model: str = Field(default="openai/gpt-oss-20b", description="Model identifier")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    max_model_len: int | None = Field(default=40000, description="Maximum model length (reduced from 262144 for 48GB GPUs)")
    gpu_memory_utilization: float = Field(default=0.8, ge=0.0, le=1.0, description="GPU memory utilization")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size")
    cpu_offload_gb: int | None = Field(default=24, ge=0, description="CPU offload memory in GB (for large models on limited GPU memory)")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    download_dir: str | None = Field(default=None, description="Model download directory")
    enable_auto_tool_choice: bool = Field(default=True, description="Enable auto tool choice for function calling")
    tool_call_parser: str | None = Field(default="hermes", description="Tool call parser (mistral for Mistral models, hermes for Qwen models)")
    tokenizer_mode: str | None = Field(default=None, description="Tokenizer mode (mistral for Mistral models)")
    config_format: str | None = Field(default=None, description="Config format (mistral for Mistral models)")
    load_format: str | None = Field(default=None, description="Load format (mistral for Mistral models)")
    dtype: str | None = Field(default=None, description='Model dtype passed to vLLM (e.g., "auto", "float16", "bfloat16")')
    speculative_config: str | None = Field(
        default=None,
        description="JSON string passed to vLLM --speculative-config (e.g. MTP settings)",
    )

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")
        cpu_offload_env = os.getenv("VLLM_CPU_OFFLOAD_GB")
        
        # Get model ID
        model_id = os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")
        
        # Get model-specific defaults from registry
        model_info = get_model_info(model_id)
        
        # Handle max_model_len: use env value if set, otherwise check model registry, then default to 40000
        if max_model_len_env:
            max_model_len = int(max_model_len_env)
        elif model_info and "max_model_len" in model_info:
            max_model_len = model_info["max_model_len"]
        else:
            max_model_len = 40000
        
        # Handle cpu_offload_gb: use env value if set, otherwise check model registry, then default to 24
        # If explicitly set to "0", use 0 (None), otherwise default to 24
        if cpu_offload_env is not None:
            if cpu_offload_env == "0":
                cpu_offload_gb = None  # Disable CPU offloading
            else:
                cpu_offload_gb = int(cpu_offload_env)
        elif model_info and "cpu_offload_gb" in model_info:
            cpu_offload_gb = model_info["cpu_offload_gb"]
        else:
            cpu_offload_gb = 24
        
        # Handle gpu_memory_utilization: use env value if set, otherwise check model registry, then default to 0.8
        gpu_memory_utilization_env = os.getenv("VLLM_GPU_MEMORY_UTILIZATION")
        if gpu_memory_utilization_env:
            gpu_memory_utilization = float(gpu_memory_utilization_env)
        elif model_info and "gpu_memory_utilization" in model_info:
            gpu_memory_utilization = model_info["gpu_memory_utilization"]
        else:
            gpu_memory_utilization = 0.8
        
        # Handle trust_remote_code: use env value if set, otherwise check model registry, then default to False
        trust_remote_code_env = os.getenv("VLLM_TRUST_REMOTE_CODE")
        if trust_remote_code_env is not None:
            trust_remote_code = trust_remote_code_env.lower() == "true"
        elif model_info and "trust_remote_code" in model_info:
            trust_remote_code = model_info["trust_remote_code"]
        else:
            trust_remote_code = False
        
        # Handle tool_call_parser: use env value if set, otherwise check model config.json, 
        # then model registry, then default to "hermes"
        # Note: gpt-oss models require --tool-call-parser openai for function calling
        # See: https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html#function-calling
        tool_call_parser_env = os.getenv("VLLM_TOOL_CALL_PARSER")
        if tool_call_parser_env:
            tool_call_parser = tool_call_parser_env
        elif model_id.startswith("openai/gpt-oss"):
            # gpt-oss models require --tool-call-parser openai for function calling
            tool_call_parser = "openai"
        else:
            # Try to read from model config.json file (if model authors included it)
            tool_call_parser_from_config = get_model_config_value(model_id, "tool_call_parser")
            if tool_call_parser_from_config:
                tool_call_parser = tool_call_parser_from_config
            elif model_info and "tool_call_parser" in model_info:
                # Fall back to registry (for models not yet updated with config.json)
                tool_call_parser = model_info["tool_call_parser"]
            else:
                tool_call_parser = "hermes"
        
        # Handle tokenizer_mode, config_format, load_format: check model registry
        tokenizer_mode = None
        config_format = None
        load_format = None
        dtype = None
        speculative_config = None
        if model_info:
            tokenizer_mode = model_info.get("tokenizer_mode")
            config_format = model_info.get("config_format")
            load_format = model_info.get("load_format")
            dtype = model_info.get("dtype")
            speculative_config = model_info.get("speculative_config")
        
        # Allow env var overrides
        if os.getenv("VLLM_TOKENIZER_MODE"):
            tokenizer_mode = os.getenv("VLLM_TOKENIZER_MODE")
        if os.getenv("VLLM_CONFIG_FORMAT"):
            config_format = os.getenv("VLLM_CONFIG_FORMAT")
        if os.getenv("VLLM_LOAD_FORMAT"):
            load_format = os.getenv("VLLM_LOAD_FORMAT")
        if os.getenv("VLLM_DTYPE"):
            dtype = os.getenv("VLLM_DTYPE")
        if os.getenv("VLLM_SPECULATIVE_CONFIG"):
            speculative_config = os.getenv("VLLM_SPECULATIVE_CONFIG")
        
        # Handle enable_auto_tool_choice: use env value if set, otherwise check model registry, then default to True
        enable_auto_tool_choice_env = os.getenv("VLLM_ENABLE_AUTO_TOOL_CHOICE")
        if enable_auto_tool_choice_env is not None:
            enable_auto_tool_choice = enable_auto_tool_choice_env.lower() == "true"
        elif model_info and "enable_auto_tool_choice" in model_info:
            enable_auto_tool_choice = model_info["enable_auto_tool_choice"]
        else:
            enable_auto_tool_choice = True
        
        return cls(
            model=model_id,
            host=os.getenv("VLLM_HOST", "0.0.0.0"),
            port=int(os.getenv("VLLM_PORT", "8000")),
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            cpu_offload_gb=cpu_offload_gb,
            trust_remote_code=trust_remote_code,
            download_dir=os.getenv("MODEL_DOWNLOAD_DIR") or None,
            enable_auto_tool_choice=enable_auto_tool_choice,
            tool_call_parser=tool_call_parser,
            tokenizer_mode=tokenizer_mode,
            config_format=config_format,
            load_format=load_format,
            dtype=dtype,
            speculative_config=speculative_config,
        )

    def to_vllm_args(self) -> list[str]:
        """Convert to vLLM command-line arguments.

        Returns:
            List of command-line arguments
        """
        args = [
            "--model",
            self.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
        ]

        if self.max_model_len:
            args.extend(["--max-model-len", str(self.max_model_len)])

        if self.cpu_offload_gb:
            args.extend(["--cpu-offload-gb", str(self.cpu_offload_gb)])

        if self.trust_remote_code:
            args.append("--trust-remote-code")

        if self.download_dir:
            args.extend(["--download-dir", self.download_dir])

        if self.dtype:
            args.extend(["--dtype", self.dtype])

        # Only add --enable-auto-tool-choice if tool_call_parser is also set,
        # as vLLM requires --tool-call-parser when --enable-auto-tool-choice is used
        if self.enable_auto_tool_choice and self.tool_call_parser:
            args.append("--enable-auto-tool-choice")
            args.extend(["--tool-call-parser", self.tool_call_parser])
        elif self.tool_call_parser:
            # Add tool_call_parser even if enable_auto_tool_choice is False
            args.extend(["--tool-call-parser", self.tool_call_parser])

        if self.tokenizer_mode:
            args.extend(["--tokenizer-mode", self.tokenizer_mode])

        if self.config_format:
            args.extend(["--config-format", self.config_format])

        if self.load_format:
            args.extend(["--load-format", self.load_format])

        if self.speculative_config:
            # Pass JSON string directly; vLLM CLI will parse it
            args.extend(["--speculative-config", self.speculative_config])

        return args
