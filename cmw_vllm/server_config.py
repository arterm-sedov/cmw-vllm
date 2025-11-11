"""Server configuration management."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    """vLLM server configuration."""

    model: str = Field(default="Qwen/Qwen3-30B-A3B-Instruct-2507", description="Model identifier")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    max_model_len: int | None = Field(default=40000, description="Maximum model length (reduced from 262144 for 48GB GPUs)")
    gpu_memory_utilization: float = Field(default=0.8, ge=0.0, le=1.0, description="GPU memory utilization")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size")
    cpu_offload_gb: int | None = Field(default=24, ge=0, description="CPU offload memory in GB (for large models on limited GPU memory)")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    download_dir: str | None = Field(default=None, description="Model download directory")
    enable_auto_tool_choice: bool = Field(default=True, description="Enable auto tool choice for function calling")
    tool_call_parser: str | None = Field(default="hermes", description="Tool call parser (hermes for Qwen models)")

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        max_model_len_env = os.getenv("VLLM_MAX_MODEL_LEN")
        cpu_offload_env = os.getenv("VLLM_CPU_OFFLOAD_GB")
        
        # Handle max_model_len: use env value if set, otherwise default to 40000
        max_model_len = int(max_model_len_env) if max_model_len_env else 40000
        
        # Handle cpu_offload_gb: use env value if set, otherwise default to 24
        # If explicitly set to "0", use 0 (None), otherwise default to 24
        if cpu_offload_env is None:
            cpu_offload_gb = 24
        elif cpu_offload_env == "0":
            cpu_offload_gb = None  # Disable CPU offloading
        else:
            cpu_offload_gb = int(cpu_offload_env)
        
        return cls(
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
            host=os.getenv("VLLM_HOST", "0.0.0.0"),
            port=int(os.getenv("VLLM_PORT", "8000")),
            max_model_len=max_model_len,
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.8")),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            cpu_offload_gb=cpu_offload_gb,
            trust_remote_code=os.getenv("VLLM_TRUST_REMOTE_CODE", "false").lower() == "true",
            download_dir=os.getenv("MODEL_DOWNLOAD_DIR") or None,
            enable_auto_tool_choice=os.getenv("VLLM_ENABLE_AUTO_TOOL_CHOICE", "true").lower() == "true",
            tool_call_parser=os.getenv("VLLM_TOOL_CALL_PARSER", "hermes") or None,
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

        if self.enable_auto_tool_choice:
            args.append("--enable-auto-tool-choice")

        if self.tool_call_parser:
            args.extend(["--tool-call-parser", self.tool_call_parser])

        return args
