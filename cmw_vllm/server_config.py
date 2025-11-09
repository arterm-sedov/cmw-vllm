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
    max_model_len: int | None = Field(default=None, description="Maximum model length")
    gpu_memory_utilization: float = Field(default=0.9, ge=0.0, le=1.0, description="GPU memory utilization")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    download_dir: str | None = Field(default=None, description="Model download directory")

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create configuration from environment variables."""
        return cls(
            model=os.getenv("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507"),
            host=os.getenv("VLLM_HOST", "0.0.0.0"),
            port=int(os.getenv("VLLM_PORT", "8000")),
            max_model_len=int(os.getenv("VLLM_MAX_MODEL_LEN", "0")) or None,
            gpu_memory_utilization=float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.9")),
            tensor_parallel_size=int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")),
            trust_remote_code=os.getenv("VLLM_TRUST_REMOTE_CODE", "false").lower() == "true",
            download_dir=os.getenv("MODEL_DOWNLOAD_DIR") or None,
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

        if self.trust_remote_code:
            args.append("--trust-remote-code")

        if self.download_dir:
            args.extend(["--download-dir", self.download_dir])

        return args
