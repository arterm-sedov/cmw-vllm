# Agent Guide for cmw-vllm

This document provides guidance for AI agents working on the cmw-vllm project. Rule set for opencode: keep solutions lean; do not overengineer.

## Git & Commits

- **Do NOT create or push commits automatically.** The user reviews all commits first. You may suggest commit messages or stage files only when explicitly asked.
- If generating a commit message: keep it concise, structured, and strictly relevant to the changes. Do not add, stage, or push.

## Project Overview

cmw-vllm is a CLI tool for managing vLLM inference servers. It provides:
- Server lifecycle management (start, stop, status)
- Model downloading and verification from HuggingFace
- Process management with PID files
- Health checking and inference testing
- GPU and disk space monitoring

## Architecture

```
cmw_vllm/
├── __init__.py          # Package exports
├── cli.py              # Click CLI commands
├── server_config.py    # Pydantic schemas and server configuration
├── server_manager.py   # Process management for vLLM servers
├── model_downloader.py # HuggingFace model downloading
├── model_verifier.py   # Model integrity verification
├── model_registry.py   # Pre-configured model definitions
├── gpu_info.py         # GPU detection and monitoring
├── disk_space.py       # Disk space monitoring
├── health_check.py     # HTTP health checks and inference testing
└── logging.py          # Logging utilities
```

## Key Components

### ServerConfig (Pydantic)
Defines vLLM server configurations including:
- model_id: HuggingFace model identifier
- port: Server port (unique per model)
- tensor_parallel_size: Number of GPUs for tensor parallelism
- gpu_memory_utilization: GPU memory fraction to use
- max_model_len: Maximum sequence length
- dtype: Data type (auto, float16, bfloat16, float32)

### ServerManager
Manages vLLM server processes:
- start(): Launch server in background with configured parameters
- stop(): Graceful shutdown with fallback to force kill
- get_status(): Check if server is running and responding
- list_running(): List all servers with PID files

### Model Management
- ModelRegistry: Pre-configured popular models with optimized settings
- ModelDownloader: Download from HuggingFace with progress tracking
- ModelVerifier: Verify model integrity and compatibility

### CLI Commands
- setup: Verify vLLM installation and GPU availability
- start <model>: Start server for model with auto-detection
- stop <model>: Stop server gracefully
- status: Show running servers and their status
- list: Show available models in registry
- download <model>: Download model from HuggingFace

## Dependencies

Core:
- click: CLI framework
- pydantic: Data validation
- requests: HTTP health checks
- huggingface-hub: Model downloading

External (user-installed):
- vllm: The actual inference server
- torch: For GPU detection

## Error Handling

- Use try/except around process and download operations
- Log errors with logger, not print
- Return structured results from manager methods
- CLI catches exceptions and exits with appropriate codes
- Handle GPU OOM and CUDA errors gracefully

## Platform Notes

- Windows: Limited signal support, use process termination
- Linux/macOS: Full signal support for graceful shutdown
- PID files stored in ~/.cmw-vllm/
- Models cached in HuggingFace default location (~/.cache/huggingface/)

## Development

- Activate the project venv before running Python or tests (e.g. `.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Linux/macOS).

## Testing

Test scenarios:
1. Start/stop Qwen server
2. Health check via HTTP API
3. Model download and verification
4. Multiple start calls (idempotent)
5. Stop non-running server
6. List running servers
7. GPU memory calculation

## Agent Behavior

- **Planning:** Plan your course of action before implementing.
- **Verification:** Run `ruff check <modified_file>` after changes. Run relevant tests. Reanalyze changes for introduced issues.
- **Linting:** Only lint files that were modified, not the entire codebase. Be critical about Ruff reports; implement only necessary changes.
- **Secrets:** Never hardcode secrets. Use environment variables.
- **No breakage:** Never break existing code.

### 12-Factor App Principles
Following twelve-factor methodology for CLI/server tools:

- **Codebase:** One codebase tracked in revision control, many deploys.
- **Dependencies:** Declare all dependencies explicitly in `pyproject.toml`. See Development section for venv activation.
- **Config:** Store all environment-specific config in env vars (never in code). Use `.env` files for local development.
- **Backing Services:** Treat vLLM servers as attached resources. Server config (ports, model paths) via env vars or CLI args.
- **Build, Release, Run:** Strictly separate build and run stages. Install package once, run anywhere.
- **Processes:** Execute the app as stateless processes. Server state externalized (PID files, process management).
- **Port Binding:** Each vLLM server exports service via configurable port. CLI binds to no port (local tool).
- **Concurrency:** Scale out by running multiple server processes (one per model/GPU). CLI is single-user.
- **Disposability:** Maximize robustness with fast startup and graceful shutdown. Servers start quickly and handle SIGTERM gracefully, finishing current requests before exiting.
- **Dev/Prod Parity:** Keep development and production vLLM versions identical. Use same model versions across environments.
- **Logs:** Treat logs as event streams. vLLM servers log to stdout/stderr; CLI logs to console. Optional file logging via `LOG_FILE_ENABLED` env var.
- **Admin Processes:** Run admin tasks (model downloads, health checks, GPU monitoring) as one-off processes using the same CLI tool.

## Code Style

- Follow Google docstring convention. Type hints required. Line length: 100. Use ruff for linting.
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes, `UPPER_CASE` for constants.
- **Imports:** At top of file; ruff handles sorting.
- **Comments:** Explain why, not what. Do not delete existing comments or logging; update if needed.
- **Error handling:** Avoid unnecessary try/except. Catch only when necessary and meaningful. Prefer robust, explicit logic over hardcoded fallbacks.
