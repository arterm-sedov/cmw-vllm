# CMW vLLM

vLLM server management tool for CMW projects. Provides easy setup, model downloading, and server management for vLLM inference servers.

## AI-Enabled Repo

Chat with DeepWiki to get answers about this repo:

[Ask DeepWiki](https://deepwiki.com/arterm-sedov/cmw-vllm)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/arterm-sedov/cmw-vllm)

## Features

- **Easy Setup**: One-command installation and verification
- **Model Management**: Download and verify models from HuggingFace
- **Server Management**: Start, stop, and monitor vLLM servers
- **Health Checks**: Verify server status and test inference
- **Configuration**: Environment-based configuration with sensible defaults

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/cmw-vllm.git
cd cmw-vllm

# Install
pip install -e .

# Or install from git
pip install git+https://github.com/your-org/cmw-vllm.git
```

## Quick Start

### 1. Setup

```bash
cmw-vllm setup
```

This verifies:
- vLLM installation
- GPU availability
- Required dependencies

### 2. Download Model

```bash
# Download Qwen3-30B-A3B-Instruct-2507 (default)
cmw-vllm download

# Or specify a different model
cmw-vllm download Qwen/Qwen3-30B-A3B-Instruct-2507
```

### 3. Start Server

```bash
# Start with default configuration
cmw-vllm start

# Or customize
cmw-vllm start --model Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8000
```

### 4. Check Status

```bash
# Check if server is running
cmw-vllm status

# Test inference
cmw-vllm status --test-inference

# Get detailed information
cmw-vllm info
```

### 5. Stop Server

```bash
cmw-vllm stop
```

## Configuration

Configuration is done via environment variables. Create a `.env` file:

```bash
# .env
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
VLLM_PORT=8000
VLLM_HOST=0.0.0.0
VLLM_MAX_MODEL_LEN=40000
VLLM_GPU_MEMORY_UTILIZATION=0.8
VLLM_CPU_OFFLOAD_GB=24
VLLM_TENSOR_PARALLEL_SIZE=1
```

**Note:** Default configuration is optimized for 48GB GPUs (RTX 4090, A6000, etc.):
- `max_model_len=40000`: Reduced from 262144 to fit KV cache in available GPU memory
- `gpu_memory_utilization=0.8`: Leaves headroom for other processes
- `cpu_offload_gb=24`: Offloads model weights to CPU RAM for large models

See `.env.example` for all available options.

## Commands

### `cmw-vllm setup`
Initial setup and verification. Checks vLLM installation, GPU availability, and dependencies.

### `cmw-vllm download [MODEL_ID]`
Download model from HuggingFace. Defaults to `Qwen/Qwen3-30B-A3B-Instruct-2507`.

**Options:**
- `--local-dir PATH`: Download to specific directory
- `--no-resume`: Don't resume interrupted downloads
- `--skip-space-check`: Skip disk space check

### `cmw-vllm start`
Start vLLM server.

**Options:**
- `--model MODEL`: Model identifier
- `--port PORT`: Server port (default: 8000)
- `--host HOST`: Server host (default: 0.0.0.0)
- `--max-model-len LEN`: Maximum model length (default: 40000 for 48GB GPUs)
- `--gpu-memory-utilization FLOAT`: GPU memory utilization (0.0-1.0, default: 0.8)
- `--cpu-offload-gb INT`: CPU offload memory in GB (default: 24 for large models)
- `--foreground, -f`: Run in foreground (don't detach)

### `cmw-vllm stop`
Stop vLLM server.

### `cmw-vllm restart`
Restart vLLM server.

### `cmw-vllm status`
Check server status.

**Options:**
- `--base-url URL`: Server base URL (default: http://localhost:8000)
- `--test-inference`: Test inference with a simple request

### `cmw-vllm info`
Show comprehensive server information including configuration and status.

### `cmw-vllm verify [MODEL_ID]`
Verify model is downloaded and valid.

## Integration with cmw-rag

To use vLLM with `cmw-rag`, configure it as a provider:

```bash
# In cmw-rag/.env
DEFAULT_LLM_PROVIDER=vllm
DEFAULT_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

Then `cmw-rag` will connect to the vLLM server via HTTP (OpenAI-compatible API).

## Testing

The repository includes comprehensive tests for inference and tool calling functionality.

### Standalone Test Script

Run the standalone test script for quick manual testing:

```bash
# Test with default server and model
python tests/test_gigachat3_standalone.py

# Test with custom server URL
python tests/test_gigachat3_standalone.py http://localhost:8000

# Test with custom server and model
python tests/test_gigachat3_standalone.py http://localhost:8000 ai-sage/GigaChat3-10B-A1.8B-bf16
```

The standalone script tests:
- Simple inference
- Tool calling
- Complete tool calling flow (with tool results)
- Streaming inference
- Streaming with tool calls

### Pytest Tests

Run pytest tests (requires `pytest` to be installed):

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_gigachat3_inference_and_tools.py

# Run with verbose output
pytest tests/ -v
```

Test files:
- `tests/test_gigachat3_inference_and_tools.py` - Pytest tests for GigaChat3 inference and tool calling
- `tests/test_gpt_oss_function_calling.py` - Pytest tests for GPT-OSS function calling
- `tests/test_tool_calls.py` - Standalone tool call test script
- `tests/test_gigachat3_standalone.py` - Standalone test script for GigaChat3

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended)
- vLLM >= 0.6.0
- HuggingFace Hub

## Project Structure

```
cmw-vllm/
├── cmw_vllm/              # Main package
│   ├── cli.py             # CLI interface
│   ├── server_manager.py  # Server process management
│   ├── server_config.py  # Configuration
│   ├── health_check.py    # Health checks
│   ├── model_downloader.py # Model downloading
│   ├── model_verifier.py  # Model verification
│   ├── model_registry.py  # Model metadata
│   ├── disk_space.py      # Disk space utilities
│   ├── gpu_info.py        # GPU detection
│   └── logging.py         # Logging setup
├── tests/                 # Tests
├── docs/                  # Documentation
└── config/                # Configuration templates
```
