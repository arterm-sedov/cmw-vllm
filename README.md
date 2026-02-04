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
- **Multi-Model Support**: LLM, embedding, and reranking models via pooling runner

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

## Supported Models

cmw-vllm supports various model types with optimized configurations:

### LLM Models (Generation)
- `Qwen/Qwen3-30B-A3B-Instruct-2507` - Qwen3 30B Mixture of Experts
- `openai/gpt-oss-20b` - OpenAI GPT OSS 20B
- `mistralai/Ministral-3-14B-Instruct-2512` - Mistral Ministral-3 14B
- `ai-sage/GigaChat3-10B-A1.8B-bf16` - Russian GigaChat3 MoE
- `cerebras/Qwen3-Coder-REAP-25B-A3B` - Qwen3 Coder code-specialized

### Embedding Models (Pooling)
- `Qwen/Qwen3-Embedding-0.6B` - Qwen3 lightweight embedding (32K context)
- `ai-forever/FRIDA` - Russian text embedding model

### Reranker Models (Pooling)
- `Qwen/Qwen3-Reranker-0.6B` - Qwen3 cross-encoder reranker
- `BAAI/bge-reranker-v2-m3` - Multilingual reranker (100+ languages)
- `DiTy/cross-encoder-russian-msmarco` - Russian reranker for MS MARCO

### Guard/Moderator Models (Pooling)
- `Qwen/Qwen3Guard-Gen-0.6B` - Safety moderation model for 119 languages, classifies outputs as Safe, Unsafe, or Controversial

### Starting Embedding/Reranker/Guard Servers

```bash
# Start embedding server
cmw-vllm start --model Qwen/Qwen3-Embedding-0.6B --port 8100

# Start reranker server
cmw-vllm start --model Qwen/Qwen3-Reranker-0.6B --port 8101
cmw-vllm start --model BAAI/bge-reranker-v2-m3 --port 8102

# Start guard/moderation server
cmw-vllm start --model Qwen/Qwen3Guard-Gen-0.6B --port 8105
```

Embedding, reranker, and guard models use vLLM's pooling runner (`--runner pooling`) with appropriate tasks:
- `--task embed` for embedding models
- `--task score` for reranker models
- `--task classify` for guard/moderator models

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
- `--task TASK`: Pooling task type (embed, score, classify) for embedding/reranker models
- `--runner RUNNER`: Model runner type (auto, generate, pooling)
- `--hf-overrides JSON`: HuggingFace config overrides (e.g. for BGE-M3 models)
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

The repository includes comprehensive tests for inference, tool calling, and pooling models.

### Standalone Test Scripts

Run standalone test scripts for quick manual testing:

```bash
# Test LLM inference and tool calling
python tests/test_gigachat3_standalone.py http://localhost:8000

# Test embedding and reranking models
python tests/test_embedding_reranker.py http://localhost:8100
```

The LLM standalone script tests:
- Simple inference
- Tool calling
- Complete tool calling flow (with tool results)
- Streaming inference
- Streaming with tool calls

The embedding/reranking script tests:
- Embedding API (OpenAI-compatible)
- Reranker API (score endpoint)
- Pooling API (generic pooling interface)

### Pytest Tests

Run pytest tests (requires `pytest` to be installed):

```bash
# Install pytest
pip install pytest

# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_embedding_reranker_models.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cmw_vllm --cov-report=term-missing
```

Test files:
- `tests/test_gigachat3_inference_and_tools.py` - Pytest tests for GigaChat3 inference and tool calling
- `tests/test_gpt_oss_function_calling.py` - Pytest tests for GPT-OSS function calling
- `tests/test_embedding_reranker_models.py` - Pytest tests for embedding and reranking models
- `tests/test_tool_calls.py` - Standalone tool call test script
- `tests/test_gigachat3_standalone.py` - Standalone test script for GigaChat3
- `tests/test_embedding_reranker.py` - Standalone test script for embedding and reranking models

## KV Cache Offloading (vLLM v1+)

For vLLM v1 (0.12.0+), use **LMCache KV cache offloading** instead of the deprecated `--cpu-offload-gb`:

```bash
# In .env
VLLM_KV_OFFLOADING_BACKEND=lmcache
VLLM_KV_OFFLOADING_SIZE=5.0  # GB
VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER=true
```

This offloads KV cache (not model weights) to CPU/disk, which is more efficient for long-context scenarios and multi-round conversations.

See [docs/lmcache_kv_offloading.md](docs/lmcache_kv_offloading.md) for detailed documentation.

## Requirements

- Python >= 3.10
- CUDA-capable GPU (recommended)
- vLLM >= 0.15.0 (v1 engine with pooling support)
- HuggingFace Hub
- lmcache (optional, for KV cache offloading)

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
