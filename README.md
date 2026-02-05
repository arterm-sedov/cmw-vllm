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

Configuration is done via environment variables. Create a `.env` file from the example:

```bash
cp .env.example .env
```

**Important:** `.env.example` is the authoritative source of truth for all configuration options. The sections below document only the most commonly used variables. See `.env.example` for complete documentation with inline comments.

### Core Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_MODEL` | `openai/gpt-oss-20b` | Model identifier (HuggingFace model ID) |
| `VLLM_PORT` | `8000` | Server port |
| `VLLM_HOST` | `0.0.0.0` | Server host address |
| `VLLM_MAX_MODEL_LEN` | `40000` | Maximum sequence length |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.8` | GPU memory usage (0.0-1.0) |
| `VLLM_TRUST_REMOTE_CODE` | `false` | Required for some custom tokenizers |

### Pooling Model Variables

For embedding, reranking, and guard models (vLLM 0.15.0+):

| Variable | Options | Description |
|----------|---------|-------------|
| `VLLM_TASK` | `embed`, `score`, `classify` | Pooling task type |
| `VLLM_RUNNER` | `auto`, `generate`, `pooling` | Model runner type (use `pooling` for pooling models) |
| `VLLM_HF_OVERRIDES` | JSON string | HuggingFace config overrides (e.g., for BGE-M3 models) |

### Function Calling Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ENABLE_AUTO_TOOL_CHOICE` | `true` | Enable auto tool choice for function calling |
| `VLLM_TOOL_CALL_PARSER` | `hermes` | Tool call parser (hermes, mistral, qwen3_xml, openai, gigachat3) |
| `VLLM_DTYPE` | `auto` | Model dtype (float16, bfloat16, auto) |

### KV Cache Offloading (vLLM v1+)

| Variable | Description |
|----------|-------------|
| `VLLM_KV_OFFLOADING_BACKEND` | Backend for KV offloading (e.g., `lmcache`) |
| `VLLM_KV_OFFLOADING_SIZE` | Size in GB for KV cache offloading |
| `VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER` | Required for LMCache offloading |

### Additional Variables

Model-specific options (see `.env.example` for full list):
- `VLLM_TOKENIZER_MODE`: Tokenizer mode (mistral for Mistral models)
- `VLLM_CONFIG_FORMAT`: Config format (mistral for Mistral models)
- `VLLM_LOAD_FORMAT`: Load format (mistral for Mistral models)
- `VLLM_SPECULATIVE_CONFIG`: JSON string for speculative decoding (MTP settings)

### Default Hardware Configuration

Optimized for **48GB GPUs** (RTX 4090, A6000, etc.):
- `VLLM_MAX_MODEL_LEN=40000`: Reduced from 262144 to fit KV cache
- `VLLM_GPU_MEMORY_UTILIZATION=0.8`: Headroom for other processes
- `VLLM_KV_OFFLOADING_SIZE=5.0`: KV cache offloading enabled (see LMCache docs)

**Deprecated:** `VLLM_CPU_OFFLOAD_GB` is deprecated in vLLM v1. Use `VLLM_KV_OFFLOADING_*` instead.

See `.env.example` for all available variables with descriptions and recommended values.

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

## Multi-Model Deployment

cmw-vllm supports running multiple pooling models (embedding, reranker, guard) simultaneously using separate vLLM instances.

### Running Models Simultaneously

```bash
# Start embedding server (port 8100)
cmw-vllm start --model Qwen/Qwen3-Embedding-0.6B --port 8100 &

# Start reranker server (port 8101)
cmw-vllm start --model Qwen/Qwen3-Reranker-0.6B --port 8101 &

# Start guard/moderation server (port 8105)
cmw-vllm start --model Qwen/Qwen3Guard-Gen-0.6B --port 8105 &
```

### VRAM Management

**Model sizes and requirements:**

| Model Type | Example | Model Size | KV Cache | Context Window | vLLM Overhead | Total VRAM |
|------------|---------|-----------|----------|----------------|---------------|-----------|
| Embedding | Qwen3-0.6B | 1.2 GB | 1-2 GB | 32K | 1-2 GB | ~4-6 GB |
| Reranker | Qwen3-0.6B | 1.2 GB | 0.5-1 GB | 32K | 1-2 GB | ~3-5 GB |
| Reranker | BGE-M3 | 2.5 GB | 0.5-1 GB | 8K | 1-2 GB | ~4-5 GB |
| Guard | Qwen3Guard | 0.8 GB | 0.5-1 GB | 32K | 1-2 GB | ~3-4 GB |
| Embedding | FRIDA | 1.0 GB | 0.5-1 GB | 512 | 1-2 GB | ~3-4 GB |

**VRAM estimates for common pooling model combinations:**

| Embedder | Reranker | Guard | Total VRAM (approx) |
|----------|----------|-------|-------------------|
| Qwen3-0.6B | Qwen3-0.6B | Qwen3Guard | ~10-15 GB |
| Qwen3-0.6B | BGE-M3 | Qwen3Guard | ~11-16 GB |
| FRIDA | DiTy | Qwen3Guard | ~8-12 GB |

**48GB GPU recommendations:**
- Pooling-only: 3 models easily fit (~10-15 GB)
- LLM + pooling: One LLM + 3 models may fit (see separate VRAM section for LLM combinations)

### Process Management

```bash
# List all running vLLM processes
cmw-vllm list --all

# Test individual server status
cmw-vllm test-embedder --base-url http://localhost:8100
cmw-vllm test-reranker --base-url http://localhost:8101
cmw-vllm test-guard --base-url http://localhost:8105

# Stop specific models (if needed)
pkill -f "Qwen3-Embedding-0.6B"
```

### Limitations

- Each vLLM instance requires a separate port
- No unified port routing without external API gateway
- GPU memory is shared among all instances
- Consider nginx/gateway for single-port API

### Comparison with cmw-mosec

cmw-mosec runs a single Mosec process with dynamic model loading on one port, while cmw-vllm runs separate vLLM instances:
- **cmw-mosec**: Single process, single port, dynamic model swapping
- **cmw-vllm**: Separate processes, separate ports, simpler architecture

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
- `--cpu-offload-gb INT`: CPU offload memory in GB (DEPRECATED, use KV offloading instead)
- `--trust-remote-code`: Trust remote code (required for some custom tokenizers)
- `--tool-call-parser`: Tool call parser (mistral, hermes, qwen3_xml, openai, gigachat3)
- `--tokenizer-mode`: Tokenizer mode (mistral for Mistral models)
- `--config-format`: Config format (mistral for Mistral models)
- `--load-format`: Load format (mistral for Mistral models)
- `--dtype`: Model dtype (auto, float16, bfloat16)
- `--speculative-config`: JSON string for vLLM --speculative-config
- `--task TASK`: Pooling task type (embed, score, classify) for pooling models
- `--runner RUNNER`: Model runner type (auto, generate, pooling)
- `--hf-overrides`: HuggingFace config overrides (e.g., for BGE-M3 models)
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

### `cmw-vllm test-embedder [--base-url URL]`
Quick test for embedding model.

**Options:**
- `--base-url URL`: Embedder server base URL (default: http://localhost:8100)

### `cmw-vllm test-reranker [--base-url URL]`
Quick test for reranker model.

**Options:**
- `--base-url URL`: Reranker server base URL (default: http://localhost:8101)

### `cmw-vllm test-guard [--base-url URL]`
Quick test for guard/moderator model.

**Options:**
- `--base-url URL`: Guard/moderator server base URL (default: http://localhost:8105)

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
- Guard/Moderator API (safety classification)

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
