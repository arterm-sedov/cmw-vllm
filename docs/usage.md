# Usage Guide

## Basic Workflow

### 1. Download Model

Download the Qwen3-30B-A3B-Instruct-2507 model:

```bash
cmw-vllm download
```

This will:
- Download model from HuggingFace
- Check disk space
- Verify model integrity
- Save to HuggingFace cache (or specified directory)

### 2. Start Server

Start the vLLM server:

```bash
cmw-vllm start
```

Server will start in background and be available at `http://localhost:8000`.

### 3. Check Status

Verify server is running:

```bash
cmw-vllm status
```

### 4. Test Inference

Test with a simple request:

```bash
cmw-vllm status --test-inference
```

### 5. Stop Server

Stop the server when done:

```bash
cmw-vllm stop
```

## Advanced Usage

### Custom Model

Use a different model:

```bash
cmw-vllm download meta-llama/Llama-3.1-70B-Instruct
cmw-vllm start --model meta-llama/Llama-3.1-70B-Instruct
```

### Custom Port

Run server on different port:

```bash
cmw-vllm start --port 9000
```

### Foreground Mode

Run server in foreground (useful for debugging):

```bash
cmw-vllm start --foreground
```

### Custom Download Location

Download model to specific directory:

```bash
cmw-vllm download --local-dir /path/to/models
```

### GPU Memory Configuration

Adjust GPU memory utilization:

```bash
cmw-vllm start --gpu-memory-utilization 0.8
```

## Environment Configuration

Create `.env` file for persistent configuration:

```bash
VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
VLLM_PORT=8000
VLLM_HOST=0.0.0.0
VLLM_MAX_MODEL_LEN=262144
VLLM_GPU_MEMORY_UTILIZATION=0.9
```

## Integration Examples

### With cmw-rag

Configure `cmw-rag` to use vLLM:

```bash
# In cmw-rag/.env
DEFAULT_LLM_PROVIDER=vllm
DEFAULT_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
```

### Direct API Usage

Use vLLM server directly via OpenAI-compatible API:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "messages": [{"role": "user", "content": "Hello!"}],
    },
)
```
