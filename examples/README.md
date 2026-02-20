# CMW vLLM Examples

This directory contains practical examples for using CMW vLLM with different models.

## Available Examples

### Qwen3 Embedding (`qwen3_embedding_vllm.py`)

Demonstrates proper usage of Qwen3 embedding models with vLLM.

**Key Concepts:**
- Instruction format for queries (required per HF docs)
- Document format (no instruction needed)
- Multilingual support (119+ languages)
- Performance comparison: wrong vs right format

**Prerequisites:**
```bash
# Start vLLM with Qwen3 embedding model
cmw-vllm start --model Qwen/Qwen3-Embedding-0.6B --port 8100
```

**Run:**
```bash
python examples/qwen3_embedding_vllm.py
```

**What You'll Learn:**
- How to format queries with instructions
- Why instruction format matters (~15% accuracy improvement)
- Cross-lingual retrieval capabilities
- vLLM embedding API usage

## Quick Reference

### Qwen3 Instruction Format

```python
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# Usage
task = 'Given a web search query, retrieve relevant passages that answer the query'
query = get_detailed_instruct(task, 'What is Python?')
# Result: 'Instruct: Given a web search query...\nQuery: What is Python?'
```

### vLLM API Call

```bash
curl -X POST http://localhost:8100/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "input": "Instruct: Given a web search query...\nQuery: What is AI?"
  }'
```

## Model Selection Guide

| Use Case | Model | Port | VRAM |
|----------|-------|------|------|
| Multilingual Embeddings | Qwen3-Embedding-0.6B | 8100 | ~2GB |
| Reranking | Qwen3-Reranker-0.6B | 8101 | ~2GB |
| Safety Guard | Qwen3Guard-Gen-0.6B | 8105 | ~2GB |

**Note on T5-based models:** FRIDA and other T5-based embedding models require **CLS pooling** which is not supported by vLLM's pooling runner (uses last-token pooling). Use [cmw-mosec](https://github.com/arterm-sedov/cmw-mosec) for T5-based embeddings instead.

## Common Mistakes

### Missing Instruction Format

❌ **Wrong:**
```python
query = "What is Python?"  # Missing instruction!
```

✅ **Correct:**
```python
query = "Instruct: Given a web search query...\nQuery: What is Python?"
```

### Wrong Server URL

❌ **Wrong:** Using LLM server port for embeddings
```python
base_url = "http://localhost:8000"  # LLM server
```

✅ **Correct:** Using embedding server port
```python
base_url = "http://localhost:8100"  # Embedding server
```

## Troubleshooting

### "Cannot connect to vLLM server"

**Solution:** Start the server first:
```bash
cmw-vllm start --model Qwen/Qwen3-Embedding-0.6B --port 8100
```

### "Low similarity scores"

**Cause:** Missing instruction format  
**Solution:** Use `get_detailed_instruct()` function for queries

### "Model not found"

**Solution:** Check available models:
```bash
cmw-vllm list
```

## References

- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
- [vLLM Pooling Models](https://docs.vllm.ai/en/latest/models/pooling_models.html)
- [CMW vLLM README](../README.md)
