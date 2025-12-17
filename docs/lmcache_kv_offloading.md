# LMCache KV Cache Offloading for vLLM v1

## Overview

In vLLM v1 engine (versions 0.12.0+), the original `--cpu-offload-gb` parameter for model weights has been **deprecated** in favor of a more flexible and high-performance approach using **LMCache for KV cache offloading**.

### Key Differences

| Feature | v0.12.0 (Old) | v1 (New) |
|---------|---------------|----------|
| **What's offloaded** | Model weights | KV cache (attention key/value data) |
| **Parameter** | `--cpu-offload-gb` | `--kv-offloading-backend` + `--kv-offloading-size` |
| **Use case** | Large models on limited GPU | Long-context scenarios, multi-round conversations |
| **Performance** | Slower inference (CPU-GPU transfers) | Better for long contexts, cache reuse |

## Benefits of LMCache KV Cache Offloading

- **Efficient for long contexts**: Offloads KV cache to CPU/disk, freeing GPU memory
- **Multi-round conversations**: Cache reuse reduces time-to-first-token (TTFT)
- **Better GPU utilization**: Model weights stay on GPU, only KV cache is offloaded
- **Scalable**: Can handle very long contexts without running out of GPU memory

## Installation

```bash
pip install lmcache vllm
```

**Note**: This works on Linux NVIDIA GPU platforms.

## Configuration

### Method 1: Environment Variables

Add to your `.env` file:

```bash
# LMCache KV cache offloading
VLLM_KV_OFFLOADING_BACKEND=lmcache
VLLM_KV_OFFLOADING_SIZE=5.0  # Size in GB for CPU cache
VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER=true  # Required for LMCache
```

### Method 2: Model Registry

Add to model configuration in `cmw_vllm/model_registry.py`:

```python
"ai-sage/GigaChat3-10B-A1.8B-bf16": {
    # ... other config ...
    "kv_offloading_backend": "lmcache",
    "kv_offloading_size": 5.0,  # GB
    "disable_hybrid_kv_cache_manager": True,
}
```

### Method 3: CLI Override

```bash
cmw-vllm start \
  --kv-offloading-backend lmcache \
  --kv-offloading-size 5.0 \
  --disable-hybrid-kv-cache-manager
```

## Usage Examples

### Example 1: Long Context Processing

For models with large context windows (e.g., 128k+ tokens), offload KV cache to handle more concurrent requests:

```bash
# .env
VLLM_MODEL=ai-sage/GigaChat3-10B-A1.8B-bf16
VLLM_MAX_MODEL_LEN=128000
VLLM_KV_OFFLOADING_BACKEND=lmcache
VLLM_KV_OFFLOADING_SIZE=10.0  # 10GB CPU cache
VLLM_DISABLE_HYBRID_KV_CACHE_MANAGER=true
```

### Example 2: Multi-Round Conversations

LMCache is particularly effective for multi-turn conversations where KV cache can be reused:

```bash
# Enables cache reuse across conversation turns
VLLM_KV_OFFLOADING_BACKEND=lmcache
VLLM_KV_OFFLOADING_SIZE=5.0
```

## Advanced Configuration

### Environment Variables for LMCache

You can also set LMCache-specific environment variables:

```bash
# Use CPU for local cache
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=5.0  # GB

# Or use disk for larger cache
export LMCACHE_LOCAL_DISK=True
export LMCACHE_MAX_LOCAL_DISK_SIZE=50.0  # GB
```

## Monitoring

To verify LMCache is working:

1. Check vLLM logs for LMCache initialization messages
2. Monitor CPU memory usage (should increase when KV cache is offloaded)
3. Observe improved handling of long contexts

## Migration from cpu_offload_gb

If you were using `--cpu-offload-gb` in older vLLM versions:

1. **Remove** `cpu_offload_gb` from your configuration
2. **Add** `kv_offloading_backend=lmcache` and `kv_offloading_size`
3. **Set** `disable_hybrid_kv_cache_manager=true`

**Note**: `cpu_offload_gb` is still supported for backward compatibility but is deprecated and may be removed in future versions.

## Limitations

- Requires `lmcache` package to be installed
- `--disable-hybrid-kv-cache-manager` flag is currently mandatory
- Works best on Linux with NVIDIA GPUs
- CPU memory usage will increase (KV cache stored in RAM/disk)

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [LMCache GitHub](https://github.com/lm-sys/lmcache)
