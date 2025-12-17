# GigaChat3 Tool Parser

Custom tool parser plugin for `ai-sage/GigaChat3-10B-A1.8B-bf16` and other GigaChat3 models.

## Overview

GigaChat3 models use a unique tool call format: `function call{...}` instead of the standard JSON format used by other models. This custom parser enables vLLM to properly parse and handle tool calls from GigaChat3 models.

## Format

GigaChat3 models output tool calls in the following format:

```
function call{"name": "tool_name", "arguments": {...}}
```

or with a role separator:

```
function call<|role_sep|>
{"name": "tool_name", "arguments": {...}}
```

## Configuration

### Model Registry

The GigaChat3 model is configured in `cmw_vllm/model_registry.py`:

```python
"ai-sage/GigaChat3-10B-A1.8B-bf16": {
    "tool_call_parser": "gigachat3",
    "gpu_memory_utilization": 0.6,
    "dtype": "auto",
    "speculative_config": '{"method": "mtp", "num_speculative_tokens": 1}',
}
```

### Environment Variables

Set in `.env`:

```bash
VLLM_MODEL=ai-sage/GigaChat3-10B-A1.8B-bf16
VLLM_TOOL_CALL_PARSER=gigachat3
VLLM_GPU_MEMORY_UTILIZATION=0.6
VLLM_CPU_OFFLOAD_GB=0  # Disabled (incompatible with vLLM v1 engine)
VLLM_DTYPE=auto
VLLM_SPECULATIVE_CONFIG={"method": "mtp", "num_speculative_tokens": 1, "disable_padded_drafter_batch": false}
```

## Implementation

### Parser Registration

The parser is registered via `vllm_wrapper.py` which runs before vLLM validates parser names:

```python
ToolParserManager.register_lazy_module(
    name="gigachat3",
    module_path="cmw_vllm.tool_parsers.gigachat3_tool_parser",
    class_name="GigaChat3ToolParser",
)
```

### Parser Location

- **Parser class**: `cmw_vllm/tool_parsers/gigachat3_tool_parser.py`
- **Registration**: `cmw_vllm/vllm_wrapper.py`
- **Based on**: vLLM v12 reference implementation

## Features

- ✅ Non-streaming tool call extraction
- ✅ Streaming tool call extraction
- ✅ Proper JSON argument parsing
- ✅ Content extraction (text before tool calls)
- ✅ Multiple tool call support

## Usage

Start the server with GigaChat3 configuration:

```bash
python -m cmw_vllm.cli start
```

The server will automatically use the `gigachat3` parser for GigaChat3 models based on the model registry configuration.

## Testing

Test tool calling with:

```bash
python test_tool_calls.py http://localhost:8000 ai-sage/GigaChat3-10B-A1.8B-bf16
```

## Compatibility

- **vLLM version**: Requires vLLM from GitHub main branch (v12+)
- **Models**: `ai-sage/GigaChat3-10B-A1.8B-bf16` and compatible GigaChat3 variants
- **Other models**: This parser does not interfere with other models. Each model uses its configured parser (hermes, openai, qwen3_xml, etc.)

## References

- [vLLM Tool Calling Documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)
- [vLLM Tool Parser Plugin Guide](https://docs.vllm.ai/en/latest/features/tool_calling/#how-to-write-a-tool-parser-plugin)
- [GigaChat3 Model Card](https://huggingface.co/ai-sage/GigaChat3-10B-A1.8B-bf16)
