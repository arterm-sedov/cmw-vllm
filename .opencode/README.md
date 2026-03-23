# CMW-vLLM Project Index

**Generated:** 2026-03-22

All documentation, code, tests, and configuration files in cmw-vllm.

---

## Core Code (`cmw_vllm/`)

| File | Description |
|------|-------------|
| <../cmw_vllm/__init__.py> | Package exports |
| <../cmw_vllm/cli.py> | CLI interface |
| <../cmw_vllm/server_config.py> | Server configuration |
| <../cmw_vllm/server_manager.py> | Server process management |
| <../cmw_vllm/vllm_wrapper.py> | vLLM wrapper |
| <../cmw_vllm/model_registry.py> | Model registry |
| <../cmw_vllm/model_config_patcher.py> | Model config patching |
| <../cmw_vllm/model_downloader.py> | Model download management |
| <../cmw_vllm/model_verifier.py> | Model verification |
| <../cmw_vllm/health_check.py> | Health check endpoints |
| <../cmw_vllm/gpu_info.py> | GPU information utilities |
| <../cmw_vllm/disk_space.py> | Disk space utilities |
| <../cmw_vllm/logging.py> | Logging configuration |

### tool_parsers/

| File | Description |
|------|-------------|
| <../cmw_vllm/tool_parsers/__init__.py> | Tool parser exports |
| <../cmw_vllm/tool_parsers/gigachat3_tool_parser.py> | GigaChat3 tool call parser |
| <../cmw_vllm/tool_parsers/plugin.py> | Tool parser plugin interface |

## Tests (`tests/`)

| File | Description |
|------|-------------|
| <../tests/test_embedding_reranker.py> | Embedding and reranker tests |
| <../tests/test_embedding_reranker_models.py> | Embedding/reranker model tests |
| <../tests/test_gigachat3_inference_and_tools.py> | GigaChat3 inference and tool calling |
| <../tests/test_gigachat3_standalone.py> | GigaChat3 standalone tests |
| <../tests/test_gpt_oss_function_calling.py> | GPT-OSS function calling tests |
| <../tests/test_tool_calls.py> | Tool calling tests |

### Root-Level Test Scripts

| File | Description |
|------|-------------|
| <../test_vllm_inference.py> | vLLM inference test |
| <../test_tool_calls.py> | Tool calling test script |
| <../test_tool_calls_comparison.py> | Tool calls comparison |
| <../test_gigachat3_streaming.py> | GigaChat3 streaming test |
| <../test_gigachat3_streaming_with_content.py> | GigaChat3 streaming with content |
| <../test_gigachat3_streaming_content_after_tool.py> | GigaChat3 streaming after tool call |
| <../test_qwen3_coder_tool_calls.py> | Qwen3 Coder tool calls test |
| <../test_curl.sh> | cURL-based API test script |

## Configuration

| File | Description |
|------|-------------|
| <../config/server_config.yaml.example> | Example vLLM server configuration |
| <../.env.example> | Environment config (48GB GPU defaults) |
| <../pyproject.toml> | Build config (setuptools + wheel) |

## Documentation (`docs/`)

| File | Description |
|------|-------------|
| <../docs/installation.md> | Installation guide (system requirements) |
| <../docs/usage.md> | Usage guide (basic workflow) |
| <../docs/gigachat3_tool_parser.md> | GigaChat3 tool parser plugin documentation |
| <../docs/lmcache_kv_offloading.md> | LMCache KV cache offloading for vLLM v1 |

## Root Documentation

| File | Description |
|------|-------------|
| <../README.md> | Project documentation: features, setup, CLI reference |
| <../AGENTS.md> | Agent guide: conventions, testing practices |
| <../UPDATE_SUMMARY.md> | Qwen3 Embedding support update (2026-02-20) |

## Examples

| File | Description |
|------|-------------|
| <../examples/README.md> | Examples documentation |
| <../examples/qwen3_embedding_vllm.py> | Qwen3 embedding vLLM examples |
