#!/usr/bin/env python3
"""Wrapper for vLLM API server that registers GigaChat3 tool parser plugin.

This wrapper is needed because vLLM validates parser names during argument parsing,
which happens before the server starts. We must register the custom parser BEFORE
that validation occurs.

According to vLLM docs: https://docs.vllm.ai/en/latest/features/tool_calling/#how-to-write-a-tool-parser-plugin
Custom parsers can be registered using ToolParserManager.register_lazy_module(),
but this must happen before vLLM's argument validation.
"""

import sys
from pathlib import Path

# Remove script's directory from sys.path[0] to prevent cmw_vllm.logging
# from shadowing stdlib logging when Python adds script dir to path
script_dir = Path(__file__).parent
if sys.path[0] == str(script_dir):
    sys.path.pop(0)

# Register GigaChat3 parser plugin BEFORE importing vLLM
# This must happen before vLLM validates parser names during argument parsing
# PYTHONPATH should be set by server_manager.py to include the project root

try:
    from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import ToolParserManager
    
    # Register our plugin using lazy loading
    # The module won't be imported until vLLM actually needs the parser
    ToolParserManager.register_lazy_module(
        name="gigachat3",
        module_path="cmw_vllm.tool_parsers.gigachat3_tool_parser",
        class_name="GigaChat3ToolParser",
    )
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register GigaChat3 parser plugin: {e}")

# Now run vLLM's API server
# Use runpy to execute the module as if run with -m
if __name__ == "__main__":
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
