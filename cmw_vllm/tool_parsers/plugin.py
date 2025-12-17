"""Plugin registration for GigaChat3 tool parser."""

from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParserManager,
)


def register_gigachat3_parser():
    """Register the GigaChat3 tool parser with vLLM."""
    ToolParserManager.register_lazy_module(
        name="gigachat3",
        module_path="cmw_vllm.tool_parsers.gigachat3_tool_parser",
        class_name="GigaChat3ToolParser",
    )


# Auto-register when module is imported
register_gigachat3_parser()
