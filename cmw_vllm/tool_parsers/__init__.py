"""GigaChat3 Tool Parser Plugin for vLLM."""

# Don't import here - let vLLM lazy-load it when needed
# This prevents import errors when the module is imported before vLLM is fully initialized
__all__ = ["GigaChat3ToolParser"]
