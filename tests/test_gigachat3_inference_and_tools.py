"""Pytest tests for GigaChat3 inference and tool calling support."""
import json
import os
from typing import Any

import pytest
import requests


@pytest.fixture
def base_url() -> str:
    """Get base URL from environment or use default."""
    return os.getenv("VLLM_BASE_URL", "http://localhost:8000")


@pytest.fixture
def model() -> str:
    """Get model from environment or use default."""
    return os.getenv("VLLM_MODEL", "ai-sage/GigaChat3-10B-A1.8B-bf16")


@pytest.fixture
def weather_tool() -> list[dict[str, Any]]:
    """Fixture for weather tool definition."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature",
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]


@pytest.fixture
def calculator_tool() -> list[dict[str, Any]]:
    """Fixture for calculator tool definition."""
    return [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Perform a mathematical calculation",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    ]


def test_server_status(base_url: str) -> None:
    """Test that the server is running and responding."""
    response = requests.get(f"{base_url}/v1/models", timeout=5)
    assert response.status_code == 200, f"Server status check failed: {response.status_code}"
    
    models_data = response.json()
    available_models = [m.get("id") for m in models_data.get("data", [])]
    assert len(available_models) > 0, "No models available"


def test_model_available(base_url: str, model: str) -> None:
    """Test that the specified model is available."""
    response = requests.get(f"{base_url}/v1/models", timeout=5)
    assert response.status_code == 200
    
    models_data = response.json()
    available_models = [m.get("id") for m in models_data.get("data", [])]
    assert model in available_models, f"Model {model} not found. Available: {available_models}"


def test_simple_inference(base_url: str, model: str) -> None:
    """Test simple inference without tools."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "What is 2+2?"}],
            "max_tokens": 50,
        },
        timeout=30,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    result = response.json()
    assert "choices" in result, "Response missing 'choices'"
    assert len(result["choices"]) > 0, "No choices in response"
    
    message = result["choices"][0].get("message", {})
    content = message.get("content", "")
    assert content is not None and len(content) > 0, "Response content is empty"


def test_tool_calling_weather(base_url: str, model: str, weather_tool: list[dict[str, Any]]) -> None:
    """Test tool calling with weather function."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
            "tools": weather_tool,
            "tool_choice": "auto",
            "max_tokens": 100,
        },
        timeout=30,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    result = response.json()
    message = result["choices"][0].get("message", {})
    tool_calls = message.get("tool_calls", [])
    
    assert len(tool_calls) > 0, "No tool calls in response"
    
    tool_call = tool_calls[0]
    assert "id" in tool_call, "Tool call missing 'id'"
    assert "type" in tool_call, "Tool call missing 'type'"
    assert tool_call["type"] == "function", f"Expected type 'function', got '{tool_call['type']}'"
    
    function = tool_call.get("function", {})
    assert function.get("name") == "get_weather", f"Expected function 'get_weather', got '{function.get('name')}'"
    assert "arguments" in function, "Function missing 'arguments'"
    
    # Validate arguments are valid JSON
    arguments_str = function["arguments"]
    arguments = json.loads(arguments_str)
    assert "location" in arguments, "Arguments missing 'location'"
    
    # Validate finish reason
    assert result["choices"][0].get("finish_reason") == "tool_calls", (
        f"Expected finish_reason 'tool_calls', got '{result['choices'][0].get('finish_reason')}'"
    )


def test_complete_tool_calling_flow(base_url: str, model: str, weather_tool: list[dict[str, Any]]) -> None:
    """Test complete tool calling flow with tool result."""
    # Step 1: Request with tool call
    response1 = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "What is the weather in Paris, France?"}],
            "tools": weather_tool,
            "tool_choice": "auto",
            "max_tokens": 100,
        },
        timeout=30,
    )
    assert response1.status_code == 200, f"Step 1 failed: {response1.status_code}"
    
    data1 = response1.json()
    message1 = data1.get("choices", [{}])[0].get("message", {})
    tool_calls = message1.get("tool_calls", [])
    
    assert len(tool_calls) > 0, "No tool calls in step 1"
    
    # Step 2: Send tool result back
    messages = [
        {"role": "user", "content": "What is the weather in Paris, France?"},
        message1,  # Assistant's message with tool call
        {
            "role": "tool",
            "tool_call_id": tool_calls[0].get("id"),
            "name": tool_calls[0].get("function", {}).get("name"),
            "content": json.dumps({"temperature": 15, "condition": "sunny", "unit": "celsius"}),
        },
    ]
    
    response2 = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "tools": weather_tool,
            "max_tokens": 100,
        },
        timeout=30,
    )
    assert response2.status_code == 200, f"Step 2 failed: {response2.status_code}"
    
    data2 = response2.json()
    message2 = data2.get("choices", [{}])[0].get("message", {})
    content = message2.get("content", "")
    
    assert content is not None and len(content) > 0, "Final response content is empty"
    # Should mention weather information
    assert "weather" in content.lower() or "15" in content or "sunny" in content.lower(), (
        f"Response doesn't seem to contain weather information: {content}"
    )


def test_streaming_inference(base_url: str, model: str) -> None:
    """Test streaming inference."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "max_tokens": 50,
            "stream": True,
        },
        stream=True,
        timeout=30,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    chunks = []
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove 'data: ' prefix
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        chunks.append(content)
                except json.JSONDecodeError:
                    pass
    
    assert len(chunks) > 0, "No content chunks received in stream"


def test_streaming_tool_calls(base_url: str, model: str, calculator_tool: list[dict[str, Any]]) -> None:
    """Test streaming with tool calls."""
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Calculate 25 * 4"}],
            "tools": calculator_tool,
            "tool_choice": "auto",
            "max_tokens": 100,
            "stream": True,
        },
        stream=True,
        timeout=30,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    tool_call_data = None
    for line in response.iter_lines():
        if line:
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choice = data.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    
                    # Check for tool calls in delta
                    if "tool_calls" in delta:
                        if tool_call_data is None:
                            tool_call_data = {"tool_calls": []}
                        for tc_delta in delta["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            while len(tool_call_data["tool_calls"]) <= idx:
                                tool_call_data["tool_calls"].append(
                                    {
                                        "index": idx,
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )
                            
                            if "id" in tc_delta:
                                tool_call_data["tool_calls"][idx]["id"] = tc_delta["id"]
                            if "function" in tc_delta:
                                func_delta = tc_delta["function"]
                                if "name" in func_delta:
                                    tool_call_data["tool_calls"][idx]["function"]["name"] = func_delta["name"]
                                if "arguments" in func_delta:
                                    tool_call_data["tool_calls"][idx]["function"]["arguments"] += func_delta[
                                        "arguments"
                                    ]
                except json.JSONDecodeError:
                    pass
    
    assert tool_call_data is not None, "No tool call data in stream"
    assert len(tool_call_data["tool_calls"]) > 0, "No tool calls in stream"
    
    tool_call = tool_call_data["tool_calls"][0]
    assert tool_call["function"]["name"] == "calculate", (
        f"Expected function 'calculate', got '{tool_call['function']['name']}'"
    )
    assert "25" in tool_call["function"]["arguments"] and "4" in tool_call["function"]["arguments"], (
        f"Arguments should contain '25' and '4', got '{tool_call['function']['arguments']}'"
    )
