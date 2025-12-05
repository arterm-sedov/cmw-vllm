"""Pytest tests for GPT-OSS function calling support."""
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
    return os.getenv("VLLM_MODEL", "openai/gpt-oss-20b")


def test_server_health(base_url: str) -> None:
    """Test that the server is running and healthy."""
    response = requests.get(f"{base_url}/health", timeout=5)
    assert response.status_code == 200, f"Server health check failed: {response.status_code}"


def test_model_available(base_url: str, model: str) -> None:
    """Test that the specified model is available."""
    response = requests.get(f"{base_url}/v1/models", timeout=5)
    assert response.status_code == 200
    
    models_data = response.json()
    available_models = [m.get("id") for m in models_data.get("data", [])]
    assert model in available_models, f"Model {model} not found. Available: {available_models}"


def test_function_calling_weather_tool(base_url: str, model: str) -> None:
    """Test function calling with a weather tool."""
    tools = [
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
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?",
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0.1,
    }
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    result = response.json()
    
    # Validate response structure
    assert "choices" in result, "Response missing 'choices'"
    assert len(result["choices"]) > 0, "No choices in response"
    
    message = result["choices"][0].get("message", {})
    
    # For gpt-oss models, tool_calls should be present
    tool_calls = message.get("tool_calls")
    assert tool_calls is not None, "tool_calls is missing from response"
    assert len(tool_calls) > 0, "tool_calls array is empty"
    
    # Validate tool call structure
    tool_call = tool_calls[0]
    assert "id" in tool_call, "Tool call missing 'id'"
    assert "type" in tool_call, "Tool call missing 'type'"
    assert tool_call["type"] == "function", f"Expected type 'function', got '{tool_call['type']}'"
    
    # Validate function details
    function = tool_call.get("function", {})
    assert "name" in function, "Function missing 'name'"
    assert function["name"] == "get_weather", f"Expected function 'get_weather', got '{function['name']}'"
    assert "arguments" in function, "Function missing 'arguments'"
    
    # Validate arguments are valid JSON
    arguments_str = function["arguments"]
    try:
        arguments = json.loads(arguments_str)
        assert "location" in arguments, "Arguments missing 'location'"
        assert arguments["location"] == "San Francisco", f"Expected location 'San Francisco', got '{arguments['location']}'"
    except json.JSONDecodeError as e:
        pytest.fail(f"Arguments are not valid JSON: {arguments_str}, error: {e}")
    
    # Validate finish reason
    assert result["choices"][0].get("finish_reason") == "tool_calls", (
        f"Expected finish_reason 'tool_calls', got '{result['choices'][0].get('finish_reason')}'"
    )
    
    # Content should be null for tool calls
    assert message.get("content") is None, "Content should be null when tool_calls are present"


def test_function_calling_calculator_tool(base_url: str, model: str) -> None:
    """Test function calling with a calculator tool."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Perform basic arithmetic calculations",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }
    ]
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Calculate 15 multiplied by 23",
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0.1,
    }
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    result = response.json()
    
    # Validate response structure
    assert "choices" in result, "Response missing 'choices'"
    assert len(result["choices"]) > 0, "No choices in response"
    
    message = result["choices"][0].get("message", {})
    tool_calls = message.get("tool_calls")
    
    assert tool_calls is not None, "tool_calls is missing from response"
    assert len(tool_calls) > 0, "tool_calls array is empty"
    
    # Validate tool call
    tool_call = tool_calls[0]
    function = tool_call.get("function", {})
    assert function["name"] == "calculator", f"Expected function 'calculator', got '{function['name']}'"
    
    # Validate arguments contain expression
    arguments_str = function["arguments"]
    arguments = json.loads(arguments_str)
    assert "expression" in arguments, "Arguments missing 'expression'"
    # Expression should contain the calculation
    assert "15" in arguments["expression"] and "23" in arguments["expression"], (
        f"Expression should contain '15' and '23', got '{arguments['expression']}'"
    )


def test_function_calling_required_tool_choice(base_url: str, model: str) -> None:
    """Test function calling with tool_choice='required'."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First number"},
                        "b": {"type": "number", "description": "Second number"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What is 25 + 17?",
            }
        ],
        "tools": tools,
        "tool_choice": "required",  # Force tool use
        "max_tokens": 200,
        "temperature": 0.1,
    }
    
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=120,
    )
    assert response.status_code == 200, f"Request failed: {response.status_code}"
    
    result = response.json()
    message = result["choices"][0].get("message", {})
    tool_calls = message.get("tool_calls")
    
    # With tool_choice='required', tool_calls must be present
    assert tool_calls is not None, "tool_calls is missing when tool_choice='required'"
    assert len(tool_calls) > 0, "tool_calls array is empty when tool_choice='required'"
    
    # Validate arguments
    tool_call = tool_calls[0]
    function = tool_call.get("function", {})
    arguments = json.loads(function["arguments"])
    assert "a" in arguments and "b" in arguments, "Arguments missing 'a' or 'b'"
    assert arguments["a"] == 25, f"Expected a=25, got a={arguments['a']}"
    assert arguments["b"] == 17, f"Expected b=17, got b={arguments['b']}"
