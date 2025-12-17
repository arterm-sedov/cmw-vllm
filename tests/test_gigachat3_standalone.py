#!/usr/bin/env python3
"""Standalone test script for GigaChat3 inference and tool calling.

This script can be run directly to test inference and tool calling functionality
without requiring pytest. Useful for quick manual testing.

Usage:
    python tests/test_gigachat3_standalone.py [base_url] [model]
    
Examples:
    python tests/test_gigachat3_standalone.py
    python tests/test_gigachat3_standalone.py http://localhost:8000
    python tests/test_gigachat3_standalone.py http://localhost:8000 ai-sage/GigaChat3-10B-A1.8B-bf16
"""
import json
import sys
from typing import Any

import requests


def test_simple_inference(base_url: str, model: str) -> bool:
    """Test simple inference without tools."""
    print("\n=== Test 1: Simple Inference ===")
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "max_tokens": 50,
            },
            timeout=30,
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"Response: {content}")
            print("✓ Simple inference test passed")
            return True
        else:
            print(f"✗ Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def test_tool_calling(base_url: str, model: str) -> bool:
    """Test tool calling with a simple function."""
    print("\n=== Test 2: Tool Calling ===")
    
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
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is the weather in San Francisco?"}],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 100,
            },
            timeout=30,
        )
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            message = data.get("choices", [{}])[0].get("message", {})
            print(f"Role: {message.get('role')}")
            print(f"Content: {message.get('content')}")
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                print(f"Tool calls: {len(tool_calls)}")
                for i, tool_call in enumerate(tool_calls):
                    print(f"  Tool call {i+1}:")
                    print(f"    ID: {tool_call.get('id')}")
                    print(f"    Function: {tool_call.get('function', {}).get('name')}")
                    print(f"    Arguments: {tool_call.get('function', {}).get('arguments')}")
                print("✓ Tool calling test passed")
                return True
            else:
                print("✗ No tool calls in response")
                return False
        else:
            print(f"✗ Error: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def test_complete_tool_flow(base_url: str, model: str) -> bool:
    """Test complete tool calling flow with tool result."""
    print("\n=== Test 3: Complete Tool Calling Flow ===")
    
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
    
    try:
        # Step 1: Request with tool call
        print("Step 1: Request with tool call")
        response1 = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "What is the weather in Paris, France?"}],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 100,
            },
            timeout=30,
        )
        
        if response1.status_code != 200:
            print(f"✗ Step 1 failed: {response1.status_code} - {response1.text}")
            return False
        
        data1 = response1.json()
        message1 = data1.get("choices", [{}])[0].get("message", {})
        tool_calls = message1.get("tool_calls", [])
        
        if not tool_calls:
            print("✗ No tool calls in step 1")
            return False
        
        print(f"✓ Tool call received: {tool_calls[0].get('function', {}).get('name')}")
        print(f"  Arguments: {tool_calls[0].get('function', {}).get('arguments')}")
        
        # Step 2: Send tool result back
        print("\nStep 2: Sending tool result")
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
                "tools": tools,
                "max_tokens": 100,
            },
            timeout=30,
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            message2 = data2.get("choices", [{}])[0].get("message", {})
            content = message2.get("content", "")
            print(f"✓ Final response received:")
            print(f"  {content}")
            print("✓ Complete tool calling flow test passed")
            return True
        else:
            print(f"✗ Step 2 failed: {response2.status_code} - {response2.text}")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_streaming_inference(base_url: str, model: str) -> bool:
    """Test streaming inference."""
    print("\n=== Test 4: Streaming Inference ===")
    try:
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
        
        if response.status_code != 200:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return False
        
        print("✓ Streaming response:")
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
                            print(content, end="", flush=True)
                            chunks.append(content)
                    except json.JSONDecodeError:
                        pass
        
        print("\n✓ Streaming completed successfully")
        return len(chunks) > 0
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def test_streaming_tool_calls(base_url: str, model: str) -> bool:
    """Test streaming with tool calls."""
    print("\n=== Test 5: Streaming with Tool Calls ===")
    
    tools = [
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
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Calculate 25 * 4"}],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 100,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )
        
        if response.status_code != 200:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return False
        
        print("✓ Streaming tool call response:")
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
                        
                        content = delta.get("content", "")
                        if content:
                            print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        if tool_call_data and tool_call_data["tool_calls"]:
            print("\n✓ Tool call detected in stream:")
            for tc in tool_call_data["tool_calls"]:
                print(f"  Function: {tc['function']['name']}")
                print(f"  Arguments: {tc['function']['arguments']}")
            print("✓ Streaming tool call completed")
            return True
        else:
            print("\n✗ No tool calls detected in stream")
            return False
    except Exception as e:
        print(f"✗ Exception: {e}")
        return False


def main() -> None:
    """Run all tests."""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "ai-sage/GigaChat3-10B-A1.8B-bf16"
    
    print("=" * 80)
    print("GigaChat3 Inference and Tool Calling Tests")
    print("=" * 80)
    print(f"\nServer URL: {base_url}")
    print(f"Model: {model}\n")
    
    # Check server status first
    try:
        response = requests.get(f"{base_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            available_models = [m.get("id") for m in models_data.get("data", [])]
            print(f"✓ Server is running")
            print(f"  Available models: {', '.join(available_models)}")
            if model not in available_models:
                print(f"⚠ Warning: Model {model} not found in available models")
        else:
            print(f"⚠ Server returned status {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {base_url}")
        print("  Make sure the server is running")
        sys.exit(1)
    except Exception as e:
        print(f"⚠ Error checking server status: {e}")
    
    # Run tests
    results = []
    results.append(("Simple Inference", test_simple_inference(base_url, model)))
    results.append(("Tool Calling", test_tool_calling(base_url, model)))
    results.append(("Complete Tool Flow", test_complete_tool_flow(base_url, model)))
    results.append(("Streaming Inference", test_streaming_inference(base_url, model)))
    results.append(("Streaming Tool Calls", test_streaming_tool_calls(base_url, model)))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
