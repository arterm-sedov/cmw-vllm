#!/usr/bin/env python3
"""Test tool calls with Qwen (hermes) and OpenAI approaches."""
import requests
import json
import sys

def test_tool_calls_qwen_approach(base_url: str = "http://localhost:8000", model: str = None):
    """Test tool calls using Qwen/hermes parser approach."""
    print("=" * 80)
    print("Testing Tool Calls - Qwen/Hermes Approach")
    print("=" * 80)
    
    # Check available models
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
        models_data = models_response.json()
        available_models = [m.get("id") for m in models_data.get("data", [])]
        print(f"\nAvailable models: {', '.join(available_models)}")
        
        if not model and available_models:
            # Try to find a Qwen model
            qwen_models = [m for m in available_models if 'qwen' in m.lower()]
            if qwen_models:
                model = qwen_models[0]
                print(f"Using Qwen model: {model}")
            else:
                model = available_models[0]
                print(f"Using available model: {model}")
    except Exception as e:
        print(f"Error checking models: {e}")
        return False
    
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
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print(f"\nRequest with model: {model}")
    print(f"Parser: hermes (Qwen approach)")
    
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        message = result['choices'][0]['message']
        tool_calls = message.get('tool_calls', [])
        
        if tool_calls:
            print("\n✓ Tool calls detected!")
            for i, tc in enumerate(tool_calls, 1):
                print(f"\n  Tool Call {i}:")
                print(f"    ID: {tc.get('id')}")
                print(f"    Function: {tc['function']['name']}")
                print(f"    Arguments: {tc['function']['arguments']}")
            return True
        else:
            print("\n⚠ No tool calls detected")
            print(f"  Content: {message.get('content', 'None')[:200]}")
            return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

def test_tool_calls_openai_approach(base_url: str = "http://localhost:8000", model: str = None):
    """Test tool calls using OpenAI parser approach."""
    print("\n" + "=" * 80)
    print("Testing Tool Calls - OpenAI Approach")
    print("=" * 80)
    
    # Check available models
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
        models_data = models_response.json()
        available_models = [m.get("id") for m in models_data.get("data", [])]
        print(f"\nAvailable models: {', '.join(available_models)}")
        
        if not model and available_models:
            # Try to find a GPT-OSS model
            gpt_models = [m for m in available_models if 'gpt' in m.lower() or 'oss' in m.lower()]
            if gpt_models:
                model = gpt_models[0]
                print(f"Using GPT model: {model}")
            else:
                model = available_models[0]
                print(f"Using available model: {model}")
    except Exception as e:
        print(f"Error checking models: {e}")
        return False
    
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
                            "description": "The mathematical expression to evaluate"
                        }
                    },
                    "required": ["expression"]
                }
            }
        }
    ]
    
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Calculate 15 multiplied by 23"}
        ],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print(f"\nRequest with model: {model}")
    print(f"Parser: openai (OpenAI approach)")
    
    try:
        response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        message = result['choices'][0]['message']
        tool_calls = message.get('tool_calls', [])
        
        if tool_calls:
            print("\n✓ Tool calls detected!")
            for i, tc in enumerate(tool_calls, 1):
                print(f"\n  Tool Call {i}:")
                print(f"    ID: {tc.get('id')}")
                print(f"    Function: {tc['function']['name']}")
                print(f"    Arguments: {tc['function']['arguments']}")
            return True
        else:
            print("\n⚠ No tool calls detected")
            print(f"  Content: {message.get('content', 'None')[:200]}")
            return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("\n" + "=" * 80)
    print("Tool Call Testing - Qwen vs OpenAI Approaches")
    print("=" * 80)
    print("\nNote: GigaChat model outputs tool calls in text format and may not")
    print("be fully compatible with vLLM tool call parsers.")
    print("\nFor proper tool call testing, use:")
    print("  - Qwen models with --tool-call-parser hermes")
    print("  - GPT-OSS models with --tool-call-parser openai")
    print("=" * 80)
    
    # Test both approaches
    qwen_result = test_tool_calls_qwen_approach(base_url)
    openai_result = test_tool_calls_openai_approach(base_url)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Qwen/Hermes approach: {'✓ PASSED' if qwen_result else '✗ FAILED'}")
    print(f"OpenAI approach: {'✓ PASSED' if openai_result else '✗ FAILED'}")
    print("=" * 80)
