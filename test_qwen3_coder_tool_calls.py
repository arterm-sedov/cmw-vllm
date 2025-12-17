#!/usr/bin/env python3
"""Test tool calls with Qwen3-Coder model using qwen3_xml parser."""
import requests
import json
import sys
import time

def test_qwen3_coder_tool_calls(base_url: str = "http://localhost:8000", model: str = "cerebras/Qwen3-Coder-REAP-25B-A3B"):
    """Test tool calls using Qwen3-Coder with qwen3_xml parser."""
    print("=" * 80)
    print("Testing Tool Calls - Qwen3-Coder (qwen3_xml parser)")
    print("=" * 80)
    
    # Wait for server to be ready
    print("\nWaiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            if response.status_code == 200:
                print("✓ Server is ready!")
                break
        except Exception:
            if i < max_retries - 1:
                print(f"  Waiting... ({i+1}/{max_retries})")
                time.sleep(2)
            else:
                print("✗ Server not responding after waiting")
                return False
    
    # Check available models
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=10)
        models_response.raise_for_status()
        models_data = models_response.json()
        available_models = [m.get("id") for m in models_data.get("data", [])]
        print(f"\nAvailable models: {', '.join(available_models)}")
        
        if model not in available_models:
            if available_models:
                model = available_models[0]
                print(f"Using available model: {model}")
            else:
                print("✗ No models available")
                return False
    except Exception as e:
        print(f"✗ Error checking models: {e}")
        return False
    
    # Define a simple tool for testing
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
    print(f"Parser: qwen3_xml (Qwen3-Coder XML format)")
    print(f"Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print("\nSending request...")
        response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        print("\n" + "=" * 80)
        print("Response:")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        
        message = result['choices'][0]['message']
        tool_calls = message.get('tool_calls', [])
        
        if tool_calls:
            print("\n" + "=" * 80)
            print("✓ Tool calls detected!")
            print("=" * 80)
            for i, tc in enumerate(tool_calls, 1):
                print(f"\n  Tool Call {i}:")
                print(f"    ID: {tc.get('id')}")
                print(f"    Type: {tc.get('type')}")
                print(f"    Function: {tc['function']['name']}")
                print(f"    Arguments: {tc['function']['arguments']}")
            return True
        else:
            print("\n" + "=" * 80)
            print("⚠ No tool calls detected")
            print("=" * 80)
            content = message.get('content', 'None')
            print(f"  Content: {content[:500]}")
            if content:
                print(f"\n  Full content length: {len(content)} characters")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response status: {e.response.status_code}")
            print(f"  Response body: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "cerebras/Qwen3-Coder-REAP-25B-A3B"
    
    success = test_qwen3_coder_tool_calls(base_url, model)
    
    print("\n" + "=" * 80)
    print("Test Result")
    print("=" * 80)
    print(f"{'✓ PASSED' if success else '✗ FAILED'}")
    print("=" * 80)
    
    sys.exit(0 if success else 1)
