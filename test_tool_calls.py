#!/usr/bin/env python3
"""Test script to verify tool call support in vLLM server."""
import requests
import json
import sys
import time

def test_tool_calls(base_url: str = "http://localhost:8000", model: str = "mistralai/Ministral-3-14B-Instruct-2512"):
    """Test tool call support with a function calling request."""
    
    print("=" * 80)
    print("Testing Tool Call Support")
    print("=" * 80)
    
    # First check if server is running
    print("\n1. Checking server status...")
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
        if models_response.status_code == 200:
            print("✓ Server is responding")
            models_data = models_response.json()
            available_models = [m.get("id") for m in models_data.get("data", [])]
            if available_models:
                print(f"  Available models: {', '.join(available_models)}")
                # Use the first available model if model not specified or not found
                if model not in available_models and available_models:
                    print(f"  ⚠ Model '{model}' not found, using '{available_models[0]}' instead")
                    model = available_models[0]
        else:
            print(f"⚠ Server returned status {models_response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {base_url}")
        print("  Make sure the server is running")
        return False
    except Exception as e:
        print(f"✗ Error checking server status: {e}")
        return False
    
    # Define a simple tool/function for testing
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
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The unit of temperature"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]
    
    # Test 1: Request with tools and ask for tool use
    print("\n2. Testing tool call request...")
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather like in San Francisco?"
            }
        ],
        "tools": tools,
        "tool_choice": "auto",  # Let the model decide whether to use tools
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Request payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        print("\n  Sending request...")
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        print("\n✓ Request successful!")
        print("\n  Response:")
        print(json.dumps(result, indent=2))
        
        # Check if tool calls are present
        choices = result.get('choices', [])
        if choices:
            message = choices[0].get('message', {})
            tool_calls = message.get('tool_calls')
            
            if tool_calls:
                print("\n" + "=" * 80)
                print("✓ TOOL CALLS SUPPORTED!")
                print("=" * 80)
                print(f"\n  Found {len(tool_calls)} tool call(s):")
                
                # Validate tool call structure
                validation_errors = []
                for i, tool_call in enumerate(tool_calls, 1):
                    print(f"\n  Tool Call {i}:")
                    print(f"    ID: {tool_call.get('id', 'N/A')}")
                    print(f"    Type: {tool_call.get('type', 'N/A')}")
                    function = tool_call.get('function', {})
                    print(f"    Function: {function.get('name', 'N/A')}")
                    print(f"    Arguments: {function.get('arguments', 'N/A')}")
                    
                    # Validate structure
                    if not tool_call.get('id'):
                        validation_errors.append(f"Tool call {i} missing 'id'")
                    if tool_call.get('type') != 'function':
                        validation_errors.append(f"Tool call {i} type should be 'function', got '{tool_call.get('type')}'")
                    if not function.get('name'):
                        validation_errors.append(f"Tool call {i} function missing 'name'")
                    if not function.get('arguments'):
                        validation_errors.append(f"Tool call {i} function missing 'arguments'")
                
                # Try to parse the arguments
                for tool_call in tool_calls:
                    function = tool_call.get('function', {})
                    args_str = function.get('arguments', '')
                    if args_str:
                        try:
                            args = json.loads(args_str)
                            print(f"\n  Parsed arguments: {json.dumps(args, indent=4)}")
                        except json.JSONDecodeError as e:
                            validation_errors.append(f"Arguments are not valid JSON: {args_str}")
                            print(f"  (Could not parse arguments as JSON: {e})")
                
                # Validate finish reason
                finish_reason = choices[0].get('finish_reason')
                if finish_reason != 'tool_calls':
                    validation_errors.append(f"Expected finish_reason 'tool_calls', got '{finish_reason}'")
                else:
                    print(f"\n  Finish reason: {finish_reason} ✓")
                
                # Validate content is null for tool calls
                content = message.get('content')
                if content is not None:
                    validation_errors.append(f"Content should be null when tool_calls are present, got: {content}")
                else:
                    print(f"  Content: null ✓ (correct for tool calls)")
                
                if validation_errors:
                    print("\n" + "=" * 80)
                    print("⚠ VALIDATION WARNINGS:")
                    print("=" * 80)
                    for error in validation_errors:
                        print(f"  - {error}")
                    return False
                
                print("\n" + "=" * 80)
                print("✓ ALL VALIDATIONS PASSED!")
                print("=" * 80)
                return True
            else:
                content = message.get('content', '')
                print("\n" + "=" * 80)
                print("⚠ TOOL CALLS NOT DETECTED")
                print("=" * 80)
                print(f"\n  Response content: {content}")
                print("\n  The model responded with text instead of making a tool call.")
                print("  This could mean:")
                print("    - Tool calls are not properly configured")
                print("    - The model chose not to use tools for this request")
                print("    - The tool_call_parser might not be working correctly")
                return False
        else:
            print("\n✗ No choices in response")
            return False
        
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out (server may still be loading the model)")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"\n✗ HTTP error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"  Details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"  Response: {e.response.text[:500]}")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "mistralai/Ministral-3-14B-Instruct-2512"
    
    print(f"\nTesting tool calls for model: {model}")
    print(f"Server URL: {base_url}\n")
    
    success = test_tool_calls(base_url, model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Tool call test PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ Tool call test FAILED or INCONCLUSIVE")
        print("=" * 80)
        sys.exit(1)
