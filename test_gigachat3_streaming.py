#!/usr/bin/env python3
"""Test streaming inference with tool calling for GigaChat3 model."""
import requests
import json
import sys
import time

def test_streaming_tool_calls(base_url: str = "http://localhost:8000", model: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"):
    """Test streaming tool call support with GigaChat3."""
    
    print("=" * 80)
    print("Testing Streaming Tool Calls with GigaChat3")
    print("=" * 80)
    
    # Check if server is running
    print("\n1. Checking server status...")
    try:
        models_response = requests.get(f"{base_url}/v1/models", timeout=5)
        if models_response.status_code == 200:
            print("✓ Server is responding")
            models_data = models_response.json()
            available_models = [m.get("id") for m in models_data.get("data", [])]
            if model not in available_models:
                print(f"⚠ Model '{model}' not found, using '{available_models[0]}' instead")
                model = available_models[0]
        else:
            print(f"⚠ Server returned status {models_response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {base_url}")
        return False
    except Exception as e:
        print(f"✗ Error checking server status: {e}")
        return False
    
    # Define tools
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
    
    # Test streaming request
    print("\n2. Testing streaming tool call request...")
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
        "tool_choice": "auto",
        "stream": True,  # Enable streaming
        "max_tokens": 200,
        "temperature": 0.1
    }
    
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Streaming: enabled")
    
    try:
        print("\n  Sending streaming request...")
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        print("\n  Streaming response chunks:")
        print("  " + "-" * 76)
        
        tool_calls_detected = False
        tool_call_id = None
        tool_name = None
        tool_arguments = ""
        content_chunks = []
        delta_count = 0
        
        for line in response.iter_lines():
            if not line:
                continue
                
            # SSE format: "data: {...}"
            if line.startswith(b"data: "):
                data_str = line[6:].decode('utf-8')
                if data_str.strip() == "[DONE]":
                    print("\n  [DONE]")
                    break
                
                try:
                    chunk = json.loads(data_str)
                    delta_count += 1
                    
                    choices = chunk.get('choices', [])
                    if choices:
                        delta = choices[0].get('delta', {})
                        
                        # Check for tool calls in delta
                        if 'tool_calls' in delta and delta['tool_calls']:
                            for tool_call_delta in delta['tool_calls']:
                                if 'id' in tool_call_delta:
                                    tool_call_id = tool_call_delta['id']
                                    tool_calls_detected = True
                                    print(f"\n  ✓ Tool call detected! ID: {tool_call_id}")
                                
                                if 'function' in tool_call_delta:
                                    func_delta = tool_call_delta['function']
                                    if 'name' in func_delta:
                                        tool_name = func_delta['name']
                                        print(f"  ✓ Tool name: {tool_name}")
                                    if 'arguments' in func_delta:
                                        tool_arguments += func_delta['arguments']
                                        print(f"  ✓ Arguments chunk: {func_delta['arguments'][:50]}...")
                        
                        # Check for content
                        if 'content' in delta and delta['content']:
                            content_chunks.append(delta['content'])
                            print(f"  Content chunk: {delta['content'][:50]}...")
                    
                    # Print every 10th chunk for brevity
                    if delta_count % 10 == 0:
                        print(f"  ... ({delta_count} chunks received)")
                        
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Failed to parse chunk: {e}")
                    continue
        
        print("\n  " + "-" * 76)
        print(f"\n  Total chunks received: {delta_count}")
        
        # Analyze results
        print("\n" + "=" * 80)
        if tool_calls_detected:
            print("✓ STREAMING TOOL CALLS WORKING!")
            print("=" * 80)
            print(f"\n  Tool Call ID: {tool_call_id}")
            print(f"  Tool Name: {tool_name}")
            print(f"  Arguments (streamed): {tool_arguments[:200]}...")
            
            # Try to parse arguments
            try:
                parsed_args = json.loads(tool_arguments)
                print(f"\n  Parsed arguments: {json.dumps(parsed_args, indent=4)}")
            except json.JSONDecodeError:
                print(f"\n  ⚠ Arguments not complete JSON (may be partial): {tool_arguments}")
            
            if content_chunks:
                print(f"\n  Content chunks: {len(content_chunks)} chunks")
                print(f"  Total content: {''.join(content_chunks)[:100]}...")
            
            return True
        else:
            print("⚠ TOOL CALLS NOT DETECTED IN STREAMING")
            print("=" * 80)
            if content_chunks:
                print(f"\n  Received {len(content_chunks)} content chunks instead")
                print(f"  Content: {''.join(content_chunks)[:200]}...")
            return False
        
    except requests.exceptions.Timeout:
        print("\n✗ Request timed out")
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
    model = sys.argv[2] if len(sys.argv) > 2 else "ai-sage/GigaChat3-10B-A1.8B-bf16"
    
    print(f"\nTesting streaming tool calls for model: {model}")
    print(f"Server URL: {base_url}\n")
    
    success = test_streaming_tool_calls(base_url, model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Streaming tool call test PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ Streaming tool call test FAILED or INCONCLUSIVE")
        print("=" * 80)
        sys.exit(1)
