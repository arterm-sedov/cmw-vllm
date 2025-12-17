#!/usr/bin/env python3
"""Test streaming inference with both content and tool calls for GigaChat3 model."""
import requests
import json
import sys
import time

def test_streaming_with_content_and_tool_calls(base_url: str = "http://localhost:8000", model: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"):
    """Test streaming with both content and tool calls."""
    
    print("=" * 80)
    print("Testing Streaming with Content + Tool Calls (GigaChat3)")
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
        },
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    # Test 1: Request that might generate content before tool call
    print("\n2. Test 1: Request that may include content before tool call...")
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "I need to check the weather in San Francisco and also search for information about Python. Can you help me with both?"
            }
        ],
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 300,
        "temperature": 0.1
    }
    
    print(f"  Request: User asks for weather check AND web search")
    print(f"  Expected: Possibly content + multiple tool calls")
    
    try:
        print("\n  Sending streaming request...")
        response = requests.post(url, json=payload, stream=True, timeout=120)
        response.raise_for_status()
        
        print("\n  Streaming response chunks:")
        print("  " + "-" * 76)
        
        content_chunks = []
        tool_calls = {}  # {tool_call_id: {name, arguments}}
        current_tool_call_id = None
        chunk_types = []  # Track order: 'content' or 'tool_call'
        
        delta_count = 0
        
        for line in response.iter_lines():
            if not line:
                continue
                
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
                        
                        # Check for content
                        if 'content' in delta and delta['content']:
                            content_chunks.append(delta['content'])
                            chunk_types.append('content')
                            print(f"  [Content] {delta['content']}")
                        
                        # Check for tool calls
                        if 'tool_calls' in delta and delta['tool_calls']:
                            for tool_call_delta in delta['tool_calls']:
                                # Tool call ID
                                if 'id' in tool_call_delta:
                                    current_tool_call_id = tool_call_delta['id']
                                    if current_tool_call_id not in tool_calls:
                                        tool_calls[current_tool_call_id] = {
                                            'name': None,
                                            'arguments': ''
                                        }
                                    chunk_types.append('tool_call')
                                    print(f"\n  [Tool Call] ID: {current_tool_call_id}")
                                
                                # Tool call function
                                if 'function' in tool_call_delta and current_tool_call_id:
                                    func_delta = tool_call_delta['function']
                                    if 'name' in func_delta:
                                        tool_calls[current_tool_call_id]['name'] = func_delta['name']
                                        print(f"  [Tool Call] Name: {func_delta['name']}")
                                    if 'arguments' in func_delta:
                                        tool_calls[current_tool_call_id]['arguments'] += func_delta['arguments']
                                        print(f"  [Tool Call] Args chunk: {func_delta['arguments'][:40]}...")
                    
                    # Print progress every 20 chunks
                    if delta_count % 20 == 0:
                        print(f"  ... ({delta_count} chunks received)")
                        
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Failed to parse chunk: {e}")
                    continue
        
        print("\n  " + "-" * 76)
        print(f"\n  Total chunks received: {delta_count}")
        print(f"  Content chunks: {len(content_chunks)}")
        print(f"  Tool calls detected: {len(tool_calls)}")
        
        # Analyze results
        print("\n" + "=" * 80)
        print("RESULTS:")
        print("=" * 80)
        
        # Content analysis
        if content_chunks:
            full_content = ''.join(content_chunks)
            print(f"\n✓ Content received: {len(full_content)} characters")
            print(f"  Content preview: {full_content[:150]}...")
        else:
            print("\n⚠ No content chunks received")
        
        # Tool calls analysis
        if tool_calls:
            print(f"\n✓ Tool calls received: {len(tool_calls)}")
            for tool_id, tool_data in tool_calls.items():
                print(f"\n  Tool Call {tool_id}:")
                print(f"    Name: {tool_data['name']}")
                print(f"    Arguments: {tool_data['arguments'][:100]}...")
                try:
                    parsed_args = json.loads(tool_data['arguments'])
                    print(f"    Parsed: {json.dumps(parsed_args, indent=6)}")
                except json.JSONDecodeError:
                    print(f"    ⚠ Arguments not valid JSON (may be incomplete)")
        else:
            print("\n⚠ No tool calls detected")
        
        # Order analysis
        print(f"\n  Chunk order (first 20): {chunk_types[:20]}")
        if 'content' in chunk_types and 'tool_call' in chunk_types:
            first_content_idx = chunk_types.index('content') if 'content' in chunk_types else -1
            first_tool_idx = chunk_types.index('tool_call') if 'tool_call' in chunk_types else -1
            if first_content_idx >= 0 and first_tool_idx >= 0:
                if first_content_idx < first_tool_idx:
                    print("  ✓ Content appears before tool calls")
                else:
                    print("  ✓ Tool calls appear before content")
        
        # Success criteria
        success = len(tool_calls) > 0
        if content_chunks:
            print("\n" + "=" * 80)
            print("✓ STREAMING WITH BOTH CONTENT AND TOOL CALLS WORKING!")
            print("=" * 80)
        elif tool_calls:
            print("\n" + "=" * 80)
            print("✓ STREAMING TOOL CALLS WORKING (no content in this response)")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("✗ NO TOOL CALLS DETECTED")
            print("=" * 80)
        
        return success
        
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
    
    print(f"\nTesting streaming with content + tool calls for model: {model}")
    print(f"Server URL: {base_url}\n")
    
    success = test_streaming_with_content_and_tool_calls(base_url, model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Test PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ Test FAILED or INCONCLUSIVE")
        print("=" * 80)
        sys.exit(1)
