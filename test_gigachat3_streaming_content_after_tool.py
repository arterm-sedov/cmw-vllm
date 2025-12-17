#!/usr/bin/env python3
"""Test streaming with content after tool call results."""
import requests
import json
import sys

def test_streaming_content_after_tool_result(base_url: str = "http://localhost:8000", model: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"):
    """Test streaming where content is generated after tool results."""
    
    print("=" * 80)
    print("Testing Streaming: Content After Tool Result (GigaChat3)")
    print("=" * 80)
    
    url = f"{base_url}/v1/chat/completions"
    
    # Step 1: Make tool call
    print("\n1. Making initial request with tool call...")
    payload1 = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 100
    }
    
    response1 = requests.post(url, json=payload1, stream=True, timeout=60)
    response1.raise_for_status()
    
    tool_call_id = None
    tool_name = None
    tool_args = ""
    
    print("  Streaming tool call...")
    for line in response1.iter_lines():
        if line.startswith(b"data: "):
            data = line[6:].decode('utf-8')
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                if 'tool_calls' in delta:
                    for tc in delta['tool_calls']:
                        if 'id' in tc:
                            tool_call_id = tc['id']
                        if 'function' in tc:
                            if 'name' in tc['function']:
                                tool_name = tc['function']['name']
                            if 'arguments' in tc['function']:
                                tool_args += tc['function']['arguments']
            except:
                pass
    
    if not tool_call_id:
        print("  ✗ No tool call received")
        return False
    
    print(f"  ✓ Tool call: {tool_name}({tool_args[:50]}...)")
    
    # Step 2: Provide tool result and expect content
    print("\n2. Providing tool result and expecting content response...")
    payload2 = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is the weather in San Francisco?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": tool_args}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps({"temperature": 72, "condition": "sunny", "location": "San Francisco"})
            }
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city and state"}
                    },
                    "required": ["location"]
                }
            }
        }],
        "stream": True,
        "max_tokens": 200
    }
    
    print("  Streaming content response...")
    response2 = requests.post(url, json=payload2, stream=True, timeout=60)
    response2.raise_for_status()
    
    content_chunks = []
    tool_call_chunks = []
    chunk_count = 0
    
    for line in response2.iter_lines():
        if line.startswith(b"data: "):
            data = line[6:].decode('utf-8')
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                chunk_count += 1
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                
                if 'content' in delta and delta['content']:
                    content_chunks.append(delta['content'])
                    print(f"  [Content] {delta['content']}", end='', flush=True)
                
                if 'tool_calls' in delta:
                    tool_call_chunks.append(delta['tool_calls'])
                    print(f"\n  [Tool Call] detected")
            except:
                pass
    
    print(f"\n\n  Total chunks: {chunk_count}")
    print(f"  Content chunks: {len(content_chunks)}")
    print(f"  Tool call chunks: {len(tool_call_chunks)}")
    
    # Results
    print("\n" + "=" * 80)
    if content_chunks:
        full_content = ''.join(content_chunks)
        print("✓ STREAMING CONTENT AFTER TOOL RESULT WORKS!")
        print("=" * 80)
        print(f"\n  Full content ({len(full_content)} chars):")
        print(f"  {full_content}")
        return True
    else:
        print("⚠ No content received after tool result")
        print("=" * 80)
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "ai-sage/GigaChat3-10B-A1.8B-bf16"
    
    print(f"\nTesting streaming content after tool result")
    print(f"Model: {model}")
    print(f"Server: {base_url}\n")
    
    success = test_streaming_content_after_tool_result(base_url, model)
    
    if success:
        print("\n" + "=" * 80)
        print("✓ Test PASSED")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("✗ Test FAILED")
        print("=" * 80)
        sys.exit(1)
