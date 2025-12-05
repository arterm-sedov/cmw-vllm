#!/usr/bin/env python3
"""Simple script to test vLLM inference."""
import requests
import json
import sys

def test_vllm_inference(base_url: str = "http://localhost:8000", model: str = "mistralai/Ministral-3-14B-Instruct-2512"):
    """Test vLLM inference with a simple request."""
    
    # First check if server is running
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✓ Server is healthy")
        else:
            print(f"⚠ Server returned status {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to server at {base_url}")
        print("  Make sure the server is running: python3 -m cmw_vllm.cli start")
        return False
    except Exception as e:
        print(f"✗ Error checking server health: {e}")
        return False
    
    # Test inference
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }
    
    print(f"\nTesting inference...")
    print(f"  URL: {url}")
    print(f"  Model: {model}")
    print(f"  Request: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        print(f"\n✓ Inference successful!")
        print(f"  Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
        
        if 'usage' in result:
            usage = result['usage']
            print(f"\n  Token usage:")
            print(f"    Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
            print(f"    Completion tokens: {usage.get('completion_tokens', 'N/A')}")
            print(f"    Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        return True
        
    except requests.exceptions.Timeout:
        print("✗ Request timed out (server may still be loading the model)")
        return False
    except requests.exceptions.HTTPError as e:
        print(f"✗ HTTP error: {e}")
        if e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"  Details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"  Response: {e.response.text[:200]}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    model = sys.argv[2] if len(sys.argv) > 2 else "mistralai/Ministral-3-14B-Instruct-2512"
    
    success = test_vllm_inference(base_url, model)
    sys.exit(0 if success else 1)
