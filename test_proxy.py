#!/usr/bin/env python3
"""
Test script for the LLM proxy server.
This script sends a request to the proxy server and prints the streaming response.
"""

import json

import requests

# Proxy server URL
PROXY_URL = "http://localhost:8081/v1/chat/completions"


def test_proxy_server():
    """Test the proxy server with a simple chat completion request."""
    headers = {
        "Content-Type": "application/json",
        "Posit-Client-Type": "positron-assistant"  # This header should be stripped by the proxy
    }

    # Sample request based on the provided example
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "developer",
                "content": "You are a helpful assistant that provides concise answers."
            },
            {
                "role": "user",
                "content": "Tell me a haiku."
            }
        ],
        "stream": True,
        "stream_options": {
            "include_usage": True
        }
    }

    print("Sending request to proxy server...")
    print("URL:", PROXY_URL)
    print("Headers:", headers)
    print("Payload:", json.dumps(payload, indent=2))
    print("\nStreaming response:")

    try:
        # Send request to the proxy server with stream=True to get chunked response
        with requests.post(PROXY_URL, headers=headers, json=payload, stream=True) as response:
            if not response.ok:
                print(f"Error: {response.status_code} {response.text}")
                return

            # Process each chunk from the streamed response
            accumulated_text = []

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    parsed_data = None

                    # Handle the line based on content
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]  # Remove 'data: ' prefix
                        try:
                            # Parse and pretty-print JSON
                            parsed_data = json.loads(data_str)
                            pretty_json = json.dumps(parsed_data, indent=2)
                            print(f"data: {pretty_json}")

                            # Try to extract text content based on format
                            text = None

                            # OpenAI format
                            if "choices" in parsed_data and len(parsed_data["choices"]) > 0:
                                delta = parsed_data["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    text = delta["content"]

                            # Anthropic format
                            elif parsed_data.get("type") == "content_block_delta":
                                delta = parsed_data.get("delta", {})
                                if delta.get("type") == "text_delta":
                                    text = delta.get("text", "")

                            # OpenAI Responses format
                            elif parsed_data.get("type") == "response.output_text.delta":
                                text = parsed_data.get("delta")

                            # Add to accumulated text if we found content
                            if text:
                                accumulated_text.append(text)

                        except json.JSONDecodeError:
                            # Not valid JSON, print as-is
                            print(decoded_line)
                    else:
                        # Not JSON data, print as-is
                        print(decoded_line)

            # Display the accumulated complete response
            if accumulated_text:
                print("\n" + "=" * 60)
                print("Accumulated Complete Response:")
                print("=" * 60)
                print("".join(accumulated_text))
                print("=" * 60)

    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return


if __name__ == "__main__":
    test_proxy_server()
