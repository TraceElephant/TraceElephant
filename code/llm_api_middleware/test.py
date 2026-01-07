#!/usr/bin/env python3
"""
Test script for tool calling functionality through the middleware.
Tests a bash command execution tool with a complete round trip.
"""

from openai import OpenAI
import json
import subprocess

# Initialize client pointing to the middleware
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="test-key"  # This will be replaced by the middleware
)

# Define a bash command execution tool
tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash_command",
            "description": "Execute a bash command and return its output. Useful for listing files, checking directories, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute, e.g. 'ls -la', 'pwd', 'echo hello'"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

def execute_bash_command(command: str) -> str:
    """Execute a bash command and return its output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout if result.stdout else result.stderr
        return output.strip() if output else "(no output)"
    except Exception as e:
        return f"Error executing command: {str(e)}"

# Make a request that should trigger tool calling
print("Testing tool calling functionality with bash command execution...")
print("=" * 70)

try:
    # Step 1: Initial request
    print("\n📤 Step 1: Sending initial request...")
    messages = [
        {"role": "user", "content": "Can you list the files in the current directory using ls -la?"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    print("✅ Request successful!")
    print(f"\nModel: {response.model}")
    print(f"Finish reason: {response.choices[0].finish_reason}")

    message = response.choices[0].message

    # Check if tool was called
    if message.tool_calls:
        print(f"\n🔧 Tool calls detected: {len(message.tool_calls)}")

        # Add assistant's response to messages
        messages.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in message.tool_calls
            ]
        })

        # Execute each tool call
        for i, tool_call in enumerate(message.tool_calls):
            print(f"\nTool call #{i+1}:")
            print(f"  ID: {tool_call.id}")
            print(f"  Function: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")

            # Parse and execute the command
            args = json.loads(tool_call.function.arguments)
            command = args.get("command", "")
            print(f"\n  📋 Executing command: {command}")

            # Execute the bash command
            result = execute_bash_command(command)
            print(f"  ✅ Command output:\n{result[:200]}..." if len(result) > 200 else f"  ✅ Command output:\n{result}")

            # Add tool response to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Step 2: Send tool results back to the model
        print("\n" + "=" * 70)
        print("📤 Step 2: Sending tool results back to model...")

        final_response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        print("✅ Final response received!")
        print(f"\nFinal message from assistant:")
        print(f"{final_response.choices[0].message.content}")

    else:
        print("\n⚠️  No tool calls in response")
        print(f"Assistant message: {message.content}")

    print("\n" + "=" * 70)
    print("✅ Test completed! Check the logs directory for saved request/response.")
    print("   You should see 2 log files: one for initial request and one for final response.")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\nMake sure the middleware server is running:")
    print("  python middleware.py")
