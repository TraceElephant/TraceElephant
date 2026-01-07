# LLM API Middleware

A middleware server that intercepts and logs all LLM API inputs and outputs. Supports OpenAI-compatible APIs.

## Features

- Supports all OpenAI-compatible API endpoints
- Automatically saves all requests and responses to JSON files
- Supports both streaming and non-streaming responses
- Fully transparent proxy that doesn't affect original API calls
- Configurable logging settings

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Copy `config.yaml.example` to `config.yaml`, and edit the `config.yaml` file:

```yaml
# Upstream LLM API Configuration (the actual API you want to call)
upstream:
  # IMPORTANT: base_url should NOT include /v1 at the end
  base_url: "https://api.openai.com"     # API endpoint (without /v1)
  api_key: "your-api-key-here"           # Your API key
  model_name: "gpt-4o"                   # Default model name

# Local Server Configuration (your middleware server)
server:
  host: "0.0.0.0"
  port: 8000

# Logging Configuration
logging:
  log_dir: "./logs"        # Directory to save request/response logs
  enabled: true            # Whether to save logs
  console_output: true     # Whether to print logs to console
```

**Important Note:** The `base_url` should **NOT** include `/v1` at the end. The middleware automatically appends the full path (e.g., `/v1/chat/completions`).

Examples:
- ✅ Correct: `https://api.openai.com`
- ❌ Wrong: `https://api.openai.com/v1`

## Usage

### 1. Start the middleware server

```bash
python middleware.py
```

The server will start at `http://localhost:8000`.

### 2. Configure your client

In your code, change the API base_url to the middleware server address:

**Python (using openai library):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"  # This will be replaced by the middleware
)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

**curl:**
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Test:**

```bash
python test.py
```

### 3. View logs

All API calls will be saved to the `logs` directory with filenames in the format `{timestamp}.json`.

Log file example:
```json
{
  "timestamp": "2024-01-07T12:34:56.789012",
  "endpoint": "/v1/chat/completions",
  "status_code": 200,
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "model": "gpt-4o",
    "stream": false
  },
  "output": {
    "id": "chatcmpl-...",
    "object": "chat.completion",
    "created": 1704628496,
    "model": "gpt-4o",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "Hello! How can I help you today?"
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 8,
      "completion_tokens": 9,
      "total_tokens": 17
    }
  }
}
```

## Important Notes

1. Ensure the `api_key` in `config.yaml` is correct
2. If you need to change the port, configure it in `config.yaml`
3. Log files may consume significant disk space, clean them regularly
4. Do not commit configuration files containing API keys to version control

## Troubleshooting

### Server won't start
- Check if the port is already in use
- Ensure the configuration file format is correct

### Request failed
- Verify the upstream API address is correct
- Confirm the API key is valid
- Check console output for error messages

### Logs not saved
- Check if `logging.enabled` is set to `true` in `config.yaml`
- Ensure you have write permissions for the `log_dir`
