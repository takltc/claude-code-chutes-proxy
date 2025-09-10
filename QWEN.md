# Claude-to-Chutes Proxy - Project Context

## Project Overview

This is a Python-based proxy service that translates Anthropic Claude API requests to Chutes (OpenAI-compatible) backend requests and vice versa. The service acts as a bridge between Anthropic-compatible clients and OpenAI-compatible backends, enabling the use of Chutes infrastructure with tools designed for Anthropic's API.

### Core Functionality
- Translates Anthropic Claude `v1/messages` requests to OpenAI-compatible `/v1/chat/completions` requests
- Converts Chutes/OpenAI-style responses back to Anthropic-compatible responses
- Supports both non-streaming and streaming (SSE) for text chats
- Handles tool/function calling:
  - Non-streaming: Anthropic `tools` + tool_use/tool_result ↔ OpenAI tool_calls/tool messages
  - Streaming: tool_use bridging with input_json_delta streaming
- Supports multimodal content (image blocks)
- Automatic model name mapping and case correction

### Technologies Used
- **Python 3.10+** as the primary language
- **FastAPI** for the web framework
- **Uvicorn** as the ASGI server
- **HTTPX** for asynchronous HTTP requests
- **Pydantic** for data validation
- **sglang** for function call parsing
- **Docker** for containerization

## Project Structure

```
.
├── app/                    # Main application code
│   ├── schemas/           # Data models and schemas
│   ├── config.py          # Configuration management
│   ├── main.py            # Main FastAPI application
│   ├── transform.py       # Request/response transformation logic
│   └── __init__.py
├── tests/                 # Test suite
├── Dockerfile             # Docker image definition
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies
├── run.sh                 # Application entry point script
├── README.md              # Project documentation
├── VERSION                # Version file
├── pytest.ini             # Pytest configuration
└── .env.example           # Example environment configuration
```

## Key Components

### Main Application (app/main.py)
- FastAPI application that handles `/v1/messages` POST requests
- Implements both streaming and non-streaming response handling
- Manages authentication header forwarding to upstream
- Provides model name resolution and case correction
- Includes error handling and mapping to Anthropic-style errors

### Transformation Logic (app/transform.py)
- `anthropic_to_openai_payload()`: Converts Anthropic requests to OpenAI format
- `openai_to_anthropic_response()`: Converts OpenAI responses to Anthropic format
- Handles complex mappings for:
  - Text content blocks
  - Image content blocks
  - Tool use/result blocks
  - System messages
  - Streaming content deltas

### Configuration (app/config.py)
- Environment-based configuration management
- Model name mapping support
- Tool name mapping support
- Authentication style configuration
- Debug and retry settings

### Data Models (app/schemas/)
- Pydantic models for Anthropic API request/response validation
- Streaming event schemas
- Error response formats

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUTES_BASE_URL` | `https://llm.chutes.ai` | Chutes/OpenAI-compatible backend URL |
| `CHUTES_API_KEY` | - | Optional API key for backend |
| `CHUTES_AUTH_STYLE` | `both` | Auth forwarding: `header`, `env`, or `both` |
| `MODEL_MAP` | `{}` | JSON string for model name mapping |
| `TOOL_NAME_MAP` | `{}` | JSON string for tool name mapping |
| `AUTO_FIX_MODEL_CASE` | `1` | Auto-correct model casing |
| `DEBUG_PROXY` | `0` | Enable request/response logging |
| `PROXY_BACKOFF_ON_429` | `1` | Retry on rate limiting |
| `PROXY_MAX_RETRY_ON_429` | `1` | Max retry attempts for 429 |
| `PROXY_MAX_RETRY_AFTER` | `2` | Max retry-after seconds |
| `UVICORN_WORKERS` | `1` | Uvicorn worker processes |
| `PORT` | `8080` | Internal container port |

## Building and Running

### Development Setup
1. Install dependencies:
   ```bash
   python -m ensurepip --upgrade
   python -m pip install -r requirements.txt
   ```

2. Set environment variables:
   ```bash
   export CHUTES_BASE_URL=https://llm.chutes.ai
   ```

3. Run the application:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8090
   ```

### Docker Deployment
1. Using Docker Compose (recommended):
   ```bash
   docker compose up --build
   ```

2. Using prebuilt image:
   ```bash
   docker run --rm \
     -p 8090:8080 \
     -e CHUTES_BASE_URL=${CHUTES_BASE_URL:-https://llm.chutes.ai} \
     ghcr.io/takltc/claude-code-chutes-proxy:0.0.1
   ```

## Testing

Run the test suite with pytest:
```bash
pytest
```

The tests cover:
- Request transformation (Anthropic → OpenAI)
- Response transformation (OpenAI → Anthropic)
- Tool call parsing
- Streaming content handling
- Model name mapping

## API Usage

### Endpoint
POST `http://localhost:8090/v1/messages`

### Example Request
```json
{
  "model": "claude-3.5-sonnet",
  "max_tokens": 512,
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
  ]
}
```

### Features Supported
- Text content (full support)
- Tool calling (both streaming and non-streaming)
- Image blocks (user/system images converted to OpenAI image_url)
- Streaming responses with Anthropic-style SSE events
- Model name mapping and automatic case correction

## Development Conventions

- Follow FastAPI best practices for route handling
- Use Pydantic models for request/response validation
- Maintain backward compatibility with Anthropic API
- Handle errors gracefully and map to Anthropic-style errors
- Keep dependencies minimal and well-documented
- Write tests for new functionality
- Use environment variables for configuration