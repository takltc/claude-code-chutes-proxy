Claude-to-Chutes Proxy

Overview

- Translates Anthropic Claude `v1/messages` requests to a Chutes LLM backend (OpenAI-compatible `/v1/chat/completions`).
- Converts Chutes/OpenAI-style responses back to Anthropic-compatible responses.
- Supports non-streaming and streaming (SSE) for text chats.
- Supports tools:
  - Non-streaming: Anthropic `tools` + tool_use/tool_result ↔ OpenAI tool_calls/tool messages.
  - Streaming: tool_use bridging is supported (streams input_json_delta for function arguments).

Quickstart

- Requirements: Python 3.10+ recommended (works with 3.13), `uvicorn` and `fastapi`.
- Env vars:
  - `CHUTES_BASE_URL`: Base URL of your Chutes/OpenAI backend (e.g. `https://llm.chutes.ai`).
  - `CHUTES_API_KEY` (optional): If your backend requires Bearer auth. If not set, the proxy will forward the inbound `x-api-key` or `Authorization` header to upstream.
  - `MODEL_MAP` (optional): JSON mapping for Anthropic→backend model names, e.g. `{"claude-3.5-sonnet": "Qwen2-72B-Instruct"}`.
- `DEBUG_PROXY` (optional): `1/true/yes` to log upstream payload metadata (helps verify model casing). The proxy preserves outward-facing casing but will auto-correct upstream model casing when enabled.
- `AUTO_FIX_MODEL_CASE` (optional, default on): Auto-correct model casing against `/v1/models` when needed; includes a small heuristic fallback for known providers (e.g., Moonshot Kimi).
 - `DISCOVERY_MIN_INTERVAL` (seconds, default 300): Minimum interval to refresh model list/schemas to avoid rate limits.
 - Schema discovery: On startup (or first request with auth), the proxy queries `/v1/models` and tries `/v1/models/{id}` to build a lightweight capability map (tools/vision/reasoning). Payloads are adapted per model.
 - `PROXY_BACKOFF_ON_429` (default on): For non-stream requests, honors small `Retry-After` and retries once.
 - Inspect discovered models: `GET /_schemas`.

Quick Start Options

### Option 1: Docker (Recommended)

```bash
# Clone and start via Docker Compose
git clone https://github.com/takltc/claude-code-chutes-proxy
cd claude-code-chutes-proxy
docker compose up --build

# The proxy will be available at http://localhost:8090
```

### Option 2: Local Python

```bash
# Install dependencies
python -m ensurepip --upgrade
python -m pip install -r requirements.txt

# Set environment and run
export CHUTES_BASE_URL=https://llm.chutes.ai
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

Install

```
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

Run

```
export CHUTES_BASE_URL=https://llm.chutes.ai
uvicorn app.main:app --host 0.0.0.0 --port 8090
```

<b>Run with Claude Code</b>

```
ANTHROPIC_BASE_URL="http://localhost:8090" ANTHROPIC_API_KEY="your-chutes-api-key" ANTHROPIC_MODEL="moonshotai/Kimi-K2-Instruct-0905" ANTHROPIC_SMALL_FAST_MODEL="moonshotai/Kimi-K2-Instruct-0905" claude --dangerously-skip-permissions
```

Usage (Anthropic-compatible)

POST `http://localhost:8090/v1/messages`

Body example:

```
{
  "model": "claude-3.5-sonnet",
  "max_tokens": 512,
  "messages": [
    {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
  ]
}
```

 Notes

- Text content is fully supported. Tools are supported both in non-streaming and streaming (tool_use) modes.
- Images and multimodal: request-side user/system image blocks are translated to OpenAI `image_url` content entries (non-streaming). Assistant image outputs are not mapped back (rare in OpenAI response).
- Streaming emits Anthropic-style SSE events for text deltas. Token usage is reported at end when available from backend.
- If your Chutes backend already exposes OpenAI-compatible endpoints (e.g. vLLM/SGLang templates), you can point `CHUTES_BASE_URL` directly to that service.

Environment Configuration

### Docker Compose with .env File

Create a `.env` file in your project root:

```env
CHUTES_BASE_URL=http://your-chutes-backend:8000
CHUTES_API_KEY=your-api-key-if-required
MODEL_MAP={"claude-3.5-sonnet": "Qwen2-72B-Instruct", "claude-3-haiku": "Llama-3.1-8B-Instruct"}
DEBUG_PROXY=1
```

Then uncomment the `env_file` line in `docker-compose.yml`:

```yaml
services:
  proxy:
    # ... existing config ...
    env_file:
      - .env
```

### Environment Variable Reference

All environment variables with their defaults and descriptions:

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

Docker

- Build and run with Compose (recommended):
  - `docker compose up --build`
  - Exposes `http://localhost:8090` → container `8080` (mapped from host port 8090).
  - Includes health checks with automatic restart
  - Authoritative list of configurable environment variables:
    - `CHUTES_BASE_URL` (default `https://llm.chutes.ai`) - Chutes/OpenAI backend URL
    - `CHUTES_API_KEY` (optional) - Backend API key
    - `CHUTES_AUTH_STYLE` (default `both`) - Auth forwarding behavior
    - `MODEL_MAP` (default `{}`) - JSON string mapping Anthropic→backend model names
    - `TOOL_NAME_MAP` (default `{}`) - JSON string mapping tool names
    - `AUTO_FIX_MODEL_CASE` (default `1`) - Auto-correct model casing
    - `DEBUG_PROXY` (default `0`) - Enable request/response logging
    - `PROXY_BACKOFF_ON_429` (default `1`) - Retry on rate limiting
    - `PROXY_MAX_RETRY_ON_429` (default `1`) - Max 429 retry attempts
    - `PROXY_MAX_RETRY_AFTER` (default `2`) - Max retry-after seconds
    - `UVICORN_WORKERS` (default `1`) - Number of Uvicorn workers
    - `PORT` (default `8080`) - Internal container port
- Manual Docker build/run:
  - Build: `docker build -t claude-chutes-proxy .`
  - Run: `docker run --rm -p 8090:8080 -e CHUTES_BASE_URL=$CHUTES_BASE_URL claude-chutes-proxy`
  - The container runs on port 8080 internally (exposed as 8090 on host)
  - Includes health checks every 30 seconds

Docker usage example

```
curl -sS -X POST http://localhost:8090/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: YOUR_KEY' \
  -d '{
    "model": "claude-3.5-sonnet",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
  }'
```
