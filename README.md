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

Install

```
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

Run

```
export CHUTES_BASE_URL=http://localhost:8000
uvicorn app.main:app --host 0.0.0.0 --port 8090
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

Docker

- Build and run with Compose (recommended):
  - `docker compose up --build`
  - Exposes `http://localhost:8090` → container `8090`.
  - Configure env via your shell or a `.env` file. Common vars:
    - `CHUTES_BASE_URL` (default `https://llm.chutes.ai`)
    - `CHUTES_API_KEY` (optional)
    - `MODEL_MAP`, `TOOL_NAME_MAP` (JSON strings)
    - `AUTO_FIX_MODEL_CASE`, `DEBUG_PROXY`
- Manual Docker build/run:
  - Build: `docker build -t claude-chutes-proxy .`
  - Run: `docker run --rm -p 8090:8090 -e CHUTES_BASE_URL=$CHUTES_BASE_URL claude-chutes-proxy`

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
