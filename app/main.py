from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator, Dict
import uuid

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from collections import deque

from .config import settings
from .schemas.anthropic import (
    ErrorResponse,
    MessageResponse,
    MessagesRequest,
)
from .transform import (
    anthropic_to_openai_payload,
    map_finish_reason,
    openai_to_anthropic_response,
    choose_tool_call_parser,
)


app = FastAPI(title="Claude-to-Chutes Proxy")
_RECENT = deque(maxlen=64)


def _auth_headers(x_api_key: str | None = None, authorization: str | None = None) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    # Build upstream auth headers according to configured style.
    style = (settings.chutes_auth_style or "both").lower()
    # Prefer explicit env var, else inbound x-api-key, else try bearer from inbound Authorization
    api_key: str | None = settings.chutes_api_key or x_api_key
    bearer_inbound: str | None = authorization

    def set_x_api_key(val: str) -> None:
        if val:
            headers["x-api-key"] = val

    def set_authorization(val: str) -> None:
        if not val:
            return
        if val.lower().startswith("bearer "):
            headers["Authorization"] = val
        else:
            headers["Authorization"] = f"Bearer {val}"

    if style == "x-api-key":
        # Only x-api-key
        if api_key:
            set_x_api_key(api_key)
        elif bearer_inbound:
            # try extracting token from inbound Authorization
            token = bearer_inbound.split(" ", 1)[1] if " " in bearer_inbound else bearer_inbound
            set_x_api_key(token)
    elif style == "authorization":
        # Only Authorization
        if bearer_inbound:
            set_authorization(bearer_inbound)
        elif api_key:
            set_authorization(api_key)
    else:
        # both (default): include as many as we can
        if api_key:
            set_x_api_key(api_key)
            set_authorization(api_key)
        else:
            if bearer_inbound:
                set_authorization(bearer_inbound)
            if x_api_key:
                set_x_api_key(x_api_key)

    return headers


def _anthropic_error_for_status(status: int, message: str) -> Dict[str, Any]:
    t = "api_error"
    if status == 400:
        t = "invalid_request_error"
    elif status == 401:
        t = "authentication_error"
    elif status == 403:
        t = "permission_error"
    elif status == 404:
        t = "not_found_error"
    elif status == 429:
        t = "rate_limit_error"
    elif 500 <= status < 600:
        t = "api_error"
    return ErrorResponse(error={"type": t, "message": message}).model_dump()


# simple in-memory cache for model resolution
_MODEL_RESOLVE_CACHE: Dict[str, str] = {}


async def _try_resolve_model(
    requested_model: str, headers: Dict[str, str]
) -> str | None:
    # Use cache first
    if requested_model in _MODEL_RESOLVE_CACHE:
        return _MODEL_RESOLVE_CACHE[requested_model]
    url = f"{settings.chutes_base_url.rstrip('/')}/v1/models"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            resp = await client.get(url, headers=headers)
        except Exception:
            return None
    if resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    items = []
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data.get("models"), list):
            items = data["models"]
    # collect ids
    ids = []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            ids.append(it["id"])
        elif isinstance(it, str):
            ids.append(it)
    # exact match only (preserve case)
    if requested_model in ids:
        _MODEL_RESOLVE_CACHE[requested_model] = requested_model
        return requested_model
    return None


def _guess_case_for_known_models(requested_model: str) -> str | None:
    """Best-effort local guess for canonical casing of some known providers.

    Used only as a last-resort when provider discovery (/v1/models) is unavailable
    or does not include the requested model id.
    """
    try:
        s = (requested_model or "").strip()
        if not s or "/" not in s:
            return None
        vendor, name = s.split("/", 1)
        vlow = vendor.lower()
        nlow = name.strip().lower()
        # Moonshot Kimi K2 family commonly requires specific title-casing
        if vlow == "moonshotai" and (nlow.startswith("kimi-k2-") or nlow == "kimi-k2"):
            parts = [p for p in nlow.split("-") if p != ""]
            fixed: list[str] = []
            for p in parts:
                if p == "kimi":
                    fixed.append("Kimi")
                elif p == "k2":
                    fixed.append("K2")
                elif p.isdigit():
                    fixed.append(p)
                else:
                    fixed.append(p[:1].upper() + p[1:].lower())
            return f"{vendor}/{'-'.join(fixed)}"
    except Exception:
        return None
    return None


async def _resolve_case_variant(
    requested_model: str, headers: Dict[str, str]
) -> str | None:
    """Resolve to provider-registered model id when request differs by case or common suffixes.

    Strategy:
    1) exact id in list -> return as is
    2) case-insensitive equality -> return canonical id
    3) fuzzy fallback: strip common trailing size/context suffixes (e.g., -75k) and match
    """
    import re

    def _normalize(mid: str) -> str:
        s = (mid or "").strip().lower().replace("_", "-")
        # Drop trailing size/context suffixes like -32k/-64k/-75k/-100k/-128k
        s = re.sub(r"-\s*\d+\s*[kK]$", "", s)
        return s

    url = f"{settings.chutes_base_url.rstrip('/')}/v1/models"
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
        try:
            resp = await client.get(url, headers=headers)
        except Exception:
            return None
    if resp.status_code >= 400:
        return None
    try:
        data = resp.json()
    except Exception:
        return None
    items = []
    if isinstance(data, dict):
        if isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data.get("models"), list):
            items = data["models"]
    ids = []
    for it in items:
        if isinstance(it, dict) and it.get("id"):
            ids.append(it["id"])
        elif isinstance(it, str):
            ids.append(it)
    # 1) exact match
    if requested_model in ids:
        return requested_model
    # 2) case-insensitive equality
    req_lower = requested_model.lower()
    lower_map = {str(mid).lower(): str(mid) for mid in ids if isinstance(mid, str)}
    if req_lower in lower_map and lower_map[req_lower] != requested_model:
        return lower_map[req_lower]
    # 3) fuzzy normalize (e.g., drop -75k, unify separators)
    req_norm = _normalize(requested_model)
    norm_map: Dict[str, str] = {}
    for mid in ids:
        if isinstance(mid, str):
            norm = _normalize(mid)
            norm_map[norm] = mid
    if req_norm in norm_map and norm_map[req_norm] != requested_model:
        return norm_map[req_norm]
    # 4) heuristic last-resort for known providers when discovery failed
    try:
        guessed = _guess_case_for_known_models(requested_model)
        if guessed and guessed != requested_model:
            return guessed
    except Exception:
        ...
    return None


@app.post("/v1/messages")
async def messages(
    request: Request,
    anthropic_version: str | None = Header(default=None, alias="anthropic-version"),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
):
    _rec: Dict[str, Any] = {"phase": "start"}
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    # Validate minimal schema
    try:
        parsed = MessagesRequest.model_validate(body)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error={"type": "invalid_request_error", "message": str(e)}).model_dump(),
        )
    _rec["request_model"] = parsed.model

    # Build OpenAI payload for Chutes（OpenAI 侧仅使用 sglang 协议模型校验/构造）
    oai_payload = anthropic_to_openai_payload(parsed.model_dump())
    _rec["oai_model"] = oai_payload.get("model")
    # Preserve original requested model casing for outward-facing responses
    display_model = parsed.model

    # Choose streaming or not based on request
    is_stream = bool(oai_payload.get("stream"))
    chutes_url = f"{settings.chutes_base_url.rstrip('/')}/v1/chat/completions"

    if not is_stream:
        # Auto-fix model case if configured
        if settings.auto_fix_model_case and oai_payload.get("model"):
            try:
                fixed = await _resolve_case_variant(oai_payload["model"], _auth_headers(x_api_key, authorization))
                if fixed and fixed != oai_payload["model"]:
                    if settings.debug:
                        print(f"[proxy] non-stream: auto-fix model case '{oai_payload['model']}' -> '{fixed}'")
                    oai_payload["model"] = fixed
            except Exception:
                ...
        if settings.debug:
            try:
                print("[proxy] upstream payload (non-stream):", json.dumps({"url": chutes_url, **{k: v for k, v in oai_payload.items() if k != "messages"}, "model": oai_payload.get("model")}))
            except Exception:
                pass
        # Record resolved model (for /_debug/last), without logging sensitive payloads
        _rec["resolved_model"] = oai_payload.get("model")
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            try:
                resp = await client.post(
                    chutes_url, json=oai_payload, headers=_auth_headers(x_api_key, authorization)
                )
            except Exception as e:
                raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

        if resp.status_code >= 400:
            # Limited backoff/retry on 429 if configured
            if resp.status_code == 429 and settings.backoff_on_429 and settings.max_retry_on_429 > 0:
                try:
                    ra = float(resp.headers.get("Retry-After", "0"))
                except Exception:
                    ra = 0.0
                if 0.0 < ra <= settings.max_retry_after_seconds:
                    await asyncio.sleep(ra)
                    try:
                        resp = await client.post(chutes_url, json=oai_payload, headers=_auth_headers(x_api_key, authorization))
                    except Exception:
                        ...
            # Map upstream errors to Anthropic-like error envelope
            try:
                data = resp.json()
                message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data)
            except Exception:
                message = resp.text
            _rec.update({"phase": "non_stream_error", "upstream_status": resp.status_code, "message": message})
            _RECENT.append(_rec)
            return JSONResponse(status_code=resp.status_code, content=_anthropic_error_for_status(resp.status_code, message))

        data = resp.json()
        _rec["phase"] = "non_stream_ok"
        _RECENT.append(_rec)
        return JSONResponse(content=openai_to_anthropic_response(
            data,
            # Always echo original request model casing outward
            requested_model=display_model,
            tools=oai_payload.get("tools"),
            tool_call_parser=choose_tool_call_parser(oai_payload.get("model")),
        ))

    # Streaming path
    async def event_stream() -> AsyncIterator[bytes]:
        # Helper to format SSE
        def sse(event: str, data: Dict[str, Any]) -> bytes:
            return f"event: {event}\n".encode() + f"data: {json.dumps(data, ensure_ascii=False)}\n\n".encode()

        # Open connection to Chutes
        model_to_use = oai_payload.get("model")
        headers0 = _auth_headers(x_api_key, authorization)
        if settings.auto_fix_model_case and model_to_use:
            try:
                resolved_model = await _resolve_case_variant(model_to_use, headers0)
                if resolved_model and resolved_model != model_to_use:
                    if settings.debug:
                        print(f"[proxy] stream: auto-fix model case '{model_to_use}' -> '{resolved_model}'")
                    model_to_use = resolved_model
            except Exception:
                ...
        stream_payload = dict(oai_payload)
        # Ensure upstream uses provider's canonical model id; keep outward display via display_model
        if model_to_use:
            stream_payload["model"] = model_to_use
        _rec["resolved_model"] = model_to_use
        if settings.debug:
            try:
                print("[proxy] upstream payload (stream):", json.dumps({"url": chutes_url, **{k: v for k, v in stream_payload.items() if k != "messages"}, "model": model_to_use}))
            except Exception:
                pass
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    chutes_url,
                    json={**stream_payload, "model": model_to_use},
                    headers={**headers0, "Accept": "text/event-stream"},
                ) as upstream:
                    _rec["phase"] = f"stream_status_{upstream.status_code}"
                    # Anthropic stream scaffold
                    final_text: list[str] = []
                    sent_start = False
                    model_name = None
                    usage_input = 0
                    usage_output = 0

                    # Content block management
                    next_block_index = 0
                    text_block_index: int | None = None
                    tool_states: Dict[int, Dict[str, Any]] = {}

                    def ensure_message_start():
                        nonlocal sent_start
                        if not sent_start:
                            msg = {
                                "id": f"msg_{uuid.uuid4().hex}",
                                "type": "message",
                                "role": "assistant",
                                # Outward-facing: always echo the originally requested casing
                                "model": display_model or model_name or stream_payload.get("model", "unknown-model"),
                                "content": [{"type": "text", "text": ""}],
                            }
                            sent_start = True
                            return sse("message_start", {"type": "message_start", "message": msg})
                        return None

                    # Prepare sglang FunctionCallParser if tools exist
                    parser = None
                    parser_name = choose_tool_call_parser(model_to_use)
                    if (stream_payload.get("tools")):
                        try:
                            from sglang.srt.function_call.function_call_parser import FunctionCallParser as _FCP

                            class _Fn:
                                def __init__(self, name: str, parameters: Dict[str, Any], strict: bool = False):
                                    self.name = name
                                    self.parameters = parameters
                                    self.strict = strict

                            class _Tool:
                                def __init__(self, fn: _Fn):
                                    self.function = fn

                            _tool_objs = []
                            for t in stream_payload.get("tools") or []:
                                fn = (t or {}).get("function") or {}
                                name = fn.get("name") or ""
                                params = fn.get("parameters") or {}
                                strict = bool(fn.get("strict", False))
                                if name:
                                    _tool_objs.append(_Tool(_Fn(name, params, strict)))
                            if _tool_objs:
                                parser = _FCP(_tool_objs, parser_name)
                        except Exception:
                            parser = None

                    # Rely solely on sglang parser when available

                    async for line in upstream.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        if payload_str == "[DONE]":
                            # Close any open blocks
                            if text_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                                text_block_index = None
                            for st in list(tool_states.values()):
                                if st.get("open"):
                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": st["cb_index"]})
                                    st["open"] = False
                            # Final message_delta and stop
                            yield sse(
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": "end_turn"},
                                    "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                                },
                            )
                            yield sse("message_stop", {"type": "message_stop"})
                            _rec["phase"] = "stream_ok"
                            _RECENT.append(_rec)
                            break
                        try:
                            chunk = json.loads(payload_str)
                        except Exception:
                            continue

                        # Extract model
                        if not model_name:
                            # Prefer requested model for case preservation
                            model_name = stream_payload.get("model") or chunk.get("model")

                        # Accumulate text deltas
                        choices = chunk.get("choices") or []
                        if choices:
                            delta = (choices[0].get("delta") or {})
                            # Text piece
                            piece = delta.get("content") or ""
                            if piece:
                                if not sent_start:
                                    ms = ensure_message_start()
                                    if ms:
                                        yield ms
                                # Use sglang FunctionCallParser for textual tool calls when available
                                normal_piece = piece
                                if parser is not None:
                                    try:
                                        normal_piece, call_items = parser.parse_stream_chunk(piece)
                                        for it in call_items or []:
                                            try:
                                                import json as _json
                                                fargs = _json.loads(it.parameters) if isinstance(it.parameters, str) else {}
                                            except Exception:
                                                fargs = {}
                                            st = tool_states.get(it.name or f"idx_{len(tool_states)}")
                                            if not st:
                                                st = tool_states[it.name or f"idx_{len(tool_states)}"] = {
                                                    "open": False,
                                                    "cb_index": next_block_index,
                                                    "id": f"call_{uuid.uuid4().hex}",
                                                    "name": it.name or "",
                                                }
                                                cb_index = next_block_index
                                                next_block_index += 1
                                                yield sse(
                                                    "content_block_start",
                                                    {
                                                        "type": "content_block_start",
                                                        "index": cb_index,
                                                        "content_block": {
                                                            "type": "tool_use",
                                                            "id": st["id"],
                                                            "name": st["name"],
                                                            "input": {},
                                                        },
                                                    },
                                                )
                                                st["open"] = True
                                            if st["open"] and fargs:
                                                yield sse(
                                                    "content_block_delta",
                                                    {
                                                        "type": "content_block_delta",
                                                        "index": st["cb_index"],
                                                        "delta": {"type": "input_json_delta", "partial_json": _json.dumps(fargs)},
                                                    },
                                                )
                                    except Exception:
                                        normal_piece = piece

                                if normal_piece:
                                    # Start text block if not exists
                                    if text_block_index is None:
                                        text_block_index = next_block_index
                                        next_block_index += 1
                                        yield sse(
                                            "content_block_start",
                                            {
                                                "type": "content_block_start",
                                                "index": text_block_index,
                                                "content_block": {"type": "text", "text": ""},
                                            },
                                        )
                                    # Emit delta
                                    yield sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": text_block_index,
                                            "delta": {"type": "text_delta", "text": normal_piece},
                                        },
                                    )

                                # Note: No custom textual tool markup handling; rely on sglang parsing only

                            # Tool call deltas (support both delta.tool_calls and message.tool_calls)
                            tool_entries = delta.get("tool_calls") or ((choices[0].get("message") or {}).get("tool_calls") or [])
                            for tci in tool_entries:
                                idx = tci.get("index", 0)
                                st = tool_states.get(idx)
                                if st is None:
                                    # Initialize state but do not start until we have a name
                                    st = {
                                        "cb_index": None,
                                        "id": tci.get("id") or None,
                                        "name": None,
                                        "args": "",
                                        "open": False,
                                    }
                                    tool_states[idx] = st
                                # Capture id/name/args fragments
                                if tci.get("id"):
                                    st["id"] = tci.get("id")
                                fname = (tci.get("function") or {}).get("name")
                                if fname:
                                    st["name"] = (st.get("name") or "") + fname if not st.get("name") else st["name"]
                                fargs = (tci.get("function") or {}).get("arguments")
                                if fargs:
                                    st["args"] += fargs

                                # Start block only when we have a name
                                if not st["open"] and st.get("name"):
                                    if not sent_start:
                                        ms = ensure_message_start()
                                        if ms:
                                            yield ms
                                    cb_index = next_block_index
                                    next_block_index += 1
                                    st["cb_index"] = cb_index
                                    st["id"] = st.get("id") or f"call_{uuid.uuid4().hex}"
                                    yield sse(
                                        "content_block_start",
                                        {
                                            "type": "content_block_start",
                                            "index": cb_index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": st["id"],
                                                "name": st["name"],
                                                "input": {},
                                            },
                                        },
                                    )
                                    st["open"] = True

                                # Stream arguments if present and block is open
                                if st["open"] and fargs:
                                    yield sse(
                                        "content_block_delta",
                                        {
                                            "type": "content_block_delta",
                                            "index": st["cb_index"],
                                            "delta": {"type": "input_json_delta", "partial_json": fargs},
                                        },
                                    )

                        # Capture usage on final chunk when present
                        usage = chunk.get("usage")
                        if usage:
                            usage_input = int(usage.get("prompt_tokens") or 0)
                            usage_output = int(usage.get("completion_tokens") or 0)

                        # If finish_reason present, finalize early
                        if choices and choices[0].get("finish_reason"):
                            fr = choices[0].get("finish_reason")
                            # Close blocks
                            if text_block_index is not None:
                                yield sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                                text_block_index = None
                            for st in list(tool_states.values()):
                                if st.get("open"):
                                    yield sse("content_block_stop", {"type": "content_block_stop", "index": st["cb_index"]})
                                    st["open"] = False
                            # message_delta
                            yield sse(
                                "message_delta",
                                {
                                    "type": "message_delta",
                                    "delta": {"stop_reason": map_finish_reason(fr)},
                                    "usage": {"input_tokens": usage_input, "output_tokens": usage_output},
                                },
                            )
                            yield sse("message_stop", {"type": "message_stop"})
                            _rec["phase"] = "stream_ok_finish"
                            _RECENT.append(_rec)
                            break
            except Exception as e:
                # Surface upstream error as Anthropic error (single event)
                data = {"type": "error", "error": {"type": "upstream_error", "message": str(e)}}
                yield f"event: error\n".encode() + f"data: {json.dumps(data)}\n\n".encode()
                _rec["phase"] = "stream_exception"
                _rec["message"] = str(e)
                _RECENT.append(_rec)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/")
async def root():
    return {"ok": True, "backend": settings.chutes_base_url}


@app.get("/v1/models")
async def list_models(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    authorization: str | None = Header(default=None, alias="authorization"),
):
    url = f"{settings.chutes_base_url.rstrip('/')}/v1/models"
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        try:
            resp = await client.get(url, headers=_auth_headers(x_api_key, authorization))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")
    if resp.status_code >= 400:
        try:
            data = resp.json()
            message = data.get("error", {}).get("message") or data.get("message") or json.dumps(data)
        except Exception:
            message = resp.text
        return JSONResponse(status_code=resp.status_code, content=_anthropic_error_for_status(resp.status_code, message))
    return JSONResponse(content=resp.json())


@app.get("/_debug/last")
async def debug_last():
    return _RECENT[-1] if _RECENT else {}


@app.on_event("startup")
async def _startup_noop():
    return None
